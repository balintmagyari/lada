/* Three behaviours: render error responses into #results, report a dead server,
   and draw the plot that comes with a result. */

(function () {
  "use strict";

  // The identity lives in app.css. Read the shared tokens from there so a palette edit
  // reaches the plot too; the literals are only a fallback if the stylesheet is missing.
  var FALLBACK = {
    "--accent": "#1b4fd8",
    "--ink": "#14232e",
    "--grid-major": "#dbe1da",
    "--rule": "#c6d0c9"
  };

  var palette = null;

  function token(name) {
    if (!palette) {
      palette = getComputedStyle(document.documentElement);
    }
    var value = (palette.getPropertyValue(name) || "").trim();
    return value || FALLBACK[name];
  }

  // Only the first series is part of the identity; the rest keep the plot readable.
  var COMPANIONS = ["#b4661b", "#2e7d5b", "#8c2f6b", "#4a6572"];

  function seriesColor(index) {
    return index === 0 ? token("--accent") : COMPANIONS[(index - 1) % COMPANIONS.length];
  }

  // Error responses carry the partial to display, so let htmx swap them in.
  document.body.addEventListener("htmx:beforeSwap", function (evt) {
    if (evt.detail.xhr && evt.detail.xhr.status >= 400) {
      evt.detail.shouldSwap = true;
      evt.detail.isError = false;
    }
  });

  document.body.addEventListener("htmx:sendError", function () {
    var results = document.getElementById("results");
    if (results) {
      results.innerHTML =
        '<section class="result result-error" id="result" role="alert">' +
        "<h2>The server could not be reached</h2>" +
        "<p>Check that the app is still running, then try again.</p>" +
        "</section>";
    }
  });

  function axisType(isLog) {
    return isLog ? "log" : "linear";
  }

  window.ladaMountResult = function (section) {
    if (!section || !section.dataset.result) {
      return;
    }
    var result = JSON.parse(section.dataset.result);
    var plot = result.plot;
    var target = section.querySelector("#result-plot");
    if (!plot || !target || typeof Plotly === "undefined") {
      return;
    }

    var traces = plot.y.map(function (name, index) {
      return {
        x: plot.series[plot.x],
        y: plot.series[name],
        name: name,
        type: "scatter",
        mode: "lines+markers",
        marker: { size: 5, color: seriesColor(index) },
        line: { width: 2, color: seriesColor(index) }
      };
    });

    var axis = {
      gridcolor: token("--grid-major"),
      zerolinecolor: token("--rule")
    };

    var layout = {
      margin: { l: 64, r: 16, t: 16, b: 52 },
      showlegend: plot.y.length > 1,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: token("--ink") },
      xaxis: {
        title: { text: plot.x_label || plot.x },
        type: axisType(plot.x_log),
        gridcolor: axis.gridcolor,
        zerolinecolor: axis.zerolinecolor
      },
      yaxis: {
        title: { text: plot.y_label || plot.y.join(", ") },
        type: axisType(plot.y_log),
        gridcolor: axis.gridcolor,
        zerolinecolor: axis.zerolinecolor
      }
    };

    Plotly.newPlot(target, traces, layout, { responsive: true, displaylogo: false });

    var controls = section.querySelectorAll(".plot-controls input[data-axis]");
    Array.prototype.forEach.call(controls, function (input) {
      input.addEventListener("change", function () {
        var update = {};
        update[input.dataset.axis + ".type"] = axisType(input.checked);
        Plotly.relayout(target, update);
      });
    });
  };
})();
