/* Four behaviours: open and close the section menus in the top bar, render error responses
   into #results, report a dead server, and draw the plot that comes with a result. */

(function () {
  "use strict";

  // --- Section navigation ---------------------------------------------------
  // Plain DOM rather than Alpine: the bar is the only way around the app, so it must not
  // depend on a CDN that a locked-down network might drop. app.js is served from this app.

  function setupNavigation() {
    var nav = document.getElementById("site-nav");
    if (!nav) {
      return;
    }

    var buttons = Array.prototype.slice.call(nav.querySelectorAll(".nav-section-button"));

    function panelOf(button) {
      return document.getElementById(button.getAttribute("aria-controls"));
    }

    function setOpen(button, open) {
      var panel = panelOf(button);
      button.setAttribute("aria-expanded", open ? "true" : "false");
      if (panel) {
        panel.classList.toggle("is-open", open);
      }
    }

    function closeAll(except) {
      buttons.forEach(function (button) {
        if (button !== except) {
          setOpen(button, false);
        }
      });
    }

    buttons.forEach(function (button) {
      setOpen(button, false);

      button.addEventListener("click", function () {
        var open = button.getAttribute("aria-expanded") === "true";
        closeAll(button);
        setOpen(button, !open);
      });

      // Down arrow on the button walks into the menu, the way a menu bar is expected to.
      button.addEventListener("keydown", function (event) {
        if (event.key !== "ArrowDown") {
          return;
        }
        event.preventDefault();
        closeAll(button);
        setOpen(button, true);
        var first = panelOf(button) && panelOf(button).querySelector(".nav-link");
        if (first) {
          first.focus();
        }
      });
    });

    // Escape closes the open menu and hands focus back to the button that owns it.
    nav.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") {
        return;
      }
      var open = buttons.filter(function (button) {
        return button.getAttribute("aria-expanded") === "true";
      });
      if (!open.length) {
        return;
      }
      open.forEach(function (button) {
        setOpen(button, false);
      });
      open[0].focus();
    });

    // Tabbing or clicking away closes what you left behind.
    document.addEventListener("focusin", function (event) {
      buttons.forEach(function (button) {
        var item = button.closest("[data-nav-item]");
        if (item && !item.contains(event.target)) {
          setOpen(button, false);
        }
      });
    });

    document.addEventListener("click", function (event) {
      if (!nav.contains(event.target)) {
        closeAll(null);
      }
    });

    var toggle = document.querySelector("[data-nav-toggle]");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var open = nav.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        if (!open) {
          closeAll(null);
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupNavigation);
  } else {
    setupNavigation();
  }

  // --- Results --------------------------------------------------------------

  // The identity lives in app.css. Read the shared tokens from there so a palette edit
  // reaches the plot too; the literals are only a fallback if the stylesheet is missing.
  var FALLBACK = {
    "--accent": "#1b4fd8",
    "--ink": "#14232e",
    "--grid-major": "#d3dbd3",
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
