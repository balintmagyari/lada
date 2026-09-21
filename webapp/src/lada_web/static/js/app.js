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

    // The series keep full precision; only the hover readout is rounded, so that a point
    // reads the same on the plot as it does in the table.
    var axis = {
      gridcolor: token("--grid-major"),
      zerolinecolor: token("--rule"),
      hoverformat: ".6g"
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
        zerolinecolor: axis.zerolinecolor,
        hoverformat: axis.hoverformat
      },
      yaxis: {
        title: { text: plot.y_label || plot.y.join(", ") },
        type: axisType(plot.y_log),
        gridcolor: axis.gridcolor,
        zerolinecolor: axis.zerolinecolor,
        hoverformat: axis.hoverformat
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

  // --- Multi-regime orchestration --------------------------------------------
  //
  // A page opts in by including partials/_regimes.html, whose "Calculate all regimes" button
  // carries the endpoints and field names to use as data-* attributes (see that file's header
  // comment). Regime 0 is the page's own fields; extra regimes are indexed 1, 2, ... with
  // `regime_<i>_<field>` names. Each regime is still one ordinary synchronous request to the
  // page's per-regime endpoint — no background job/polling is involved here — followed by one
  // request to a merge endpoint that combines the resulting series.

  function copyNamedValue(form, name, target) {
    var inputs = form.querySelectorAll('[name="' + name + '"]');
    Array.prototype.forEach.call(inputs, function (input) {
      if (input.type === "file") {
        Array.prototype.forEach.call(input.files, function (file) {
          target.append(name, file);
        });
      } else if (input.type === "checkbox" || input.type === "radio") {
        if (input.checked) {
          target.append(name, input.value);
        }
      } else {
        target.append(name, input.value);
      }
    });
  }

  function regimeFormData(form, regimeIndex, regimeFields, sharedFields) {
    var data = new FormData();
    regimeFields.forEach(function (field) {
      var sourceName = regimeIndex === 0 ? field : "regime_" + regimeIndex + "_" + field;
      var temp = new FormData();
      copyNamedValue(form, sourceName, temp);
      temp.getAll(sourceName).forEach(function (value) {
        data.append(field, value);
      });
    });
    sharedFields.forEach(function (field) {
      copyNamedValue(form, field, data);
    });
    return data;
  }

  // --- Client-side result rendering (job results arrive as JSON, not a pre-rendered partial) --
  //
  // A regime-merge result is rendered by asking the server for the htmx partial directly (see
  // ladaRunRegimes below): the merge endpoint is a real Jinja-templated route. A job's result
  // has no equivalent "ask again as HTML" step — /api/jobs/{id} is deliberately one generic,
  // JSON-only endpoint shared by every domain (see routers/jobs.py) — so this mirrors
  // templates/partials/table_result.html closely enough to look and behave the same.

  var SIGNIFICANT_FIGURES = 6;
  var EXACT_INTEGER_LIMIT = Math.pow(2, 53);

  function sigfig(value) {
    if (value === null || value === undefined) {
      return "";
    }
    if (typeof value === "number") {
      if (!isFinite(value)) {
        return String(value);
      }
      if (Number.isInteger(value) && Math.abs(value) < EXACT_INTEGER_LIMIT) {
        return String(value);
      }
      var rounded = Number(value.toPrecision(SIGNIFICANT_FIGURES));
      return String(rounded);
    }
    return String(value);
  }

  // Safe for text AND quoted attribute values. (Serializing a text node through innerHTML would
  // not do: it leaves quotes as they are, so JSON placed in data-result="..." ended the attribute
  // at its first quote and JSON.parse then saw only "{".)
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderTableResultHtml(result, downloadUrl) {
    var summaryParts = [
      result.n_rows_total + " " + (result.n_rows_total === 1 ? "row" : "rows") + ", ",
      result.columns.length + " " + (result.columns.length === 1 ? "column" : "columns") + "."
    ];
    Object.keys(result.meta || {}).forEach(function (key) {
      var label = key.replace(/_/g, " ");
      label = label.charAt(0).toUpperCase() + label.slice(1);
      summaryParts.push(" " + escapeHtml(label) + ": " + escapeHtml(result.meta[key]) + ".");
    });

    var plotHtml = "";
    if (result.plot) {
      plotHtml =
        '<div class="plot-controls">' +
        '<label><input type="checkbox" data-axis="xaxis"' +
        (result.plot.x_log ? " checked" : "") +
        "> Log x</label>" +
        '<label><input type="checkbox" data-axis="yaxis"' +
        (result.plot.y_log ? " checked" : "") +
        "> Log y</label>" +
        "</div>" +
        '<div class="plot" id="result-plot"></div>';
    }

    var headerCells = result.columns
      .map(function (c) {
        return '<th scope="col">' + escapeHtml(c) + "</th>";
      })
      .join("");
    var bodyRows = result.rows
      .map(function (row) {
        var cells = row
          .map(function (v) {
            return "<td>" + escapeHtml(sigfig(v)) + "</td>";
          })
          .join("");
        return "<tr>" + cells + "</tr>";
      })
      .join("");

    var note = result.truncated
      ? "Showing the first " +
        result.rows.length +
        " of " +
        result.n_rows_total +
        " rows. Values are shown to " +
        SIGNIFICANT_FIGURES +
        " significant figures; downloads carry every row at full precision."
      : "Values are shown to " +
        SIGNIFICANT_FIGURES +
        " significant figures; downloads carry full precision.";

    var downloadsHtml = "";
    if (downloadUrl) {
      downloadsHtml =
        '<div class="downloads">' +
        '<button type="button" onclick="window.ladaDownloadJobResult(this, \'csv\')" ' +
        'data-download-url="' +
        escapeHtml(downloadUrl) +
        '">Download CSV</button>' +
        '<button type="button" onclick="window.ladaDownloadJobResult(this, \'pgfplots\')" ' +
        'data-download-url="' +
        escapeHtml(downloadUrl) +
        '">Download pgfplots table</button>' +
        "</div>";
    }

    var resultJson = escapeHtml(JSON.stringify(result));
    return (
      '<section class="result" id="result" data-result="' +
      resultJson +
      '">' +
      '<header class="result-header"><h2>' +
      escapeHtml(result.title) +
      '</h2><p class="result-summary">' +
      summaryParts.join("") +
      "</p></header>" +
      plotHtml +
      '<div class="table-scroll" tabindex="0" role="region" aria-label="' +
      escapeHtml(result.title) +
      ' table"><table class="numbers"><thead><tr>' +
      headerCells +
      "</tr></thead><tbody>" +
      bodyRows +
      "</tbody></table></div>" +
      '<p class="result-note">' +
      note +
      "</p>" +
      downloadsHtml +
      "</section>"
    );
  }

  // window.ladaDownloadJobResult is set by ladaRunJob per result, since a job result has no
  // upload form to resubmit for a download the way the synchronous pages do — it re-POSTs the
  // exact same form data used to start the job, to the same domain's ordinary (non-job)
  // download route, which already knows how to turn that into a file.

  function readValue(form, name) {
    var input = form.querySelector('[name="' + name + '"]');
    if (!input) {
      return null;
    }
    return input.value === "" ? null : input.value;
  }

  function readNumber(form, name) {
    var value = readValue(form, name);
    return value === null ? null : Number(value);
  }

  function renderRegimeProgress(message) {
    var results = document.getElementById("results");
    if (results) {
      results.innerHTML = '<p class="results-empty">' + message + "</p>";
    }
  }

  function renderRegimeError(message) {
    var results = document.getElementById("results");
    if (results) {
      results.innerHTML =
        '<section class="result result-error" id="result" role="alert">' +
        "<h2>That didn't work</h2><p></p></section>";
      results.querySelector("p").textContent = message;
    }
  }

  function dynamicsMergeBody(button, regimeResults) {
    var valueColumn = button.dataset.valueColumn;
    return {
      regimes: regimeResults.map(function (result, index) {
        return {
          label: "regime " + (index + 1),
          lag_time: result.plot.series.lag_time,
          value: result.plot.series[valueColumn]
        };
      }),
      value_column: valueColumn,
      title: button.dataset.title,
      y_label: button.dataset.yLabel,
      download_stem: button.dataset.downloadStem
    };
  }

  function rheologyGtMergeBody(regimeResults, downloadStem) {
    return {
      regimes: regimeResults.map(function (result, index) {
        return {
          label: "regime " + (index + 1),
          lag_time: result.plot.series.lag_time,
          G_GK: result.plot.series.G_GK,
          G_FSR: result.plot.series.G_FSR
        };
      }),
      download_stem: downloadStem
    };
  }

  function rheologyModuliMergeBody(form, regimeResults, downloadStem) {
    var body = rheologyGtMergeBody(regimeResults, downloadStem);
    body.method = readValue(form, "method") || "GK";
    body.t_min = readNumber(form, "t_min");
    body.t_cutoff = readNumber(form, "t_cutoff");
    body.n_modes = readNumber(form, "n_modes");
    body.n_omega = readNumber(form, "n_omega");
    body.omega_min = readNumber(form, "omega_min");
    body.omega_max = readNumber(form, "omega_max");
    return body;
  }

  // FastAPI answers a field-constraint failure with `detail` as a LIST of error objects, and the
  // app's own friendly-string formatter only runs for htmx/HTML requests — these fetch() calls
  // send neither header. String(list-of-objects) renders as "[object Object]", so unpack the list
  // here into "field: message", the way the server's HTML branch already does.
  function detailMessage(body, fallback) {
    var detail = body && body.detail;
    if (!detail) {
      return fallback;
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      var parts = detail
        .map(function (item) {
          if (!item || typeof item !== "object") {
            return String(item);
          }
          var where = Array.isArray(item.loc)
            ? item.loc.filter(function (piece) {
                return piece !== "body" && piece !== "query";
              })
            : [];
          var message = item.msg || "is not valid";
          return where.length ? where.join(" ") + ": " + message : message;
        })
        .filter(Boolean);
      if (parts.length) {
        return parts.join("; ");
      }
    }
    return fallback;
  }

  async function startJob(url, formData) {
    var response = await fetch(url, { method: "POST", body: formData });
    var body = await response.json();
    if (!response.ok) {
      throw new Error(detailMessage(body, "Could not start the calculation."));
    }
    return body.job_id;
  }

  async function pollJob(jobId, onProgress) {
    for (;;) {
      var response = await fetch("/api/jobs/" + jobId);
      var body = await response.json();
      if (!response.ok) {
        throw new Error(detailMessage(body, "Lost track of the job."));
      }
      if (onProgress) {
        onProgress(body.progress || 0, body.message || "");
      }
      if (body.status === "done") {
        return body.result;
      }
      if (body.status === "error") {
        throw new Error(body.error || "The calculation failed.");
      }
      await new Promise(function (resolve) {
        setTimeout(resolve, 500);
      });
    }
  }

  function setProgressBar(fraction, message) {
    var wrap = document.getElementById("job-progress");
    var bar = document.getElementById("job-progress-bar");
    var label = document.getElementById("job-progress-label");
    if (wrap) {
      wrap.hidden = false;
    }
    if (bar) {
      bar.value = Math.round((fraction || 0) * 100);
    }
    if (label) {
      label.textContent = message || "Calculating…";
    }
  }

  function hideProgressBar() {
    var wrap = document.getElementById("job-progress");
    if (wrap) {
      wrap.hidden = true;
    }
  }

  window.ladaRunJob = async function (button) {
    var form = button.closest("form");
    var startUrl = button.dataset.startUrl;
    var downloadUrl = startUrl.replace(/\/start$/, "/download");
    var results = document.getElementById("results");
    button.disabled = true;
    setProgressBar(0, "Starting…");
    try {
      var result = await pollJob(
        await startJob(startUrl, new FormData(form)),
        function (fraction, message) {
          setProgressBar(fraction, message || "Calculating…");
        }
      );
      if (results) {
        results.innerHTML = renderTableResultHtml(result, downloadUrl);
        window.ladaMountResult(document.getElementById("result"));
      }
    } catch (error) {
      renderRegimeError(error && error.message ? error.message : String(error));
    } finally {
      button.disabled = false;
      hideProgressBar();
    }
  };

  window.ladaDownloadJobResult = async function (button, fmt) {
    var form = button.closest("form") || document.getElementById("calc-form");
    var downloadUrl = button.dataset.downloadUrl;
    if (!form || !downloadUrl) {
      return;
    }
    button.disabled = true;
    try {
      var response = await fetch(downloadUrl + "?fmt=" + fmt, {
        method: "POST",
        body: new FormData(form)
      });
      if (!response.ok) {
        var body = await response.json().catch(function () {
          return null;
        });
        throw new Error(detailMessage(body, "The download failed."));
      }
      var blob = await response.blob();
      var disposition = response.headers.get("content-disposition") || "";
      var match = /filename="?([^"]+)"?/.exec(disposition);
      var url = URL.createObjectURL(blob);
      var link = document.createElement("a");
      link.href = url;
      link.download = match ? match[1] : "download";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      renderRegimeError(error && error.message ? error.message : String(error));
    } finally {
      button.disabled = false;
    }
  };

  window.ladaRunRegimes = async function (button) {
    var form = button.closest("form");
    var count = parseInt(button.dataset.regimeCount, 10) || 1;
    var regimeFields = button.dataset.regimeFields.split(",");
    var sharedFields = button.dataset.sharedFields
      ? button.dataset.sharedFields.split(",")
      : [];
    var viewUrl = button.dataset.viewUrl;
    var mergeUrl = button.dataset.mergeUrl;
    var mergeKind = button.dataset.mergeKind;
    var downloadStem = button.dataset.downloadStem || "regimes";

    button.disabled = true;
    try {
      var regimeResults = [];
      for (var i = 0; i < count; i++) {
        var data = regimeFormData(form, i, regimeFields, sharedFields);
        var result = await pollJob(
          await startJob(viewUrl + "/start", data),
          function (fraction, message) {
            var prefix = "Regime " + (i + 1) + " of " + count + ": ";
            renderRegimeProgress(prefix + (message || Math.round((fraction || 0) * 100) + "%"));
          }
        );
        regimeResults.push(result);
      }

      renderRegimeProgress("Combining " + count + " regimes…");
      var mergeBody;
      if (mergeKind === "rheology-gt") {
        mergeBody = rheologyGtMergeBody(regimeResults, downloadStem);
      } else if (mergeKind === "rheology-moduli") {
        mergeBody = rheologyModuliMergeBody(form, regimeResults, downloadStem);
      } else {
        mergeBody = dynamicsMergeBody(button, regimeResults);
      }

      var mergeResponse = await fetch(mergeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "HX-Request": "true" },
        body: JSON.stringify(mergeBody)
      });
      if (!mergeResponse.ok) {
        var errorBody = await mergeResponse.json().catch(function () {
          return null;
        });
        throw new Error(
          detailMessage(errorBody, "Merging the regimes failed.")
        );
      }
      var html = await mergeResponse.text();
      var results = document.getElementById("results");
      if (results) {
        results.innerHTML = html;
        // The injected partial's own <script> tag does not execute via innerHTML, so the plot
        // is mounted explicitly here instead of relying on it.
        window.ladaMountResult(document.getElementById("result"));
      }
    } catch (error) {
      renderRegimeError(error && error.message ? error.message : String(error));
    } finally {
      button.disabled = false;
    }
  };

  // Exposed for tests/test_frontend_helpers.py. Both are pure string functions, and both have
  // had a bug that reached a user: escapeHtml left quotes alone, which broke the attribute
  // holding a job's result JSON, and detailMessage's predecessor printed "[object Object]" for
  // every field-validation failure. Neither is reachable by a Python-level route test.
  window.ladaTestHooks = { escapeHtml: escapeHtml, detailMessage: detailMessage };
})();
