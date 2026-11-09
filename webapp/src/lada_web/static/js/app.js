/* Four behaviours: the sidebar's quick-jump box and recently used tools, render error responses
   into #results, report a dead server, and draw the plot that comes with a result. */

(function () {
  "use strict";

  // --- Navigation -------------------------------------------------------------
  // Plain DOM rather than Alpine: the sidebar is the only way around the app, so it must not
  // depend on a CDN that a locked-down network might drop. app.js is served from this app.

  var RECENT_KEY = "lada-recent";
  var RECENT_MAX = 5;

  function readRecent() {
    try {
      var list = JSON.parse(window.localStorage.getItem(RECENT_KEY) || "[]");
      return Array.isArray(list) ? list : [];
    } catch (e) {
      return [];
    }
  }

  function writeRecent(list) {
    try {
      window.localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    } catch (e) {
      /* Private window or blocked storage: the list is a convenience, so skip it. */
    }
  }

  // Every tool, read back from the sidebar so there is one source of truth.
  function collectTools(nav) {
    var tools = [];
    nav.querySelectorAll(".nav-group").forEach(function (group) {
      var title = group.querySelector(".nav-group-title");
      var sectionTitle = title ? title.textContent.trim() : "";
      var workspace = group.closest(".nav-workspace");
      group.querySelectorAll(".nav-link").forEach(function (link) {
        tools.push({
          path: link.getAttribute("href"),
          title: link.textContent.trim(),
          section: group.dataset.section,
          sectionTitle: sectionTitle,
          workspace: workspace ? workspace.dataset.workspace : "",
          workspaceTitle: workspace ? workspace.dataset.workspaceTitle : ""
        });
      });
    });
    return tools;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text) {
      node.textContent = text;
    }
    return node;
  }

  function beadFor(section) {
    var bead = el("span", "bead");
    bead.setAttribute("aria-hidden", "true");
    bead.dataset.section = section;
    return bead;
  }

  function setupMobileToggle(nav) {
    var toggle = document.querySelector("[data-nav-toggle]");
    if (!toggle) {
      return;
    }
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // Remember the tool this page is, and list the latest few on the overview.
  function setupRecent(tools) {
    var here = window.location.pathname;
    var current = tools.filter(function (tool) {
      return tool.path === here;
    })[0];
    var list = readRecent().filter(function (path) {
      return path !== here;
    });
    if (current) {
      list.unshift(here);
      writeRecent(list.slice(0, RECENT_MAX));
      return;
    }

    var holder = document.getElementById("recent");
    if (!holder) {
      return;
    }
    var only = holder.dataset.recentWorkspace;
    var known = list
      .map(function (path) {
        return tools.filter(function (tool) {
          return tool.path === path;
        })[0];
      })
      .filter(function (tool) {
        return tool && (!only || tool.workspace === only);
      });
    if (!known.length) {
      return;
    }
    var ul = holder.querySelector(".recent-list");
    known.forEach(function (tool) {
      var li = el("li");
      var a = el("a");
      a.href = tool.path;
      a.dataset.section = tool.section;
      a.appendChild(beadFor(tool.section));
      a.appendChild(document.createTextNode(tool.title));
      li.appendChild(a);
      ul.appendChild(li);
    });
    holder.hidden = false;
  }

  // "/" or Ctrl/Cmd+K opens a box; type part of a name, Enter goes there.
  function setupJump(tools) {
    var dialog = document.getElementById("jump");
    var input = document.getElementById("jump-input");
    var list = document.getElementById("jump-list");
    if (!dialog || !input || !list || typeof dialog.showModal !== "function") {
      document.querySelectorAll("[data-jump-open]").forEach(function (b) {
        b.hidden = true;
      });
      return;
    }

    var matches = [];
    var active = 0;

    function score(tool, words) {
      var hay = (tool.title + " " + tool.sectionTitle + " " + tool.workspaceTitle).toLowerCase();
      var title = tool.title.toLowerCase();
      var total = 0;
      for (var i = 0; i < words.length; i++) {
        var at = hay.indexOf(words[i]);
        if (at < 0) {
          return -1;
        }
        total += title.indexOf(words[i]) === 0 ? 0 : at < title.length ? 1 : 2;
      }
      return total;
    }

    function render() {
      var words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var recent = readRecent();
      matches = tools
        .map(function (tool, order) {
          var s = words.length ? score(tool, words) : 0;
          var r = recent.indexOf(tool.path);
          // With nothing typed, the tools you used last come first.
          return { tool: tool, s: s, r: r < 0 ? 99 : r, order: order };
        })
        .filter(function (m) {
          return m.s >= 0;
        })
        .sort(function (a, b) {
          return a.s - b.s || a.r - b.r || a.order - b.order;
        })
        .map(function (m) {
          return m.tool;
        });
      active = Math.min(active, Math.max(matches.length - 1, 0));

      list.textContent = "";
      if (!matches.length) {
        list.appendChild(el("li", "jump-empty", "No tool matches that name."));
        input.removeAttribute("aria-activedescendant");
        return;
      }
      matches.forEach(function (tool, index) {
        var li = el("li");
        var option = el("a", "jump-option");
        option.id = "jump-option-" + index;
        option.href = tool.path;
        option.setAttribute("role", "option");
        option.setAttribute("aria-selected", index === active ? "true" : "false");
        option.dataset.section = tool.section;
        option.appendChild(beadFor(tool.section));
        option.appendChild(el("span", "", tool.title));
        option.appendChild(
          el("span", "jump-option-section", tool.workspaceTitle + " / " + tool.sectionTitle)
        );
        option.addEventListener("mousemove", function () {
          if (active !== index) {
            active = index;
            render();
          }
        });
        li.appendChild(option);
        list.appendChild(li);
      });
      input.setAttribute("aria-activedescendant", "jump-option-" + active);
      var chosen = document.getElementById("jump-option-" + active);
      if (chosen && chosen.scrollIntoView) {
        chosen.scrollIntoView({ block: "nearest" });
      }
    }

    function open() {
      if (dialog.open) {
        return;
      }
      input.value = "";
      active = 0;
      render();
      dialog.showModal();
      input.focus();
    }

    input.addEventListener("input", function () {
      active = 0;
      render();
    });

    input.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (matches.length) {
          var step = event.key === "ArrowDown" ? 1 : -1;
          active = (active + step + matches.length) % matches.length;
          render();
        }
      } else if (event.key === "Enter") {
        event.preventDefault();
        if (matches[active]) {
          window.location.href = matches[active].path;
        }
      }
    });

    // A click on the dimmed backdrop (the dialog itself, outside the box) closes it.
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) {
        dialog.close();
      }
    });

    document.querySelectorAll("[data-jump-open]").forEach(function (button) {
      button.addEventListener("click", open);
    });

    document.addEventListener("keydown", function (event) {
      var target = event.target;
      var typing =
        target &&
        (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
      var isSlash = event.key === "/" && !typing && !event.metaKey && !event.ctrlKey;
      var isK = (event.key === "k" || event.key === "K") && (event.metaKey || event.ctrlKey);
      if (isSlash || isK) {
        event.preventDefault();
        open();
      }
    });
  }

  // --- Colour theme -----------------------------------------------------------
  // The inline script in base.html has already applied the theme before first paint; this
  // wires up the switch and follows the system while the choice is "auto".

  var THEME_KEY = "lada-theme";
  var THEME_ORDER = ["auto", "light", "dark"];
  var THEME_LABEL = { auto: "Auto", light: "Light", dark: "Dark" };

  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function applyTheme(pref) {
    var root = document.documentElement;
    var dark = pref === "dark" || (pref === "auto" && systemDark());
    var before = root.getAttribute("data-theme");
    root.setAttribute("data-theme", dark ? "dark" : "light");
    root.setAttribute("data-theme-pref", pref);
    var toggle = document.querySelector("[data-theme-toggle]");
    if (toggle) {
      toggle.querySelector(".theme-label").textContent = THEME_LABEL[pref];
      toggle.setAttribute(
        "aria-label",
        "Colour theme: " + THEME_LABEL[pref] + ". Activate to change."
      );
    }
    if (before !== root.getAttribute("data-theme") && window.ladaRepaintPlots) {
      window.ladaRepaintPlots();
    }
  }

  function setupTheme() {
    var toggle = document.querySelector("[data-theme-toggle]");
    var pref = document.documentElement.getAttribute("data-theme-pref") || "auto";
    if (THEME_ORDER.indexOf(pref) < 0) {
      pref = "auto";
    }
    applyTheme(pref);
    if (toggle) {
      toggle.hidden = false;
      toggle.addEventListener("click", function () {
        pref = THEME_ORDER[(THEME_ORDER.indexOf(pref) + 1) % THEME_ORDER.length];
        try {
          window.localStorage.setItem(THEME_KEY, pref);
        } catch (e) {
          /* Blocked storage: the choice lasts for this page only. */
        }
        applyTheme(pref);
      });
    }
    if (window.matchMedia) {
      var query = window.matchMedia("(prefers-color-scheme: dark)");
      var follow = function () {
        if (pref === "auto") {
          applyTheme("auto");
        }
      };
      if (query.addEventListener) {
        query.addEventListener("change", follow);
      } else if (query.addListener) {
        query.addListener(follow);
      }
    }
  }

  // --- Workspaces ---------------------------------------------------------------
  // The switch only changes which workspace's tools the sidebar lists. A tool page always
  // opens on its own workspace; the home page opens on the one this browser used last.

  var WORKSPACE_KEY = "lada-workspace";

  function setupWorkspaces(nav) {
    var buttons = Array.prototype.slice.call(
      document.querySelectorAll("[data-workspace-switch]")
    );
    if (!buttons.length) {
      return;
    }

    function show(key) {
      nav.querySelectorAll(".nav-workspace").forEach(function (group) {
        group.classList.toggle("is-active", group.dataset.workspace === key);
      });
      buttons.forEach(function (button) {
        button.setAttribute(
          "aria-pressed",
          button.dataset.workspaceSwitch === key ? "true" : "false"
        );
      });
    }

    var here = document.body.dataset.workspace;
    if (here) {
      try {
        window.localStorage.setItem(WORKSPACE_KEY, here);
      } catch (e) {
        /* Blocked storage: the home page simply opens on the first workspace. */
      }
    } else {
      var last = null;
      try {
        last = window.localStorage.getItem(WORKSPACE_KEY);
      } catch (e) {
        last = null;
      }
      if (last && nav.querySelector('.nav-workspace[data-workspace="' + last + '"]')) {
        show(last);
      }
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        show(button.dataset.workspaceSwitch);
      });
    });
  }

  function setupNavigation() {
    setupTheme();
    var nav = document.getElementById("site-nav");
    if (!nav) {
      return;
    }
    var tools = collectTools(nav);
    setupWorkspaces(nav);
    setupMobileToggle(nav);
    setupRecent(tools);
    setupJump(tools);
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
    "--accent": "#2f4fae",
    "--ink": "#1e2832",
    "--ink-muted": "#5e6b77",
    "--line": "#e1e5ea",
    "--line-strong": "#c9d0d7",
    "--font-sans": "system-ui, sans-serif"
  };

  // Read from <body>, freshly each time: that is where the page's section sets --accent, and
  // the values change when the theme does.
  function token(name) {
    var value = (getComputedStyle(document.body).getPropertyValue(name) || "").trim();
    return value || FALLBACK[name];
  }

  // The first series takes the section's colour and the second the text colour, so a pair
  // (G_GK / G_FSR, G' / G'') reads in either theme; the rest are mid-tones that work on both
  // grounds and are not mistaken for another section.
  var COMPANIONS = ["#8a96a1", "#3a8fb7", "#b08850"];

  function seriesColor(index) {
    if (index === 0) {
      return token("--accent");
    }
    if (index === 1) {
      return token("--ink");
    }
    return COMPANIONS[(index - 2) % COMPANIONS.length];
  }

  function themeLayout() {
    return {
      "font.color": token("--ink"),
      "xaxis.gridcolor": token("--line"),
      "yaxis.gridcolor": token("--line"),
      "xaxis.zerolinecolor": token("--line-strong"),
      "yaxis.zerolinecolor": token("--line-strong"),
      "legend.font.color": token("--ink")
    };
  }

  // Called when the theme switches: recolour every plot on the page in place, keeping its
  // data, zoom and log-axis state.
  window.ladaRepaintPlots = function () {
    if (typeof Plotly === "undefined") {
      return;
    }
    document.querySelectorAll(".plot").forEach(function (target) {
      if (!target.data || !target.data.length) {
        return;
      }
      var colors = target.data.map(function (_, index) {
        return seriesColor(index);
      });
      Plotly.update(
        target,
        { "marker.color": colors, "line.color": colors },
        themeLayout()
      );
    });
  };

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
      gridcolor: token("--line"),
      zerolinecolor: token("--line-strong"),
      hoverformat: ".6g"
    };

    var layout = {
      margin: { l: 64, r: 16, t: 16, b: 52 },
      showlegend: plot.y.length > 1,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: token("--ink"), family: token("--font-sans"), size: 12 },
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
  // comment). Regime 0 is the page's own fields; each extra regime's inputs are named
  // `regime_<key>_<field>`, where <key> is the stable key the partial gave that block (its
  // data-regime-key), not its position: removing a middle regime leaves a gap in the keys. Each regime is still one ordinary synchronous request to the
  // page's per-regime endpoint — no background job/polling is involved here — followed by one
  // request to a merge endpoint that combines the resulting series.

  function copyNamedValue(form, name, target) {
    var inputs = form.querySelectorAll('[name="' + name + '"]');
    Array.prototype.forEach.call(inputs, function (input) {
      // Skip what the form itself would skip. The topology panes disable the inputs of whichever
      // mode is not selected and new FormData(form) honours that; without this line a regime
      // request would carry a data file that was chosen and then switched away from.
      if (input.disabled) {
        return;
      }
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

  // The field-name keys of every regime, in page order: null for the page's own fields, then
  // each extra block's data-regime-key. Counting blocks and numbering them 1..n broke as soon
  // as a middle regime was removed, because the remaining blocks keep their original keys.
  function regimeKeys(button) {
    var fieldset = button.closest("fieldset.regimes");
    var blocks = fieldset ? fieldset.querySelectorAll("[data-regime-key]") : [];
    return [null].concat(
      Array.prototype.map.call(blocks, function (block) {
        return block.dataset.regimeKey;
      })
    );
  }

  function regimeFormData(form, regimeKey, regimeFields, sharedFields) {
    var data = new FormData();
    regimeFields.forEach(function (field) {
      var sourceName = regimeKey === null ? field : "regime_" + regimeKey + "_" + field;
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
    var keys = regimeKeys(button);
    var count = keys.length;
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
        var data = regimeFormData(form, keys[i], regimeFields, sharedFields);
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

  // --- Chain topology read from a data file ----------------------------------
  //
  // Fired when a file is chosen on the segment-ACF or Rouse-mode page, so a wrong file is caught
  // before a calculation that can run for minutes. `state` is the Alpine data object of
  // templates/dynamics/_topology.html; writing to it is what updates the panel.
  //
  // It deliberately does not check the file against the trajectory: that would mean uploading
  // the .npz twice, once to look at and once to compute with. The atom counts are compared
  // server-side when the calculation runs.

  var TOPOLOGY_PREVIEW_URL = "/api/topology/preview";
  var TOPOLOGY_UNREADABLE = "This data file could not be read.";

  function readTopology(event, state, fetcher) {
    var input = event.target;
    var send = fetcher || window.fetch;
    state.preview = null;
    state.previewError = "";
    if (!input.files || !input.files.length) {
      return Promise.resolve();
    }
    state.previewing = true;
    var body = new FormData();
    body.append("file", input.files[0]);
    return send(TOPOLOGY_PREVIEW_URL, { method: "POST", body: body })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (result.ok) {
          state.preview = result.data;
        } else {
          state.previewError = detailMessage(result.data, TOPOLOGY_UNREADABLE);
        }
      })
      .catch(function () {
        state.previewError = TOPOLOGY_UNREADABLE;
      })
      .then(function () {
        state.previewing = false;
      });
  }

  window.ladaReadTopology = readTopology;

  // Exposed for tests/test_frontend_helpers.py. These have no Python-level route test between
  // them, and two of them have had a bug that reached a user: escapeHtml left quotes alone,
  // which broke the attribute holding a job's result JSON, and detailMessage's predecessor
  // printed "[object Object]" for every field-validation failure.
  window.ladaTestHooks = {
    escapeHtml: escapeHtml,
    detailMessage: detailMessage,
    readTopology: readTopology,
    regimeKeys: regimeKeys,
    regimeFormData: regimeFormData,
  };
})();
