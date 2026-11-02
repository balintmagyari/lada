/* The graph studio: one graph page of the Rheometer workspace.

   The datasets come from window.ladaRheoStore (imported on My data). What to draw and how it
   looks is one plain object, the figure spec, saved per page in localStorage; it holds choices
   only, never data, and the same spec will drive figure export. Every change to a control
   writes the spec and redraws, so the graph always shows the current settings. */

(function () {
  "use strict";

  var root = document.querySelector(".studio");
  if (!root) {
    return;
  }
  var P = window.ladaRheoPresets;
  var store = window.ladaRheoStore;
  var key = root.dataset.preset;
  var preset = P.presets[key] || P.presets.custom;
  var SPEC_KEY = "lada-rheo-spec:" + key;
  var NAVIGATION_QUANTITIES = [
    "angular_frequency", "frequency", "shear_rate", "strain", "stress", "temperature", "time"
  ];

  var byQ = {};
  var entries = []; // every table: { dataset, block, id }
  var spec = null;
  var view = "explore"; // or "figure"
  var tts = null; // the last shift factors from /api/rheometry/tts (Master curve only)
  var ttsSent = ""; // the request they answer, to skip asking twice
  var lastResult = null; // the last drawable build(), for export

  function $(id) {
    return document.getElementById(id);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = text;
    }
    return node;
  }

  function token(name, fallback) {
    var value = getComputedStyle(document.body).getPropertyValue(name).trim();
    return value || fallback;
  }

  // --- The spec -----------------------------------------------------------------------

  function merge(base, saved) {
    Object.keys(saved || {}).forEach(function (k) {
      var value = saved[k];
      if (value && typeof value === "object" && !Array.isArray(value) &&
          base[k] && typeof base[k] === "object" && !Array.isArray(base[k])) {
        merge(base[k], value);
      } else if (k in base) {
        base[k] = value;
      }
    });
    return base;
  }

  function loadSpec() {
    var saved = null;
    try {
      saved = JSON.parse(window.localStorage.getItem(SPEC_KEY) || "null");
    } catch (e) {
      saved = null;
    }
    return merge(P.defaultSpec(key), saved && saved.version === 1 ? saved : null);
  }

  var saveTimer = null;
  function saveSpec() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      try {
        window.localStorage.setItem(SPEC_KEY, JSON.stringify(spec));
      } catch (e) {
        /* Storage blocked: the graph still works, it just is not remembered. */
      }
    }, 200);
  }

  function getPath(path) {
    return path.split(".").reduce(function (obj, part) {
      return obj === undefined || obj === null ? undefined : obj[part];
    }, spec);
  }

  function setPath(path, value) {
    var parts = path.split(".");
    var last = parts.pop();
    var target = parts.reduce(function (obj, part) {
      if (obj[part] === undefined || obj[part] === null) {
        obj[part] = {};
      }
      return obj[part];
    }, spec);
    target[last] = value;
  }

  // --- Tables and columns -------------------------------------------------------------

  function stem(filename) {
    return filename.replace(/\.[^.]+$/, "");
  }

  function defaultLabel(entry) {
    var name = stem(entry.dataset.filename);
    if (entry.dataset.blocks.length === 1) {
      return name;
    }
    var t = entry.block.temperature;
    return name + ", " + (t !== null && t !== undefined ? t + " °C" : entry.block.name);
  }

  // A table starts ticked when it is the page's kind of test; on the master curve it must
  // also have been measured at one temperature, or it could not be shifted.
  function defaultOn(entry) {
    var block = entry.block;
    var kind = preset.testTypes.indexOf(block.test_type) >= 0;
    var temperature = block.temperature !== null && block.temperature !== undefined;
    return kind && (!preset.superpose || temperature);
  }

  function blockState(entry) {
    var saved = spec.blocks[entry.id] || {};
    return {
      on: saved.on !== undefined ? saved.on : defaultOn(entry),
      label: saved.label || ""
    };
  }

  // Within one file the file name says nothing that tells the curves apart, so the legend
  // keeps only what does: the temperature, or the step's name.
  function shortLabel(entry) {
    var t = entry.block.temperature;
    return t !== null && t !== undefined ? t + " °C" : entry.block.name;
  }

  function oneFile(active) {
    return active.length > 1 && active.every(function (entry) {
      return entry.dataset.id === active[0].dataset.id;
    });
  }

  function labelOf(entry, active) {
    var saved = blockState(entry).label;
    if (saved) {
      return saved;
    }
    return active && oneFile(active) ? shortLabel(entry) : defaultLabel(entry);
  }

  function activeEntries() {
    return entries.filter(function (entry) {
      return blockState(entry).on;
    });
  }

  // A column is named by an identifier: "q:<quantity>" for a recognised one, "c:<label>" for
  // one that was not, so a custom plot can still use it.
  function identOf(column) {
    return column.quantity ? "q:" + column.quantity : "c:" + column.label;
  }

  function findColumn(block, ident) {
    var found = null;
    block.columns.forEach(function (column) {
      if (identOf(column) === ident && (!found || (found.derived && !column.derived))) {
        found = column;
      }
    });
    return found;
  }

  function quantityOf(ident) {
    return ident && ident.indexOf("q:") === 0 ? ident.slice(2) : null;
  }

  function identText(ident) {
    var q = quantityOf(ident);
    return q && byQ[q] ? byQ[q].name + " " + byQ[q].symbol : ident.slice(2);
  }

  function identSymbol(ident) {
    var q = quantityOf(ident);
    return q && byQ[q] ? byQ[q].symbol : ident.slice(2);
  }

  function availableIdents(list) {
    var seen = {};
    var out = [];
    list.forEach(function (entry) {
      entry.block.columns.forEach(function (column) {
        var ident = identOf(column);
        if (!seen[ident] && !column.is_index) {
          seen[ident] = true;
          out.push(ident);
        }
      });
    });
    return out;
  }

  // --- Units --------------------------------------------------------------------------

  function unitOf(q) {
    var quantity = byQ[q];
    if (!quantity) {
      return null;
    }
    var wanted = spec.units[quantity.unit];
    return quantity.units.filter(function (u) {
      return u.symbol === wanted;
    })[0] || quantity.units[0];
  }

  function toDisplay(values, ident) {
    var q = quantityOf(ident);
    var unit = q ? unitOf(q) : null;
    return values.map(function (v) {
      if (v === null || v === undefined) {
        return null;
      }
      return unit ? (v - unit.offset) / unit.factor : v;
    });
  }

  function unitText(ident, column) {
    var q = quantityOf(ident);
    if (q) {
      var unit = unitOf(q);
      return unit ? unit.symbol : "";
    }
    return column ? column.unit || column.source_unit || "" : "";
  }

  // --- What goes on each axis ---------------------------------------------------------

  function firstPresent(list, candidates) {
    for (var i = 0; i < candidates.length; i++) {
      var ident = "q:" + candidates[i];
      for (var j = 0; j < list.length; j++) {
        if (findColumn(list[j].block, ident)) {
          return ident;
        }
      }
    }
    return candidates.length ? "q:" + candidates[0] : "";
  }

  function plan() {
    var active = activeEntries();
    if (key === "custom") {
      var idents = availableIdents(active.length ? active : entries);
      var x = spec.custom.x && idents.indexOf(spec.custom.x) >= 0 ? spec.custom.x : "";
      if (!x) {
        x = idents.filter(function (ident) {
          return NAVIGATION_QUANTITIES.indexOf(quantityOf(ident)) >= 0;
        })[0] || idents[0] || "";
      }
      var notX = function (i) {
        return i !== x;
      };
      return {
        active: active,
        x: x,
        left: spec.custom.left.filter(notX),
        right: spec.custom.right.filter(notX)
      };
    }
    return {
      active: active,
      x: firstPresent(active, preset.x),
      left: preset.y.map(function (q) {
        return "q:" + q;
      }),
      right: spec.right && preset.right.indexOf(spec.right) >= 0 ? ["q:" + spec.right] : []
    };
  }

  function axisTitle(idents, override) {
    if (override) {
      return override;
    }
    if (!idents.length) {
      return "";
    }
    var sample = null;
    entries.some(function (entry) {
      sample = findColumn(entry.block, idents[0]);
      return !!sample;
    });
    var units = idents.map(function (ident) {
      return unitText(ident, sample);
    });
    var same = units.every(function (u) {
      return u === units[0];
    });
    var text = idents.map(identSymbol).join(", ");
    if (same && units[0]) {
      text += " (" + units[0] + ")";
    }
    return text;
  }

  // On a master curve the horizontal axis is the reduced frequency aT·ω.
  function xTitle(p) {
    if (spec.axes.x.title) {
      return spec.axes.x.title;
    }
    var title = axisTitle([p.x], "");
    return preset.superpose && title ? "aT·" + title : title;
  }

  // --- Superposition (Master curve) ----------------------------------------------------

  function hasTemperature(entry) {
    return entry.block.temperature !== null && entry.block.temperature !== undefined;
  }

  // The sweeps the shift factors are found for: ticked, at one temperature, with ω and a
  // modulus to overlap.
  function ttsEntries() {
    return activeEntries().filter(function (entry) {
      return hasTemperature(entry) && findColumn(entry.block, "q:angular_frequency") &&
        (findColumn(entry.block, "q:storage_modulus") || findColumn(entry.block, "q:loss_modulus"));
    });
  }

  function ttsRequest() {
    var manual = {};
    Object.keys(spec.tts.manual || {}).forEach(function (id) {
      var value = parseFloat(spec.tts.manual[id]);
      if (isFinite(value)) {
        manual[id] = value;
      }
    });
    var reference = parseFloat(spec.tts.reference);
    return {
      curves: ttsEntries().map(function (entry) {
        var omega = findColumn(entry.block, "q:angular_frequency");
        var g1 = findColumn(entry.block, "q:storage_modulus");
        var g2 = findColumn(entry.block, "q:loss_modulus");
        return {
          id: entry.id,
          label: labelOf(entry),
          temperature: entry.block.temperature,
          omega: omega.values,
          storage: g1 ? g1.values : [],
          loss: g2 ? g2.values : []
        };
      }),
      reference: isFinite(reference) ? reference : null,
      use: spec.tts.use,
      vertical: !!spec.tts.vertical,
      manual: manual
    };
  }

  var ttsTimer = null;
  function requestTTS() {
    if (!preset.superpose) {
      return;
    }
    clearTimeout(ttsTimer);
    ttsTimer = setTimeout(function () {
      var body = ttsRequest();
      var sent = JSON.stringify(body);
      if (body.curves.length < 2) {
        tts = null;
        ttsSent = "";
        renderTTS();
        draw();
        return;
      }
      if (sent === ttsSent && tts) {
        return;
      }
      ttsSent = sent;
      fetch("/api/rheometry/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: sent
      })
        .then(function (r) {
          if (!r.ok) {
            return errorText(r).then(function (text) {
              throw new Error(text);
            });
          }
          return r.json();
        })
        .then(function (result) {
          if (sent !== ttsSent) {
            return; // a newer request has been sent since
          }
          tts = result;
          renderTTS();
          draw();
        })
        .catch(function (error) {
          tts = null;
          renderTTS(String(error.message || error));
          draw();
        });
    }, 150);
  }

  function shiftOf(id) {
    if (!tts) {
      return null;
    }
    return tts.shifts.filter(function (s) {
      return s.id === id;
    })[0] || null;
  }

  // A copy of a sweep moved onto the master curve: ω × aT, moduli ÷ bT, |η*| ÷ (aT bT).
  // Quantities that do not shift (tan δ, δ) are kept; anything else is left out.
  function shifted(entry, shift) {
    var a = Math.pow(10, shift.log_at);
    var b = Math.pow(10, shift.log_bt);
    var scale = {
      angular_frequency: a,
      frequency: a,
      storage_modulus: 1 / b,
      loss_modulus: 1 / b,
      complex_modulus: 1 / b,
      complex_viscosity: 1 / (a * b),
      tan_delta: 1,
      phase_angle: 1
    };
    var columns = entry.block.columns.filter(function (c) {
      return c.quantity && scale[c.quantity] !== undefined;
    }).map(function (c) {
      var k = scale[c.quantity];
      return Object.assign({}, c, {
        values: c.values.map(function (v) {
          return v === null || v === undefined ? v : v * k;
        })
      });
    });
    var block = Object.assign({}, entry.block, { columns: columns });
    return { dataset: entry.dataset, block: block, id: entry.id };
  }

  function superposed(active, notes) {
    var out = [];
    active.forEach(function (entry) {
      var shift = shiftOf(entry.id);
      if (shift) {
        out.push(shifted(entry, shift));
      } else if (!hasTemperature(entry)) {
        notes.push(labelOf(entry) + " was not measured at one temperature, so it cannot be " +
          "shifted and is not drawn.");
      }
    });
    return out;
  }

  // --- Building the graph -------------------------------------------------------------

  function build() {
    var p = plan();
    var palette = (P.palettes[spec.style.palette] || P.palettes.lada).colors;
    var traces = [];
    var notes = [];
    if (preset.superpose) {
      p.ticked = p.active;
      p.active = superposed(p.active, notes);
    }
    var single = p.active.length === 1;
    var series = p.left.map(function (ident) {
      return { ident: ident, axis: "y" };
    }).concat(p.right.map(function (ident) {
      return { ident: ident, axis: "y2" };
    }));
    var many = series.length > 1;
    // Several samples and several quantities: the legend lists each sample once (by colour)
    // and adds a grey key for the markers, instead of one entry per sample and quantity.
    var keyed = many && !single;
    var ink = token("--ink-muted", "#5e6b77");

    p.active.forEach(function (entry, n) {
      var color = palette[n % palette.length];
      var xColumn = p.x ? findColumn(entry.block, p.x) : null;
      if (!xColumn) {
        if (p.x) {
          notes.push(labelOf(entry) + " has no " + identText(p.x) + ", so it is not drawn.");
        }
        return;
      }
      var xs = toDisplay(xColumn.values, p.x);
      var missing = [];
      series.forEach(function (s, index) {
        var column = findColumn(entry.block, s.ident);
        if (!column) {
          missing.push(identText(s.ident));
          return;
        }
        var marker = P.markerFor(quantityOf(s.ident), index);
        var sample = labelOf(entry, p.active);
        var name = single && many ? identSymbol(s.ident) : sample;
        var first = !keyed || !traces.some(function (t) {
          return t.legendgroup === entry.id;
        });
        traces.push({
          type: "scatter",
          legendgroup: entry.id,
          showlegend: first,
          mode: spec.style.mode === "lines" ? "lines"
            : spec.style.mode === "both" ? "lines+markers" : "markers",
          name: name,
          x: xs,
          y: toDisplay(column.values, s.ident),
          yaxis: s.axis,
          marker: {
            symbol: marker.filled ? marker.symbol : marker.symbol + "-open",
            size: Number(spec.style.markerSize) || 7,
            color: color,
            line: { color: color, width: marker.filled ? 0 : 1.5 }
          },
          line: {
            color: color,
            width: Number(spec.style.lineWidth) || 1.5,
            dash: s.axis === "y2" ? "dot" : "solid"
          },
          hovertemplate: "%{x:.4g}, %{y:.4g}<extra>" +
            (keyed ? sample + ", " + identSymbol(s.ident) : name).replace(/</g, "&lt;") +
            "</extra>"
        });
      });
      if (series.length && missing.length === series.length) {
        notes.push(labelOf(entry) + " has none of " + missing.join(", ") + ".");
      } else if (missing.length) {
        notes.push(labelOf(entry) + " has no " + missing.join(" or ") + "; the rest is drawn.");
      }
    });
    if (keyed && traces.length) {
      series.forEach(function (s, index) {
        var marker = P.markerFor(quantityOf(s.ident), index);
        traces.push({
          type: "scatter",
          mode: "markers",
          name: identSymbol(s.ident) + (s.axis === "y2" ? " (right)" : ""),
          x: [null],
          y: [null],
          yaxis: s.axis,
          legendgroup: "key",
          hoverinfo: "skip",
          marker: {
            symbol: marker.filled ? marker.symbol : marker.symbol + "-open",
            size: Number(spec.style.markerSize) || 7,
            color: ink,
            line: { color: ink, width: marker.filled ? 0 : 1.5 }
          }
        });
      });
    }
    return { plan: p, traces: traces, notes: notes };
  }

  function range(axis) {
    var lo = parseFloat(axis.min);
    var hi = parseFloat(axis.max);
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) {
      return null;
    }
    if (axis.log) {
      return lo > 0 && hi > 0 ? [Math.log10(lo), Math.log10(hi)] : null;
    }
    return [lo, hi];
  }

  var LEGENDS = {
    above: { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom" },
    "inside-top-right": { x: 0.98, y: 0.98, xanchor: "right", yanchor: "top" },
    "inside-top-left": { x: 0.02, y: 0.98, xanchor: "left", yanchor: "top" },
    "inside-bottom-right": { x: 0.98, y: 0.02, xanchor: "right", yanchor: "bottom" },
    "inside-bottom-left": { x: 0.02, y: 0.02, xanchor: "left", yanchor: "bottom" },
    outside: { x: 1.02, y: 1, xanchor: "left", yanchor: "top" }
  };

  function axisLayout(axis, title, extra) {
    var ink = token("--ink", "#1e2832");
    var layout = {
      title: { text: title, standoff: 8 },
      type: axis.log ? "log" : "linear",
      showline: true,
      linecolor: ink,
      linewidth: 1,
      ticks: "inside",
      ticklen: 5,
      tickcolor: ink,
      // Powers of ten on a log axis; plain numbers on a linear one.
      exponentformat: axis.log ? "power" : "none",
      showexponent: "all",
      separatethousands: false,
      showgrid: !!spec.style.grid,
      gridcolor: token("--line", "#e1e5ea"),
      zeroline: false,
      hoverformat: ".4g"
    };
    if (axis.log) {
      layout.minor = { ticks: "inside", ticklen: 3, tickcolor: ink, showgrid: false };
    }
    var r = range(axis);
    if (r) {
      layout.range = r;
      layout.autorange = false;
    } else {
      layout.autorange = true;
    }
    Object.keys(extra || {}).forEach(function (k) {
      layout[k] = extra[k];
    });
    return layout;
  }

  function message(text, link) {
    var box = $("studio-message");
    box.textContent = "";
    if (!text) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    box.appendChild(document.createTextNode(text));
    if (link) {
      box.appendChild(document.createTextNode(" "));
      var a = el("a", "", link.text);
      a.href = link.href;
      box.appendChild(a);
    }
  }

  function notesList(notes) {
    var list = $("studio-notes");
    list.textContent = "";
    list.hidden = !notes.length;
    notes.forEach(function (note) {
      list.appendChild(el("li", "", note));
    });
  }

  function emptyReason(result) {
    if (!entries.length) {
      return ["Nothing to plot yet. Your rheometer files appear here once imported on",
        { text: "My data.", href: "/rheometer/my-data" }];
    }
    if (!(result.plan.ticked || result.plan.active).length) {
      return [key === "custom"
        ? "Tick the tables to plot on the left, then choose what goes on each axis."
        : "Tick a table on the left to draw it.", null];
    }
    if (preset.superpose && ttsEntries().length < 2) {
      return ["A master curve needs sweeps at two or more temperatures. Tick another, or " +
        "import a temperature series on", { text: "My data.", href: "/rheometer/my-data" }];
    }
    if (preset.superpose && !tts) {
      return ["Working out the shift factors…", null];
    }
    if (key === "custom" &&
        (!result.plan.x || !result.plan.left.length && !result.plan.right.length)) {
      return ["Choose what goes on the left or right axis under Graph.", null];
    }
    if (!result.traces.length) {
      return ["None of the ticked tables has what this graph needs; the notes below say what " +
        "is missing.", null];
    }
    return null;
  }

  function draw() {
    var target = $("studio-plot");
    var result = build();
    var reason = emptyReason(result);
    notesList(result.notes);
    if (reason) {
      lastResult = null;
      message(reason[0], reason[1]);
      target.hidden = true;
      $("studio-figure").hidden = true;
      $("studio-export").hidden = true;
      if (window.Plotly) {
        window.Plotly.purge(target);
      }
      return;
    }
    message("");
    lastResult = result;
    $("studio-export").hidden = false;
    if (view === "figure") {
      target.hidden = true;
      $("studio-figure").hidden = false;
      refreshFigureSoon();
      return;
    }
    $("studio-figure").hidden = true;
    target.hidden = false;
    if (!window.Plotly) {
      message("The plotting library could not be loaded. Check the network and reload.");
      return;
    }

    var p = result.plan;
    var size = P.sizes[spec.style.size] || P.sizes.single;
    var width = target.clientWidth || 600;
    var height = Math.max(260, Math.min(width * size.height / size.width,
      window.innerHeight * 0.78));
    var px = (Number(spec.style.fontSize) || 9) * 1.333;
    var hasY2 = result.traces.some(function (t) {
      return t.yaxis === "y2";
    });
    var legend = LEGENDS[spec.style.legend];
    var ink = token("--ink", "#1e2832");

    var layout = {
      height: height,
      margin: { l: 72, r: hasY2 ? 72 : 20, t: spec.style.legend === "above" ? 44 : 16, b: 58 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {
        family: spec.style.font === "serif"
          ? "'STIX Two Text', 'Times New Roman', Georgia, serif"
          : token("--font-sans", "system-ui, sans-serif"),
        size: px,
        color: ink
      },
      showlegend: !!legend,
      xaxis: axisLayout(spec.axes.x, xTitle(p), { mirror: "ticks" }),
      yaxis: axisLayout(spec.axes.y, axisTitle(p.left, spec.axes.y.title),
        { mirror: hasY2 ? false : "ticks" }),
      hovermode: "closest"
    };
    if (legend) {
      layout.legend = Object.assign(
        {
          bgcolor: "rgba(0,0,0,0)",
          borderwidth: 0,
          traceorder: "normal",
          font: { size: px * 0.92, color: ink }
        },
        legend,
        hasY2 && spec.style.legend === "outside" ? { x: 1.14 } : {}
      );
    }
    if (hasY2) {
      layout.yaxis2 = axisLayout(spec.axes.y2, axisTitle(p.right, spec.axes.y2.title),
        { overlaying: "y", side: "right" });
    }
    // On a log axis spanning under two decades Plotly labels every minor tick (2…9), which
    // crowds the axis; labelling only the 2s and 5s keeps it readable.
    [["xaxis", "x", null], ["yaxis", "y", "y"], ["yaxis2", "y", "y2"]].forEach(function (a) {
      var axis = layout[a[0]];
      if (!axis || axis.type !== "log") {
        return;
      }
      var values = [];
      result.traces.forEach(function (t) {
        if (a[2] === null || (t.yaxis || "y") === a[2]) {
          (t[a[1]] || []).forEach(function (v) {
            if (v > 0) {
              values.push(v);
            }
          });
        }
      });
      if (values.length > 1) {
        var span = Math.log10(Math.max.apply(null, values) / Math.min.apply(null, values));
        if (span < 2) {
          axis.dtick = "D2";
        }
      }
    });
    if (spec.style.legend === "outside") {
      layout.margin.r = hasY2 ? 200 : 160;
    }

    window.Plotly.react(target, result.traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
      toImageButtonOptions: { format: "png", scale: 3, filename: key.replace(/_/g, "-") }
    });
  }

  // --- Export -------------------------------------------------------------------------
  // The figure spec sent to the server is built from the same traces the Explore view draws,
  // already in display units, so the export can only differ from it in typesetting.

  function number(text) {
    var value = parseFloat(text);
    return isFinite(value) ? value : null;
  }

  function axisSpec(axis, idents) {
    return {
      title: axisTitle(idents, axis.title),
      log: !!axis.log,
      min: number(axis.min),
      max: number(axis.max)
    };
  }

  function figureSpec(result) {
    var p = result.plan;
    var size = P.sizes[spec.style.size] || P.sizes.single;
    var hasY2 = result.traces.some(function (t) {
      return t.yaxis === "y2";
    });
    return {
      width_mm: size.width,
      height_mm: size.height,
      font: spec.style.font,
      font_size: Number(spec.style.fontSize) || 9,
      mode: spec.style.mode,
      marker_size: Number(spec.style.markerSize) || 7,
      line_width: Number(spec.style.lineWidth) || 1.5,
      legend: spec.style.legend,
      grid: !!spec.style.grid,
      x: Object.assign(axisSpec(spec.axes.x, [p.x]), { title: xTitle(p) }),
      y: axisSpec(spec.axes.y, p.left),
      y2: hasY2 ? axisSpec(spec.axes.y2, p.right) : null,
      filename: key.replace(/_/g, "-"),
      series: result.traces.map(function (t) {
        var symbol = t.marker.symbol;
        var open = /-open$/.test(symbol);
        return {
          name: t.name || "",
          axis: t.yaxis === "y2" ? "y2" : "y",
          x: t.x,
          y: t.y,
          color: t.legendgroup === "key" ? "#6b6b6b" : t.marker.color,
          marker: symbol.replace(/-open$/, ""),
          filled: !open,
          show_legend: t.showlegend !== false,
          key: t.legendgroup === "key"
        };
      })
    };
  }

  function errorText(response) {
    return response.json().then(
      function (body) {
        var detail = body && body.detail;
        if (Array.isArray(detail)) {
          return detail.map(function (d) {
            return d.msg;
          }).join(" ");
        }
        return detail || "The figure could not be made.";
      },
      function () {
        return "The figure could not be made.";
      }
    );
  }

  function postFigure(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      if (!response.ok) {
        return errorText(response).then(function (text) {
          throw new Error(text);
        });
      }
      var notes = response.headers.get("X-Figure-Notes") || "";
      return response.blob().then(function (blob) {
        blob.notes = notes;
        return blob;
      });
    });
  }

  var figureUrl = null;
  var figureTimer = null;
  var figureRequest = 0;
  function refreshFigure() {
    if (!lastResult || view !== "figure") {
      return;
    }
    var note = $("studio-figure-note");
    var img = $("studio-figure-img");
    var ticket = ++figureRequest;
    note.hidden = false;
    note.textContent = "Drawing the figure…";
    postFigure("/api/rheometry/figure?fmt=svg&preview=true", figureSpec(lastResult))
      .then(function (blob) {
        if (ticket !== figureRequest) {
          return; // a newer change is already being drawn
        }
        if (figureUrl) {
          URL.revokeObjectURL(figureUrl);
        }
        figureUrl = URL.createObjectURL(blob);
        img.src = figureUrl;
        var size = P.sizes[spec.style.size] || P.sizes.single;
        note.textContent = "Exactly as exported: " + size.width + " × " +
          Math.round(size.height) + " mm, " + spec.style.fontSize + " pt " +
          (spec.style.font === "serif" ? "serif." : "sans-serif.") +
          (blob.notes ? " " + blob.notes : "");
        note.classList.toggle("is-advice", !!blob.notes);
      })
      .catch(function (error) {
        if (ticket === figureRequest) {
          note.textContent = String(error.message || error);
        }
      });
  }

  function refreshFigureSoon() {
    clearTimeout(figureTimer);
    figureTimer = setTimeout(refreshFigure, 450);
  }

  function setView(next) {
    view = next;
    ["explore", "figure"].forEach(function (name) {
      var tab = $("tab-" + name);
      tab.setAttribute("aria-selected", name === next ? "true" : "false");
      tab.tabIndex = name === next ? 0 : -1;
    });
    draw();
  }

  function download(blob, filename) {
    var link = el("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () {
      URL.revokeObjectURL(link.href);
    }, 2000);
  }

  function exportAs(kind, button) {
    if (!lastResult) {
      return;
    }
    var status = $("studio-export-status");
    var payload = figureSpec(lastResult);
    var dpi = $("studio-dpi").value;
    var url = kind === "pgfplots"
      ? "/api/rheometry/pgfplots"
      : "/api/rheometry/figure?fmt=" + kind + (kind === "png" ? "&dpi=" + dpi : "");
    var filename = payload.filename + (kind === "pgfplots" ? "-pgfplots.zip" : "." + kind);
    button.disabled = true;
    status.className = "studio-export-status";
    status.textContent = "Preparing " + filename + "…";
    postFigure(url, payload)
      .then(function (blob) {
        download(blob, filename);
        status.textContent = "Saved " + filename + ".";
      })
      .catch(function (error) {
        status.className = "studio-export-status is-error";
        status.textContent = String(error.message || error);
      })
      .then(function () {
        button.disabled = false;
      });
  }

  // --- The shift-factor panel ---------------------------------------------------------

  function fixed(value, digits) {
    return value === null || value === undefined ? "" : Number(value).toFixed(digits);
  }

  function renderTTS(error) {
    var panel = $("tts-panel");
    if (!panel) {
      return;
    }
    var list = ttsEntries();
    panel.hidden = list.length < 2 && !error;
    if (panel.hidden) {
      return;
    }

    // Reference choices: the temperatures that were measured.
    var select = $("tts-reference");
    var temps = list.map(function (e) {
      return e.block.temperature;
    }).filter(function (t, i, all) {
      return all.indexOf(t) === i;
    }).sort(function (a, b) {
      return a - b;
    });
    var current = tts ? tts.reference : parseFloat(spec.tts.reference);
    select.textContent = "";
    temps.forEach(function (t) {
      var o = el("option", "", t + " °C");
      o.value = String(t);
      o.selected = t === current;
      select.appendChild(o);
    });
    $("tts-use").value = spec.tts.use;
    $("tts-vertical").checked = !!spec.tts.vertical;

    var rows = $("tts-rows");
    rows.textContent = "";
    var notes = $("tts-notes");
    notes.textContent = "";
    if (error) {
      notes.hidden = false;
      notes.appendChild(el("li", "", error));
    }
    if (!tts) {
      $("tts-fits").textContent = "";
      return;
    }
    tts.shifts.forEach(function (shift) {
      var tr = el("tr");
      var isRef = shift.temperature === tts.reference;
      tr.appendChild(el("td", "", fixed(shift.temperature, 1)));
      var cell = el("td");
      var input = el("input");
      input.type = "number";
      input.step = "0.01";
      input.value = fixed(shift.log_at, 3);
      input.dataset.shift = shift.id;
      input.disabled = isRef;
      input.setAttribute("aria-label", "log aT at " + shift.temperature + " °C");
      if (shift.auto_log_at !== null) {
        input.placeholder = fixed(shift.auto_log_at, 3);
      }
      cell.appendChild(input);
      tr.appendChild(cell);
      tr.appendChild(el("td", "tts-at", Math.pow(10, shift.log_at).toPrecision(3)));
      var source = el("td", "tts-source");
      if (isRef) {
        source.textContent = "reference";
      } else if (shift.manual) {
        var reset = el("button", "quiet-button", "Use overlap");
        reset.type = "button";
        reset.dataset.reset = shift.id;
        reset.title = shift.auto_log_at !== null
          ? "Overlap found " + fixed(shift.auto_log_at, 3) : "Overlap found no shift";
        source.appendChild(reset);
      } else {
        source.textContent = "by overlap";
      }
      tr.appendChild(source);
      rows.appendChild(tr);
    });

    var fits = $("tts-fits");
    fits.textContent = "";
    function fact(term, text) {
      fits.appendChild(el("dt", "", term));
      fits.appendChild(el("dd", "", text));
    }
    var ref = fixed(tts.reference, 1) + " °C";
    // Three sweeps and two parameters fit exactly; say so rather than print "6e-17".
    function rms(value) {
      return value < 5e-4 ? "fits every point" : "rms " + value.toPrecision(2) + " in log aT";
    }
    fact("WLF", tts.wlf
      ? "C1 = " + tts.wlf.c1.toPrecision(4) + ", C2 = " + tts.wlf.c2.toPrecision(4) +
        " K at " + ref + " (" + rms(tts.wlf.rms) + ")"
      : tts.wlf_note);
    fact("Arrhenius", tts.arrhenius
      ? "Ea = " + tts.arrhenius.ea_kj_per_mol.toPrecision(4) + " kJ/mol (" +
        rms(tts.arrhenius.rms) + ")"
      : tts.arrhenius_note);

    notes.hidden = !tts.notes.length && !error;
    tts.notes.forEach(function (note) {
      notes.appendChild(el("li", "", note));
    });
    drawTTSPlot();
  }

  function drawTTSPlot() {
    var target = $("tts-plot");
    if (!target || !tts || !window.Plotly) {
      return;
    }
    var ink = token("--ink", "#1e2832");
    var accent = token("--accent", "#2f4fae");
    var ts = tts.shifts.map(function (s) {
      return s.temperature;
    });
    var lo = Math.min.apply(null, ts);
    var hi = Math.max.apply(null, ts);
    var pad = (hi - lo) * 0.08 || 5;
    var grid = [];
    for (var i = 0; i <= 60; i++) {
      grid.push(lo - pad + (hi - lo + 2 * pad) * i / 60);
    }
    var traces = [{
      type: "scatter",
      mode: "markers",
      name: "Shift factors",
      x: ts,
      y: tts.shifts.map(function (s) {
        return s.log_at;
      }),
      marker: {
        size: 8,
        color: accent,
        symbol: tts.shifts.map(function (s) {
          return s.manual ? "circle-open" : "circle";
        }),
        line: { color: accent, width: 1.5 }
      },
      hovertemplate: "%{x} °C, log aT %{y:.3f}<extra></extra>"
    }];
    var dt = function (t) {
      return t - tts.reference;
    };
    if (tts.wlf) {
      var c1 = tts.wlf.c1;
      var c2 = tts.wlf.c2;
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "WLF",
        x: grid,
        y: grid.map(function (t) {
          return c2 + dt(t) > 0 ? -c1 * dt(t) / (c2 + dt(t)) : null;
        }),
        line: { color: ink, width: 1.2 },
        hoverinfo: "skip"
      });
    }
    if (tts.arrhenius) {
      var k = tts.arrhenius.ea_kj_per_mol * 1000 / 8.314462618 / Math.LN10;
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Arrhenius",
        x: grid,
        y: grid.map(function (t) {
          return k * (1 / (t + 273.15) - 1 / (tts.reference + 273.15));
        }),
        line: { color: ink, width: 1.2, dash: "dot" },
        hoverinfo: "skip"
      });
    }
    var axis = {
      showline: true,
      linecolor: ink,
      ticks: "inside",
      tickcolor: ink,
      mirror: "ticks",
      zeroline: false,
      showgrid: false
    };
    window.Plotly.react(target, traces, {
      height: 240,
      margin: { l: 56, r: 12, t: 28, b: 44 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: token("--font-sans", "system-ui, sans-serif"), size: 11, color: ink },
      showlegend: true,
      legend: { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom",
        bgcolor: "rgba(0,0,0,0)" },
      xaxis: Object.assign({ title: { text: "T (°C)" } }, axis),
      yaxis: Object.assign({ title: { text: "log aT" } }, axis)
    }, { displaylogo: false, responsive: true, displayModeBar: false });
  }

  function exportShifts(kind) {
    if (!tts) {
      return;
    }
    var lines;
    if (kind === "csv") {
      lines = ["temperature_C,log_aT,aT,log_bT,set_by"];
      tts.shifts.forEach(function (s) {
        lines.push([s.temperature, s.log_at, Math.pow(10, s.log_at), s.log_bt,
          s.manual ? "hand" : "overlap"].join(","));
      });
    } else {
      lines = ["% Shift factors, reference " + tts.reference + " C", "T log_aT"];
      tts.shifts.forEach(function (s) {
        lines.push(s.temperature + " " + s.log_at);
      });
    }
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
    download(blob, "shift-factors." + kind);
  }

  function setupTTS() {
    var panel = $("tts-panel");
    if (!panel) {
      return;
    }
    panel.addEventListener("change", function (event) {
      var target = event.target;
      if (target.dataset.tts) {
        spec.tts[target.dataset.tts] = target.type === "checkbox" ? target.checked : target.value;
      } else if (target.dataset.shift) {
        var value = target.value.trim();
        if (value === "" || !isFinite(parseFloat(value))) {
          delete spec.tts.manual[target.dataset.shift];
        } else {
          spec.tts.manual[target.dataset.shift] = parseFloat(value);
        }
      } else {
        return;
      }
      saveSpec();
      requestTTS();
    });
    panel.addEventListener("click", function (event) {
      var reset = event.target.closest("[data-reset]");
      if (reset) {
        delete spec.tts.manual[reset.dataset.reset];
        saveSpec();
        requestTTS();
      }
      var exporter = event.target.closest("[data-tts-export]");
      if (exporter) {
        exportShifts(exporter.dataset.ttsExport);
      }
    });
  }

  var drawTimer = null;
  function redrawSoon() {
    clearTimeout(drawTimer);
    drawTimer = setTimeout(draw, 120);
  }

  // --- The data panel -----------------------------------------------------------------

  function renderData() {
    var host = $("studio-data");
    host.textContent = "";
    if (!entries.length) {
      var empty = el("p", "studio-hint", "No files imported yet. ");
      var link = el("a", "", "Import files on My data");
      link.href = "/rheometer/my-data";
      empty.appendChild(link);
      empty.appendChild(document.createTextNode(", and they will be listed here."));
      host.appendChild(empty);
      return;
    }
    var palette = (P.palettes[spec.style.palette] || P.palettes.lada).colors;
    var active = activeEntries();
    var matching = entries.filter(function (entry) {
      return preset.testTypes.indexOf(entry.block.test_type) >= 0;
    });
    var others = entries.filter(function (entry) {
      return matching.indexOf(entry) < 0;
    });

    function row(entry) {
      var state = blockState(entry);
      var wrap = el("div", "block-row" + (state.on ? " is-on" : ""));
      var id = "blk-" + entry.id.replace(/[^a-zA-Z0-9_-]/g, "_");
      var box = el("input");
      box.type = "checkbox";
      box.id = id;
      box.checked = state.on;
      box.dataset.block = entry.id;
      var label = el("label", "block-row-label");
      label.htmlFor = id;
      var swatch = el("span", "block-swatch");
      swatch.setAttribute("aria-hidden", "true");
      var n = active.indexOf(entry);
      if (n >= 0) {
        swatch.style.background = palette[n % palette.length];
        swatch.style.borderColor = palette[n % palette.length];
      }
      label.appendChild(swatch);
      var text = el("span", "block-row-text");
      text.appendChild(el("span", "block-row-name", defaultLabel(entry)));
      var meta = [entry.block.n_points + " points"];
      if (entry.dataset.blocks.length > 1) {
        meta.unshift(entry.block.name);
      }
      text.appendChild(el("span", "block-row-meta", meta.join(", ")));
      label.appendChild(text);
      wrap.appendChild(box);
      wrap.appendChild(label);
      if (state.on) {
        var name = el("input", "block-row-legend");
        name.type = "text";
        name.value = state.label;
        name.placeholder = "Legend: " + defaultLabel(entry);
        name.setAttribute("aria-label", "Name in the legend for " + defaultLabel(entry));
        name.dataset.legend = entry.id;
        wrap.appendChild(name);
      }
      return wrap;
    }

    if (key === "custom" || !matching.length) {
      if (key !== "custom") {
        host.appendChild(el("p", "studio-hint",
          "None of your tables looks like a " + preset.title.toLowerCase() +
          ". Tick any of them to plot it anyway."));
      }
      entries.forEach(function (entry) {
        host.appendChild(row(entry));
      });
      return;
    }
    matching.forEach(function (entry) {
      host.appendChild(row(entry));
    });
    if (others.length) {
      var more = el("details", "block-more");
      more.open = others.some(function (entry) {
        return blockState(entry).on;
      });
      more.appendChild(el("summary", "", "Other tables (" + others.length + ")"));
      others.forEach(function (entry) {
        more.appendChild(row(entry));
      });
      host.appendChild(more);
    }
  }

  $("studio-data").addEventListener("change", function (event) {
    var blockId = event.target.dataset.block;
    if (!blockId) {
      return;
    }
    var current = spec.blocks[blockId] || {};
    current.on = event.target.checked;
    spec.blocks[blockId] = current;
    saveSpec();
    renderData();
    renderForm();
    draw();
    requestTTS();
    var again = document.querySelector('[data-block="' + CSS.escape(blockId) + '"]');
    if (again) {
      again.focus();
    }
  });

  $("studio-data").addEventListener("input", function (event) {
    var blockId = event.target.dataset.legend;
    if (!blockId) {
      return;
    }
    var current = spec.blocks[blockId] || { on: true };
    current.label = event.target.value;
    spec.blocks[blockId] = current;
    saveSpec();
    redrawSoon();
  });

  // --- The graph form -----------------------------------------------------------------

  var fieldCount = 0;
  function field(labelText, control, hint) {
    var wrap = el("div", "studio-field");
    var label = el("label", "", labelText);
    if (!control.id) {
      fieldCount += 1;
      control.id = "studio-f" + fieldCount;
    }
    label.htmlFor = control.id;
    wrap.appendChild(label);
    wrap.appendChild(control);
    if (hint) {
      wrap.appendChild(el("small", "studio-field-hint", hint));
    }
    return wrap;
  }

  function select(path, options, value) {
    var node = el("select");
    node.dataset.path = path;
    options.forEach(function (opt) {
      var o = el("option", "", opt[1]);
      o.value = opt[0];
      o.selected = String(opt[0]) === String(value);
      node.appendChild(o);
    });
    return node;
  }

  function checkbox(path, labelText, checked, extra) {
    var wrap = el("label", "studio-check");
    var box = el("input");
    box.type = "checkbox";
    box.checked = !!checked;
    if (path) {
      box.dataset.path = path;
    }
    Object.keys(extra || {}).forEach(function (k) {
      box.dataset[k] = extra[k];
    });
    wrap.appendChild(box);
    wrap.appendChild(document.createTextNode(labelText));
    return wrap;
  }

  function input(path, type, value, attrs) {
    var node = el("input");
    node.type = type;
    node.value = value === undefined || value === null ? "" : value;
    node.dataset.path = path;
    Object.keys(attrs || {}).forEach(function (k) {
      node.setAttribute(k, attrs[k]);
    });
    return node;
  }

  function group(title) {
    var fieldset = el("fieldset", "studio-group");
    fieldset.appendChild(el("legend", "", title));
    return fieldset;
  }

  function renderForm() {
    var form = $("studio-form");
    var active = document.activeElement;
    var focused = active && form.contains(active)
      ? { path: active.dataset.path || "", ident: active.dataset.ident || "",
        list: active.dataset.list || "" }
      : null;
    fieldCount = 0;
    form.textContent = "";
    var p = plan();

    // What to plot
    if (key === "custom") {
      var idents = availableIdents(p.active.length ? p.active : entries);
      var what = group("Axes");
      if (!idents.length) {
        what.appendChild(el("p", "studio-hint", "Tick a table to choose from its columns."));
      } else {
        what.appendChild(field("Horizontal axis", select("custom.x",
          idents.map(function (ident) {
            return [ident, identText(ident)];
          }), p.x)));
        ["left", "right"].forEach(function (side) {
          var list = el("div", "studio-checklist");
          list.appendChild(el("span", "studio-checklist-title",
            side === "left" ? "Left axis" : "Right axis"));
          idents.forEach(function (ident) {
            if (ident !== p.x) {
              list.appendChild(checkbox(null, identText(ident),
                spec.custom[side].indexOf(ident) >= 0, { list: "custom." + side, ident: ident }));
            }
          });
          what.appendChild(list);
        });
      }
      form.appendChild(what);
    } else if (preset.right.length) {
      var extra = group("Right axis");
      extra.appendChild(field("Also show", select("right",
        [["", "Nothing"]].concat(preset.right.map(function (q) {
          return [q, byQ[q] ? byQ[q].name + " " + byQ[q].symbol : q];
        })), spec.right)));
      form.appendChild(extra);
    }

    // Units: one choice per kind of unit on show, so G′ and G″ can never disagree on one axis.
    var families = {};
    var order = [];
    [p.x].concat(p.left, p.right).map(quantityOf).forEach(function (q) {
      if (!q || !byQ[q] || byQ[q].units.length < 2) {
        return;
      }
      var unit = byQ[q].unit;
      if (!families[unit]) {
        families[unit] = [];
        order.push(unit);
      }
      if (families[unit].indexOf(q) < 0) {
        families[unit].push(q);
      }
    });
    if (order.length) {
      var units = group("Units");
      order.forEach(function (unit) {
        var qs = families[unit];
        var name = qs.map(function (q) {
          return byQ[q].symbol;
        }).join(", ");
        units.appendChild(field(name, select("units." + unit,
          byQ[qs[0]].units.map(function (u) {
            return [u.symbol, u.symbol || "no unit"];
          }), (unitOf(qs[0]) || {}).symbol)));
      });
      form.appendChild(units);
    }

    // Axes
    var axes = group("Axis scales");
    [["x", "Horizontal", [p.x]], ["y", p.right.length ? "Left" : "Vertical", p.left],
      ["y2", "Right", p.right]].forEach(function (a) {
      if (a[0] === "y2" && !p.right.length) {
        return;
      }
      var box = el("div", "studio-axis");
      var head = el("div", "studio-axis-head");
      head.appendChild(el("span", "studio-axis-name", a[1]));
      head.appendChild(checkbox("axes." + a[0] + ".log", "Log scale", spec.axes[a[0]].log));
      box.appendChild(head);
      var limits = el("div", "studio-pair");
      limits.appendChild(field("From", input("axes." + a[0] + ".min", "number",
        spec.axes[a[0]].min, { step: "any", placeholder: "auto" })));
      limits.appendChild(field("To", input("axes." + a[0] + ".max", "number",
        spec.axes[a[0]].max, { step: "any", placeholder: "auto" })));
      box.appendChild(limits);
      box.appendChild(field("Title", input("axes." + a[0] + ".title", "text",
        spec.axes[a[0]].title, { placeholder: axisTitle(a[2], "") })));
      axes.appendChild(box);
    });
    form.appendChild(axes);

    // Look
    var look = group("Look");
    look.appendChild(field("Size", select("style.size", Object.keys(P.sizes).map(function (k) {
      return [k, P.sizes[k].name];
    }), spec.style.size), "The shape the figure will be exported at."));
    look.appendChild(field("Colours", select("style.palette",
      Object.keys(P.palettes).map(function (k) {
        return [k, P.palettes[k].name];
      }), spec.style.palette)));
    look.appendChild(field("Draw as", select("style.mode",
      [["markers", "Points"], ["both", "Points and lines"], ["lines", "Lines"]],
      spec.style.mode)));
    var pair = el("div", "studio-pair");
    pair.appendChild(field("Point size", input("style.markerSize", "number",
      spec.style.markerSize, { min: 2, max: 16, step: 1 })));
    pair.appendChild(field("Line width", input("style.lineWidth", "number",
      spec.style.lineWidth, { min: 0.5, max: 5, step: 0.25 })));
    look.appendChild(pair);
    var pair2 = el("div", "studio-pair");
    pair2.appendChild(field("Font", select("style.font",
      [["sans", "Sans-serif"], ["serif", "Serif"]], spec.style.font)));
    pair2.appendChild(field("Font size, pt", input("style.fontSize", "number",
      spec.style.fontSize, { min: 6, max: 24, step: 0.5 })));
    look.appendChild(pair2);
    look.appendChild(field("Legend", select("style.legend", [
      ["above", "Above the plot"],
      ["inside-top-right", "Inside, top right"],
      ["inside-top-left", "Inside, top left"],
      ["inside-bottom-right", "Inside, bottom right"],
      ["inside-bottom-left", "Inside, bottom left"],
      ["outside", "Outside, right"],
      ["none", "No legend"]
    ], spec.style.legend)));
    look.appendChild(checkbox("style.grid", "Grid lines", spec.style.grid));
    form.appendChild(look);

    var reset = el("button", "quiet-button studio-reset", "Reset this graph");
    reset.type = "button";
    reset.addEventListener("click", function () {
      var blocks = spec.blocks;
      spec = P.defaultSpec(key);
      spec.blocks = blocks; // which tables are ticked is a data choice, not styling: keep it
      saveSpec();
      renderData();
      renderForm();
      draw();
    });
    form.appendChild(reset);

    if (focused) {
      var again = Array.prototype.filter.call(form.querySelectorAll("input, select"),
        function (node) {
          return (node.dataset.path || "") === focused.path &&
            (node.dataset.ident || "") === focused.ident &&
            (node.dataset.list || "") === focused.list;
        })[0];
      if (again) {
        again.focus();
      }
    }
  }

  // Structural choices change which controls exist; everything else only redraws.
  function isStructural(path) {
    return path === "right" || path === "custom.x" || path.indexOf("units.") === 0;
  }

  function onFormChange(event) {
    var target = event.target;
    if (target.dataset.list) {
      if (event.type !== "change") {
        return;
      }
      var ident = target.dataset.ident;
      var list = (getPath(target.dataset.list) || []).filter(function (i) {
        return i !== ident;
      });
      if (target.checked) {
        list.push(ident);
        // A quantity sits on one side only.
        var other = target.dataset.list === "custom.left" ? "custom.right" : "custom.left";
        setPath(other, (getPath(other) || []).filter(function (i) {
          return i !== ident;
        }));
      }
      // The first quantity put on an axis sets its scale, the way the presets do.
      var axisKey = target.dataset.list === "custom.left" ? "y" : "y2";
      var q = quantityOf(ident);
      if (target.checked && list.length === 1 && q && byQ[q]) {
        spec.axes[axisKey].log = !!byQ[q].log;
      }
      setPath(target.dataset.list, list);
      saveSpec();
      renderForm();
      draw();
      return;
    }
    var path = target.dataset.path;
    if (!path) {
      return;
    }
    setPath(path, target.type === "checkbox" ? target.checked : target.value);
    // A quantity that spans decades (tan δ, |η*|, stress) starts on a log axis.
    if (path === "right" && event.type === "change" && byQ[target.value]) {
      spec.axes.y2.log = !!byQ[target.value].log;
    }
    if (path === "custom.x" && event.type === "change") {
      var xq = quantityOf(target.value);
      spec.axes.x.log = !!(xq && byQ[xq] && byQ[xq].log);
    }
    saveSpec();
    if (event.type === "change" && isStructural(path)) {
      renderForm();
      draw();
      return;
    }
    redrawSoon();
    if (path === "style.palette") {
      renderData();
    }
  }

  $("studio-form").addEventListener("change", onFormChange);
  $("studio-form").addEventListener("input", function (event) {
    if (event.target.type === "text" || event.target.type === "number") {
      onFormChange(event);
    }
  });
  $("studio-form").addEventListener("submit", function (event) {
    event.preventDefault();
  });

  // --- Start --------------------------------------------------------------------------

  function start() {
    spec = loadSpec();
    Promise.all([
      fetch("/api/rheometry/quantities").then(function (r) {
        if (!r.ok) {
          throw new Error("quantities");
        }
        return r.json();
      }),
      store.all()
    ])
      .then(function (results) {
        results[0].quantities.forEach(function (q) {
          byQ[q.key] = q;
        });
        entries = [];
        results[1].forEach(function (dataset) {
          dataset.blocks.forEach(function (block) {
            entries.push({ dataset: dataset, block: block, id: block.id });
          });
        });
        renderData();
        renderForm();
        draw();
        requestTTS();
      })
      .catch(function () {
        message("The page could not reach the server. Check that it is running, then reload.");
      });
    setupTTS();

    $("tab-explore").addEventListener("click", function () {
      setView("explore");
    });
    $("tab-figure").addEventListener("click", function () {
      setView("figure");
    });
    root.querySelector(".studio-tabs").addEventListener("keydown", function (event) {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        var next = view === "explore" ? "figure" : "explore";
        setView(next);
        $("tab-" + next).focus();
      }
    });
    $("studio-export").addEventListener("click", function (event) {
      var button = event.target.closest("[data-export]");
      if (button) {
        exportAs(button.dataset.export, button);
      }
    });

    // A theme switch recolours the axes and text; the series keep their palette colours.
    var previous = window.ladaRepaintPlots;
    window.ladaRepaintPlots = function () {
      if (previous) {
        previous();
      }
      draw();
      drawTTSPlot();
    };
    if (window.ResizeObserver) {
      var lastWidth = 0;
      new ResizeObserver(function (items) {
        var width = Math.round(items[0].contentRect.width);
        if (Math.abs(width - lastWidth) > 4) {
          lastWidth = width;
          redrawSoon();
        }
      }).observe($("studio-plot").parentNode);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
