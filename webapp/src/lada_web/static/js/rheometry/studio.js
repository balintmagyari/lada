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
  var skippedDatasets = 0; // stored datasets that failed store.validDataset
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
  var ttsPending = false; // whether that request is still waiting for its answer
  var ttsError = ""; // why the last request failed, until one succeeds
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
      var plain = function (v) {
        return v && typeof v === "object" && !Array.isArray(v);
      };
      if (plain(value) && plain(base[k]) && !Object.keys(base[k]).length) {
        // An open map (tables ticked, units, styles, shifts set by hand): its keys are the
        // user's, not the defaults', so it comes back whole.
        base[k] = JSON.parse(JSON.stringify(value));
      } else if (plain(value) && plain(base[k])) {
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

  // --- Markers and colours -------------------------------------------------------------

  var HEX = /^#[0-9a-fA-F]{6}$/;

  function hex(value) {
    return typeof value === "string" && HEX.test(value) ? value.toLowerCase() : null;
  }

  function knownShape(symbol) {
    return P.shapes.some(function (s) {
      return s[0] === symbol;
    });
  }

  // How a quantity is drawn: the rheology convention (G′ filled, G″ open …) unless the user
  // chose otherwise. `color` is null when the quantity takes each sample's colour.
  function seriesStyle(ident, index) {
    var base = P.markerFor(quantityOf(ident), index);
    var own = (spec.styles.series || {})[ident] || {};
    var size = parseFloat(own.size);
    return {
      symbol: knownShape(own.symbol) ? own.symbol : base.symbol,
      filled: typeof own.filled === "boolean" ? own.filled : base.filled,
      size: isFinite(size) && size >= 1 && size <= 30 ? size : Number(spec.style.markerSize) || 7,
      color: hex(own.color)
    };
  }

  // A sample's colour: the one picked for it, else its place in the palette.
  function sampleColor(entry, n) {
    var palette = (P.palettes[spec.style.palette] || P.palettes.lada).colors;
    return hex((spec.styles.samples || {})[entry.id]) || palette[n % palette.length];
  }

  // A series' colour. One sample: each quantity in its own colour, so G′ and G″ never look
  // alike. Several samples: colour tells the samples apart and the marker the quantities.
  function seriesPaint(style, entry, n, index, single) {
    if (style.color) {
      return style.color;
    }
    if (single) {
      var palette = (P.palettes[spec.style.palette] || P.palettes.lada).colors;
      return palette[index % palette.length];
    }
    return sampleColor(entry, n);
  }

  var SVG_NS = "http://www.w3.org/2000/svg";
  var SHAPE_POINTS = {
    diamond: "8,1.5 14.5,8 8,14.5 1.5,8",
    "triangle-up": "8,2 14.5,13.5 1.5,13.5",
    "triangle-down": "1.5,2.5 14.5,2.5 8,14",
    cross: "6,1.5 10,1.5 10,6 14.5,6 14.5,10 10,10 10,14.5 6,14.5 6,10 1.5,10 1.5,6 6,6"
  };

  // A small drawing of a series' marker, exactly as the graph draws it.
  function markerPreview(symbol, filled, color) {
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("width", "18");
    svg.setAttribute("height", "18");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", "series-preview");
    var shape;
    if (symbol === "square") {
      shape = document.createElementNS(SVG_NS, "rect");
      shape.setAttribute("x", "2.5");
      shape.setAttribute("y", "2.5");
      shape.setAttribute("width", "11");
      shape.setAttribute("height", "11");
    } else if (SHAPE_POINTS[symbol]) {
      shape = document.createElementNS(SVG_NS, "polygon");
      shape.setAttribute("points", SHAPE_POINTS[symbol]);
    } else {
      shape = document.createElementNS(SVG_NS, "circle");
      shape.setAttribute("cx", "8");
      shape.setAttribute("cy", "8");
      shape.setAttribute("r", "5.5");
    }
    shape.setAttribute("fill", filled ? color : "none");
    shape.setAttribute("stroke", color);
    shape.setAttribute("stroke-width", filled ? "0" : "1.6");
    svg.appendChild(shape);
    return svg;
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
    var own = spec.override;
    if (own && own.on) {
      // The page's preset, overridden from the panel. A master curve keeps the reduced
      // frequency along x: superposition shifts along it.
      var ownX = preset.superpose || !own.x ? firstPresent(active, preset.x) : own.x;
      var notOwnX = function (i) {
        return i !== ownX;
      };
      return {
        active: active,
        x: ownX,
        left: (own.left || []).filter(notOwnX),
        right: (own.right || []).filter(notOwnX),
        overridden: true
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
    // Each column by its own unit: a recognised quantity is converted to its display unit, so
    // it has one; an unrecognised column is drawn as the file gave it, so its unit is read
    // from its own column in every table (and may differ from one table to the next).
    var units = idents.map(function (ident) {
      if (quantityOf(ident)) {
        return unitText(ident, null);
      }
      var seen = [];
      entries.forEach(function (entry) {
        var column = findColumn(entry.block, ident);
        var u = column ? unitText(ident, column) : null;
        if (u !== null && seen.indexOf(u) < 0) {
          seen.push(u);
        }
      });
      return seen.filter(function (u) {
        return u;
      }).join(" or ");
    });
    var same = units.every(function (u) {
      return u === units[0];
    });
    if (same) {
      var text = idents.map(identSymbol).join(", ");
      return units[0] ? text + " (" + units[0] + ")" : text;
    }
    return idents.map(function (ident, i) {
      return units[i] ? identSymbol(ident) + " (" + units[i] + ")" : identSymbol(ident);
    }).join(", ");
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

  // The id of the sweep the server will take as the reference: the one measured nearest the
  // chosen temperature (or the middle one), the first of them in temperature order on a tie,
  // exactly as /api/rheometry/tts picks it.
  function referenceId(list, reference) {
    var sorted = list.slice().sort(function (a, b) {
      return a.block.temperature - b.block.temperature;
    });
    if (!sorted.length) {
      return null;
    }
    var target = isFinite(reference) ? reference
      : sorted[Math.floor(sorted.length / 2)].block.temperature;
    var best = sorted[0];
    sorted.forEach(function (entry) {
      if (Math.abs(entry.block.temperature - target) < Math.abs(best.block.temperature - target)) {
        best = entry;
      }
    });
    return best.id;
  }

  // The reference is never shifted, so a value typed for a sweep before it became the
  // reference is dropped rather than kept to reappear when the reference moves on.
  function dropReferenceManual(id) {
    if (id && spec.tts.manual && Object.prototype.hasOwnProperty.call(spec.tts.manual, id)) {
      delete spec.tts.manual[id];
      saveSpec();
    }
  }

  function ttsRequest() {
    var reference = parseFloat(spec.tts.reference);
    var list = ttsEntries();
    dropReferenceManual(referenceId(list, reference));
    var manual = {};
    Object.keys(spec.tts.manual || {}).forEach(function (id) {
      var value = parseFloat(spec.tts.manual[id]);
      if (isFinite(value)) {
        manual[id] = value;
      }
    });
    return {
      curves: list.map(function (entry) {
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
        ttsError = "";
        ttsSent = "";
        renderTTS();
        draw();
        return;
      }
      if (sent === ttsSent && (tts || ttsPending)) {
        return;
      }
      ttsSent = sent;
      ttsPending = true;
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
          ttsPending = false;
          tts = result;
          ttsError = "";
          dropReferenceManual(result.reference_id);
          renderTTS();
          draw();
        })
        .catch(function (error) {
          if (sent !== ttsSent) {
            return; // an old request failing must not hide a newer answer
          }
          ttsPending = false;
          ttsSent = ""; // so the same request can be tried again
          tts = null;
          ttsError = String(error.message || error);
          renderTTS();
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
      var color = sampleColor(entry, n);
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
        var marker = seriesStyle(s.ident, index);
        var paint = seriesPaint(marker, entry, n, index, single);
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
          meta: { exportColor: paint },
          marker: {
            symbol: marker.filled ? marker.symbol : marker.symbol + "-open",
            size: marker.size,
            color: paint,
            line: { color: paint, width: marker.filled ? 0 : 1.5 }
          },
          line: {
            color: paint,
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
        var marker = seriesStyle(s.ident, index);
        var keyInk = marker.color || ink;
        traces.push({
          type: "scatter",
          mode: "markers",
          meta: { exportColor: marker.color || "#6b6b6b" },
          name: identSymbol(s.ident) + (s.axis === "y2" ? " (right)" : ""),
          x: [null],
          y: [null],
          yaxis: s.axis,
          legendgroup: "key",
          hoverinfo: "skip",
          marker: {
            symbol: marker.filled ? marker.symbol : marker.symbol + "-open",
            size: marker.size,
            color: keyInk,
            line: { color: keyInk, width: marker.filled ? 0 : 1.5 }
          }
        });
      });
    }
    return { plan: p, traces: traces, notes: notes };
  }

  // One end of an axis as Plotly wants it (log10 on a log axis), or null to fit the data.
  function limitValue(text, log) {
    var value = parseFloat(text);
    if (!isFinite(value) || (log && value <= 0)) {
      return null;
    }
    return log ? Math.log10(value) : value;
  }

  // [low, high] with either end null when only the other was set, or null to fit both.
  function range(axis) {
    var lo = limitValue(axis.min, axis.log);
    var hi = limitValue(axis.max, axis.log);
    if ((lo === null && hi === null) || lo === hi) {
      return null;
    }
    return [lo, hi];
  }

  // What the limit fields of an axis amount to, in a sentence under them.
  function limitHint(axis) {
    var lo = parseFloat(axis.min);
    var hi = parseFloat(axis.max);
    if (axis.log && ((isFinite(lo) && lo <= 0) || (isFinite(hi) && hi <= 0))) {
      return "A log axis only takes limits above zero; a limit at or below zero is ignored.";
    }
    if (isFinite(lo) && isFinite(hi) && lo === hi) {
      return "From and To are equal, so the axis fits the data.";
    }
    if (isFinite(lo) && isFinite(hi) && lo > hi) {
      return "From is above To, so the axis runs backwards.";
    }
    return "Leave an end empty to fit it to the data.";
  }

  // Where the legend can go: the same places the exported figure offers.
  var LEGENDS = {
    above: { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom" },
    // Below: at the bottom of the whole graph, under the axis title.
    below: { orientation: "h", x: 0.5, xanchor: "center", y: 0, yref: "container",
      yanchor: "bottom" },
    "inside-top-left": { x: 0.02, y: 0.98, xanchor: "left", yanchor: "top" },
    "inside-top-center": { x: 0.5, y: 0.98, xanchor: "center", yanchor: "top" },
    "inside-top-right": { x: 0.98, y: 0.98, xanchor: "right", yanchor: "top" },
    "inside-center-left": { x: 0.02, y: 0.5, xanchor: "left", yanchor: "middle" },
    "inside-center": { x: 0.5, y: 0.5, xanchor: "center", yanchor: "middle" },
    "inside-center-right": { x: 0.98, y: 0.5, xanchor: "right", yanchor: "middle" },
    "inside-bottom-left": { x: 0.02, y: 0.02, xanchor: "left", yanchor: "bottom" },
    "inside-bottom-center": { x: 0.5, y: 0.02, xanchor: "center", yanchor: "bottom" },
    "inside-bottom-right": { x: 0.98, y: 0.02, xanchor: "right", yanchor: "bottom" },
    outside: { x: 1.02, y: 1, xanchor: "left", yanchor: "top" }
  };

  var LEGEND_CHOICES = [
    ["above", "Above the plot"],
    ["below", "Below the plot"],
    ["inside-top-left", "Inside, top left"],
    ["inside-top-center", "Inside, top centre"],
    ["inside-top-right", "Inside, top right"],
    ["inside-center-left", "Inside, middle left"],
    ["inside-center", "Inside, centre"],
    ["inside-center-right", "Inside, middle right"],
    ["inside-bottom-left", "Inside, bottom left"],
    ["inside-bottom-center", "Inside, bottom centre"],
    ["inside-bottom-right", "Inside, bottom right"],
    ["outside", "Outside, right"],
    ["custom", "Where I dragged it"],
    ["none", "No legend"]
  ];

  function fraction(value, fallback) {
    var v = parseFloat(value);
    return isFinite(v) ? Math.min(2, Math.max(-1, v)) : fallback;
  }

  function legendPlace() {
    if (spec.style.legend === "custom") {
      return { x: fraction(spec.style.legendX, 0.02), y: fraction(spec.style.legendY, 0.98),
        xanchor: "left", yanchor: "top" };
    }
    return LEGENDS[spec.style.legend];
  }

  // The figure's size in mm: a named size, or the width and height typed in.
  function figureSize() {
    if (spec.style.size === "custom") {
      var mm = function (value, fallback) {
        var v = parseFloat(value);
        return isFinite(v) ? Math.min(600, Math.max(20, v)) : fallback;
      };
      return { width: mm(spec.style.widthMm, 85), height: mm(spec.style.heightMm, 65) };
    }
    return P.sizes[spec.style.size] || P.sizes.single;
  }

  function sizeHint() {
    var size = figureSize();
    return "= " + (size.width / 25.4).toFixed(2) + " × " + (size.height / 25.4).toFixed(2) +
      " in. From 20 to 600 mm each way.";
  }

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
      // An end left empty still fits the data.
      layout.autorange = r[0] === null ? "min" : r[1] === null ? "max" : false;
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
      return [ttsError ? "The shift factors could not be worked out: " + ttsError
        : "Working out the shift factors…", null];
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
    var size = figureSize();
    var width = target.clientWidth || 600;
    var height = Math.max(260, Math.min(width * size.height / size.width,
      window.innerHeight * 0.78));
    var px = (Number(spec.style.fontSize) || 9) * 1.333;
    var hasY2 = result.traces.some(function (t) {
      return t.yaxis === "y2";
    });
    var legend = legendPlace();
    var ink = token("--ink", "#1e2832");

    var layout = {
      height: height,
      margin: { l: 72, r: hasY2 ? 72 : 20, t: spec.style.legend === "above" ? 44 : 16,
        b: spec.style.legend === "below" ? 96 : 58 },
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
    // crowds the axis; labelling only the 2s and 5s keeps it readable. Under half a decade
    // (a near-constant tan δ in a time sweep) those digits may not fall on the axis at all,
    // so Plotly's own evenly spaced labels are kept.
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
        if (span >= 0.5 && span < 2) {
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
      toImageButtonOptions: { format: "png", scale: 3, filename: key.replace(/_/g, "-") },
      // The legend can be dragged; where it is dropped becomes the "custom" place.
      edits: { legendPosition: true }
    });
    if (!target.ladaLegendDrag && target.on) {
      target.ladaLegendDrag = true;
      target.on("plotly_relayout", function (change) {
        var moved = Object.keys(change || {}).some(function (k) {
          return k === "legend.x" || k === "legend.y";
        });
        if (moved) {
          legendDropped(target);
        }
      });
    }
  }

  // The dragged legend's top left corner as fractions of the plot area, which is what the
  // exported figure anchors a "custom" legend by.
  function legendDropped(target) {
    var legendNode = target.querySelector(".legend");
    var svg = target.querySelector(".main-svg");
    var plotArea = target._fullLayout && target._fullLayout._size;
    if (!legendNode || !svg || !plotArea || !plotArea.w || !plotArea.h) {
      return;
    }
    var box = legendNode.getBoundingClientRect();
    var origin = svg.getBoundingClientRect();
    var x = (box.left - origin.left - plotArea.l) / plotArea.w;
    var y = 1 - (box.top - origin.top - plotArea.t) / plotArea.h;
    spec.style.legend = "custom";
    spec.style.legendX = Math.round(fraction(x, 0.02) * 1000) / 1000;
    spec.style.legendY = Math.round(fraction(y, 0.98) * 1000) / 1000;
    saveSpec();
    renderForm();
    draw();
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
    var size = figureSize();
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
      legend_x: fraction(spec.style.legendX, 0.02),
      legend_y: fraction(spec.style.legendY, 0.98),
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
          color: (t.meta && t.meta.exportColor) || t.marker.color,
          marker: symbol.replace(/-open$/, ""),
          marker_size: t.marker.size,
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
        var size = figureSize();
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

  function renderTTS() {
    var error = ttsError;
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
      notes.hidden = !error; // nothing stale from an earlier answer
      return;
    }
    tts.shifts.forEach(function (shift) {
      var tr = el("tr");
      // Only the reference sweep itself is fixed; another measured at the same temperature
      // gets a shift of its own. (An older server without reference_id: by temperature.)
      var isRef = tts.reference_id ? shift.id === tts.reference_id
        : shift.temperature === tts.reference;
      var lost = !isRef && !shift.manual && shift.auto_log_at === null;
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
      } else if (lost) {
        source.textContent = "not found";
        source.title = "Overlap found no shift: the sweep is drawn beside its neighbour and " +
          "left out of the fits. Type a value to set it by hand.";
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
          // Hand-set: open; not found (left out of the fits): a cross.
          return s.fitted === false ? "x-thin-open" : s.manual ? "circle-open" : "circle";
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
          s.manual ? "hand" : s.fitted === false ? "not found" : "overlap"].join(","));
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
    if (skippedDatasets) {
      var damaged = el("p", "studio-hint",
        skippedDatasets === 1 ? "One saved file could not be read and is left out. "
          : skippedDatasets + " saved files could not be read and are left out. ");
      var fix = el("a", "", "Remove it on My data");
      fix.href = "/rheometer/my-data";
      damaged.appendChild(fix);
      damaged.appendChild(document.createTextNode("."));
      host.appendChild(damaged);
    }
    if (!entries.length) {
      var empty = el("p", "studio-hint",
        skippedDatasets ? "No other files imported yet. " : "No files imported yet. ");
      var link = el("a", "", "Import files on My data");
      link.href = "/rheometer/my-data";
      empty.appendChild(link);
      empty.appendChild(document.createTextNode(", and they will be listed here."));
      host.appendChild(empty);
      return;
    }
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
      // Colour names a sample only when several are drawn; one sample's quantities each have
      // their own colour, shown under Series.
      if (n >= 0 && active.length > 1) {
        swatch.style.background = sampleColor(entry, n);
        swatch.style.borderColor = sampleColor(entry, n);
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

  // A control that edits spec.styles rather than a spec path.
  function styleControl(node, kind, id, prop) {
    node.dataset.style = kind;
    node.dataset.styleId = id;
    node.dataset.styleProp = prop;
    return node;
  }

  function colorInput(value) {
    var node = el("input");
    node.type = "color";
    node.value = hex(value) || "#2f6fd0";
    return node;
  }

  // The horizontal-axis choice and the left and right checklists: all of Custom plot, and on
  // every other page the way to put something else on an axis.
  function axisChooser(container, p, base, idents) {
    if (base === "override" && preset.superpose) {
      container.appendChild(el("p", "studio-hint",
        "The horizontal axis stays the shifted frequency aT·ω: superposition works along it."));
    } else {
      var xs = idents.slice();
      if (p.x && xs.indexOf(p.x) < 0) {
        xs.unshift(p.x);
      }
      container.appendChild(field("Horizontal axis", select(base + ".x",
        xs.map(function (ident) {
          return [ident, identText(ident)];
        }), p.x)));
    }
    ["left", "right"].forEach(function (side) {
      var list = el("div", "studio-checklist");
      list.appendChild(el("span", "studio-checklist-title",
        side === "left" ? "Left axis" : "Right axis"));
      idents.forEach(function (ident) {
        if (ident !== p.x) {
          list.appendChild(checkbox(null, identText(ident), p[side].indexOf(ident) >= 0,
            { list: base + "." + side, ident: ident }));
        }
      });
      container.appendChild(list);
    });
  }

  var chooserOpen = false; // whether "Choose every axis yourself" was left open

  function renderForm() {
    var form = $("studio-form");
    var active = document.activeElement;
    var focused = active && form.contains(active)
      ? { path: active.dataset.path || "", ident: active.dataset.ident || "",
        list: active.dataset.list || "",
        style: (active.dataset.styleId || "") + "|" + (active.dataset.styleProp || "") }
      : null;
    fieldCount = 0;
    form.textContent = "";
    if (!entries.length) {
      // Nothing imported: controls with nothing to act on would only puzzle, so say what
      // comes first instead.
      form.appendChild(el("p", "studio-hint",
        "The graph's settings appear here once there is something to draw."));
      return;
    }
    var p = plan();

    // What to plot
    if (key === "custom") {
      var idents = availableIdents(p.active.length ? p.active : entries);
      var what = group("Axes");
      if (!idents.length) {
        what.appendChild(el("p", "studio-hint", "Tick a table to choose from its columns."));
      } else {
        axisChooser(what, p, "custom", idents);
      }
      form.appendChild(what);
    } else {
      var plotted = group("What is plotted");
      if (preset.right.length && !p.overridden) {
        plotted.appendChild(field("Also show on the right", select("right",
          [["", "Nothing"]].concat(preset.right.map(function (q) {
            return [q, byQ[q] ? byQ[q].name + " " + byQ[q].symbol : q];
          })), spec.right)));
      }
      var chosen = availableIdents(p.active.length ? p.active : entries);
      if (chosen.length) {
        var more = el("details", "studio-more");
        more.open = chooserOpen || !!p.overridden;
        more.addEventListener("toggle", function () {
          chooserOpen = more.open;
        });
        more.appendChild(el("summary", "", "Choose every axis yourself"));
        axisChooser(more, p, "override", chosen);
        if (p.overridden) {
          var back = el("button", "quiet-button", "Back to this page's usual axes");
          back.type = "button";
          back.addEventListener("click", function () {
            spec.override = { on: false, x: "", left: [], right: [] };
            spec.axes.x.log = !!preset.logX;
            spec.axes.y.log = !!preset.logY;
            spec.axes.y2.log = !!(spec.right && byQ[spec.right] && byQ[spec.right].log);
            saveSpec();
            renderForm();
            draw();
          });
          more.appendChild(back);
        }
        plotted.appendChild(more);
      }
      if (plotted.children.length > 1) {
        form.appendChild(plotted);
      }
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
      box.appendChild(head);
      var scale = select("axes." + a[0] + ".log",
        [["false", "Linear"], ["true", "Logarithmic"]], String(!!spec.axes[a[0]].log));
      scale.dataset.bool = "1";
      box.appendChild(field("Scale", scale));
      var limits = el("div", "studio-pair");
      limits.appendChild(field("From", input("axes." + a[0] + ".min", "number",
        spec.axes[a[0]].min, { step: "any", placeholder: "auto" })));
      limits.appendChild(field("To", input("axes." + a[0] + ".max", "number",
        spec.axes[a[0]].max, { step: "any", placeholder: "auto" })));
      box.appendChild(limits);
      var hint = el("small", "studio-field-hint", limitHint(spec.axes[a[0]]));
      hint.id = "limit-hint-" + a[0];
      box.appendChild(hint);
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
    if (spec.style.size === "custom") {
      var dims = el("div", "studio-pair");
      dims.appendChild(field("Width, mm", input("style.widthMm", "number", spec.style.widthMm,
        { min: 20, max: 600, step: 0.5 })));
      dims.appendChild(field("Height, mm", input("style.heightMm", "number", spec.style.heightMm,
        { min: 20, max: 600, step: 0.5 })));
      look.appendChild(dims);
      var inches = el("small", "studio-field-hint", sizeHint());
      inches.id = "size-hint";
      look.appendChild(inches);
    }
    look.appendChild(field("Colours", select("style.palette",
      Object.keys(P.palettes).map(function (k) {
        return [k, P.palettes[k].name];
      }), spec.style.palette)));
    if (p.active.length > 1) {
      var swatches = el("div", "studio-swatches");
      p.active.forEach(function (entry, n) {
        swatches.appendChild(field(labelOf(entry, p.active),
          styleControl(colorInput(sampleColor(entry, n)), "sample", entry.id, "color")));
      });
      look.appendChild(swatches);
      if (Object.keys(spec.styles.samples || {}).length) {
        var palette = el("button", "quiet-button", "Use the palette's colours again");
        palette.type = "button";
        palette.addEventListener("click", function () {
          spec.styles.samples = {};
          saveSpec();
          renderData();
          renderForm();
          draw();
        });
        look.appendChild(palette);
      }
    }
    look.appendChild(field("Draw as", select("style.mode",
      [["markers", "Points"], ["both", "Points and lines"], ["lines", "Lines"]],
      spec.style.mode)));
    var pair = el("div", "studio-pair");
    pair.appendChild(field("Point size", input("style.markerSize", "number",
      spec.style.markerSize, { min: 2, max: 16, step: 1 }),
    "Each quantity can have its own under Markers."));
    pair.appendChild(field("Line width", input("style.lineWidth", "number",
      spec.style.lineWidth, { min: 0.5, max: 5, step: 0.25 })));
    look.appendChild(pair);
    var pair2 = el("div", "studio-pair");
    pair2.appendChild(field("Font", select("style.font",
      [["sans", "Sans-serif"], ["serif", "Serif"]], spec.style.font)));
    pair2.appendChild(field("Font size, pt", input("style.fontSize", "number",
      spec.style.fontSize, { min: 6, max: 24, step: 0.5 })));
    look.appendChild(pair2);
    look.appendChild(field("Legend", select("style.legend", LEGEND_CHOICES, spec.style.legend),
      "Or drag the legend on the Explore graph: the exported figure puts it there too."));
    look.appendChild(checkbox("style.grid", "Grid lines", spec.style.grid));
    form.appendChild(look);

    // Markers: one row per quantity on show. Unset choices follow the convention (G′ filled,
    // G″ open, each sample in its colour) and the point size above.
    var shown = [];
    p.left.concat(p.right).forEach(function (ident) {
      if (shown.indexOf(ident) < 0) {
        shown.push(ident);
      }
    });
    if (shown.length) {
      var marks = group("Series");
      var oneSample = p.active.length === 1;
      marks.appendChild(el("p", "studio-hint", oneSample
        ? "Each quantity has its own colour and marker."
        : "Colour tells the samples apart; the marker tells the quantities apart."));
      shown.forEach(function (ident, index) {
        var st = seriesStyle(ident, index);
        var own = (spec.styles.series || {})[ident] || {};
        var shownColour = oneSample ? seriesPaint(st, p.active[0], 0, index, true)
          : st.color || token("--ink-muted", "#6b6b6b");
        var row = el("div", "studio-series");
        row.style.setProperty("--series-color", shownColour);
        var head = el("div", "studio-series-head");
        head.appendChild(markerPreview(st.symbol, st.filled, shownColour));
        head.appendChild(el("span", "studio-axis-name", identText(ident)));
        head.appendChild(el("span", "studio-series-axis",
          p.right.indexOf(ident) >= 0 ? "Right axis" : "Left axis"));
        row.appendChild(head);
        var shape = el("div", "studio-pair");
        shape.appendChild(field("Shape",
          styleControl(select("", P.shapes, st.symbol), "series", ident, "symbol")));
        shape.appendChild(field("Fill", styleControl(
          select("", [["true", "Filled"], ["false", "Open"]], String(st.filled)),
          "series", ident, "filled")));
        row.appendChild(shape);
        var paint = el("div", "studio-pair");
        var size = el("input");
        size.type = "number";
        size.min = "1";
        size.max = "30";
        size.step = "0.5";
        size.placeholder = String(Number(spec.style.markerSize) || 7);
        size.value = own.size === undefined || own.size === null ? "" : own.size;
        paint.appendChild(field("Size", styleControl(size, "series", ident, "size")));
        var colour = styleControl(colorInput(st.color ||
          (oneSample ? shownColour : p.active[0] ? sampleColor(p.active[0], 0) : "")),
        "series", ident, "color");
        colour.disabled = !st.color;
        var colourField = field("Colour", colour);
        var ownColour = checkbox(null, oneSample ? "Choose colour" : "Same for every sample",
          !!st.color);
        var ownBox = styleControl(ownColour.querySelector("input"), "series", ident, "ownColor");
        ownBox.dataset.colorFor = colour.id;
        colourField.appendChild(ownColour);
        paint.appendChild(colourField);
        row.appendChild(paint);
        marks.appendChild(row);
      });
      if (Object.keys(spec.styles.series || {}).length) {
        var usual = el("button", "quiet-button", "Back to the usual markers");
        usual.type = "button";
        usual.addEventListener("click", function () {
          spec.styles.series = {};
          saveSpec();
          renderForm();
          draw();
        });
        marks.appendChild(usual);
      }
      form.appendChild(marks);
    }

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
            (node.dataset.list || "") === focused.list &&
            (node.dataset.styleId || "") + "|" + (node.dataset.styleProp || "") === focused.style;
        })[0];
      if (again) {
        again.focus();
      }
    }
  }

  // Structural choices change which controls exist; everything else only redraws.
  function isStructural(path) {
    return path === "right" || path === "custom.x" || path === "override.x" ||
      path === "style.palette" || path === "style.markerSize" || path === "style.size" ||
      path.indexOf("units.") === 0;
  }

  // The first change to what a graph page plots starts from what it was plotting.
  function seedOverride() {
    if (!spec.override.on) {
      var current = plan();
      spec.override = {
        on: true,
        x: current.x,
        left: current.left.slice(),
        right: current.right.slice()
      };
    }
  }

  function onStyleChange(event, target) {
    var id = target.dataset.styleId;
    var prop = target.dataset.styleProp;
    if (target.dataset.style === "sample") {
      if (hex(target.value)) {
        spec.styles.samples[id] = hex(target.value);
      }
      saveSpec();
      renderData();
      redrawSoon();
      return;
    }
    if (event.type !== "change" && target.type !== "color" && target.type !== "number") {
      return;
    }
    var own = spec.styles.series[id] || (spec.styles.series[id] = {});
    if (prop === "filled") {
      own.filled = target.value === "true";
    } else if (prop === "size") {
      var size = parseFloat(target.value);
      if (isFinite(size) && size >= 1 && size <= 30) {
        own.size = size;
      } else {
        delete own.size;
      }
    } else if (prop === "ownColor") {
      var colour = $(target.dataset.colorFor);
      colour.disabled = !target.checked;
      if (target.checked) {
        own.color = hex(colour.value);
      } else {
        delete own.color;
      }
    } else if (prop === "color") {
      own.color = hex(target.value) || own.color;
    } else if (prop === "symbol" && knownShape(target.value)) {
      own.symbol = target.value;
    }
    saveSpec();
    redrawSoon();
    if (event.type === "change") {
      renderForm(); // so each series card's preview shows the new look
    }
  }

  function onFormChange(event) {
    var target = event.target;
    if (target.dataset.style) {
      onStyleChange(event, target);
      return;
    }
    if (target.dataset.list) {
      if (event.type !== "change") {
        return;
      }
      var listPath = target.dataset.list;
      var listBase = listPath.split(".")[0];
      if (listBase === "override") {
        seedOverride();
      }
      var ident = target.dataset.ident;
      var list = (getPath(listPath) || []).filter(function (i) {
        return i !== ident;
      });
      var onLeft = /\.left$/.test(listPath);
      if (target.checked) {
        list.push(ident);
        // A quantity sits on one side only.
        var other = listBase + (onLeft ? ".right" : ".left");
        setPath(other, (getPath(other) || []).filter(function (i) {
          return i !== ident;
        }));
      }
      // The first quantity put on an axis sets its scale, the way the presets do.
      var axisKey = onLeft ? "y" : "y2";
      var q = quantityOf(ident);
      if (target.checked && list.length === 1 && q && byQ[q]) {
        spec.axes[axisKey].log = !!byQ[q].log;
      }
      setPath(listPath, list);
      saveSpec();
      renderForm();
      draw();
      return;
    }
    var path = target.dataset.path;
    if (!path) {
      return;
    }
    if (path.indexOf("override.") === 0) {
      seedOverride();
    }
    setPath(path, target.type === "checkbox" ? target.checked
      : target.dataset.bool ? target.value === "true" : target.value);
    if ((path === "style.widthMm" || path === "style.heightMm") && $("size-hint")) {
      $("size-hint").textContent = sizeHint();
    }
    var axisMatch = /^axes\.(x|y|y2)\./.exec(path);
    if (axisMatch && $("limit-hint-" + axisMatch[1])) {
      $("limit-hint-" + axisMatch[1]).textContent = limitHint(spec.axes[axisMatch[1]]);
    }
    // A quantity that spans decades (tan δ, |η*|, stress) starts on a log axis.
    if (path === "right" && event.type === "change" && byQ[target.value]) {
      spec.axes.y2.log = !!byQ[target.value].log;
    }
    if ((path === "custom.x" || path === "override.x") && event.type === "change") {
      var xq = quantityOf(target.value);
      spec.axes.x.log = !!(xq && byQ[xq] && byQ[xq].log);
    }
    saveSpec();
    if (path === "style.palette") {
      renderData();
    }
    if (event.type === "change" && isStructural(path)) {
      renderForm();
      draw();
      return;
    }
    redrawSoon();
  }

  $("studio-form").addEventListener("change", onFormChange);
  $("studio-form").addEventListener("input", function (event) {
    var typed = event.target.type;
    if (typed === "text" || typed === "number" || typed === "color") {
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
        var readable = results[1].filter(function (dataset) {
          return store.validDataset(dataset, function (key) {
            return !!byQ[key];
          });
        });
        skippedDatasets = results[1].length - readable.length;
        readable.forEach(function (dataset) {
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
