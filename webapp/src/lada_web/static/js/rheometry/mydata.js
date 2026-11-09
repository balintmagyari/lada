/* My data: import rheometer files, show what was read, let the user correct it.

   Files go to /api/rheometry/parse and come back as datasets, which window.ladaRheoStore keeps
   in this browser. Everything a file contributes (names, headers, units, warnings) reaches the
   page through textContent, never as HTML. */

(function () {
  "use strict";

  var store = window.ladaRheoStore;
  var registry = null; // from /api/rheometry/quantities
  var byQuantity = {};
  var editing = null; // { datasetId, blockId, columnKey }
  var openTables = {}; // block id -> how many rows of its data table are shown
  var TABLE_STEP = 500;

  // Where each kind of test is drawn, and the section colour it wears.
  var PAGES = {
    frequency_sweep: ["/rheometer/frequency-sweep", "oscillation"],
    amplitude_sweep: ["/rheometer/amplitude-sweep", "oscillation"],
    temperature_sweep: ["/rheometer/temperature-sweep", "oscillation"],
    time_sweep: ["/rheometer/time-sweep", "oscillation"],
    flow_curve: ["/rheometer/flow-curve", "flow"],
    creep: ["/rheometer/creep", "flow"],
    stress_relaxation: ["/rheometer/stress-relaxation", "flow"],
    unknown: ["/rheometer/custom-plot", "custom"]
  };
  var PROJECT_KIND = "lada-rheometer-project";

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

  function status(message, kind) {
    var box = $("mydata-status");
    box.textContent = "";
    if (!message) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    box.className = "mydata-status" + (kind ? " is-" + kind : "");
    if (Array.isArray(message)) {
      message.forEach(function (line, i) {
        box.appendChild(el(i === 0 ? "strong" : "p", "", line));
      });
    } else {
      box.textContent = message;
    }
  }

  // FastAPI's validation errors arrive as a list of {loc, msg}; anything else as a string.
  function detailText(detail) {
    if (Array.isArray(detail)) {
      return detail.map(function (d) {
        return d && d.msg ? String(d.msg).replace(/^Value error, /, "") : String(d);
      }).join(" ");
    }
    return detail ? String(detail) : "";
  }

  function readJson(response, fallback) {
    return response.json().then(
      function (body) {
        if (!response.ok) {
          throw new Error(detailText(body && body.detail) || fallback);
        }
        return body;
      },
      function () {
        throw new Error(fallback);
      }
    );
  }

  function loadRegistry() {
    return fetch("/api/rheometry/quantities")
      .then(function (r) {
        return readJson(r, "The list of quantities could not be loaded.");
      })
      .then(function (data) {
        registry = data;
        data.quantities.forEach(function (q) {
          byQuantity[q.key] = q;
        });
      });
  }

  // --- Rendering -------------------------------------------------------------------------

  function columnText(column) {
    var spec = column.quantity && byQuantity[column.quantity];
    if (!spec) {
      return column.source_unit ? column.label + " (" + column.source_unit + ")" : column.label;
    }
    return spec.unit ? spec.symbol + " (" + spec.unit + ")" : spec.symbol;
  }

  function isEditing(dataset, block, column) {
    return !!editing && editing.datasetId === dataset.id && editing.blockId === block.id &&
      editing.columnKey === column.key;
  }

  function chip(dataset, block, column) {
    var spec = column.quantity && byQuantity[column.quantity];
    var button = el("button", "col-chip", columnText(column));
    button.type = "button";
    if (column.derived) {
      button.classList.add("is-derived");
      button.disabled = true;
      button.title = (spec ? spec.name : column.label) + ", worked out from the measured columns";
      return button;
    }
    if (!spec) {
      button.classList.add(column.is_index ? "is-index" : "is-unknown");
      button.title = column.is_index
        ? "“" + column.label + "” numbers the points. Click to say what it is instead."
        : "“" + column.label + "” was not recognised. Click to say what it is.";
    } else {
      button.title = "Read from “" + column.label + "”" +
        (column.source_unit ? " in " + column.source_unit : "") + ". Click to correct.";
    }
    button.setAttribute("aria-expanded", isEditing(dataset, block, column) ? "true" : "false");
    button.addEventListener("click", function () {
      editing = isEditing(dataset, block, column)
        ? null
        : { datasetId: dataset.id, blockId: block.id, columnKey: column.key };
      render();
    });
    return button;
  }

  // --- The data as a table -------------------------------------------------------------

  // A number as a spreadsheet would show it: six significant figures, scientific notation
  // only for very large or very small values.
  function cellText(value) {
    if (value === null || value === undefined || typeof value !== "number" || !isFinite(value)) {
      return "";
    }
    var size = Math.abs(value);
    if (size !== 0 && (size >= 1e6 || size < 1e-3)) {
      return value.toExponential(4);
    }
    return String(Number(value.toPrecision(6)));
  }

  function columnName(column) {
    var spec = column.quantity && byQuantity[column.quantity];
    return spec ? spec.symbol : column.label;
  }

  function addRows(body, block, from, to) {
    for (var i = from; i < to; i++) {
      var tr = el("tr");
      var index = el("th", "ds-grid-index", String(i + 1));
      index.scope = "row";
      tr.appendChild(index);
      block.columns.forEach(function (column) {
        tr.appendChild(el("td", column.derived ? "is-derived" : "", cellText(column.values[i])));
      });
      body.appendChild(tr);
    }
  }

  function tableCsv(block) {
    var quote = function (text) {
      return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
    };
    var lines = [block.columns.map(function (column) {
      return quote(column.unit ? columnName(column) + " (" + column.unit + ")" : columnName(column));
    }).join(",")];
    for (var i = 0; i < block.n_points; i++) {
      lines.push(block.columns.map(function (column) {
        var v = column.values[i];
        return typeof v === "number" && isFinite(v) ? String(v) : "";
      }).join(","));
    }
    return lines.join("\n") + "\n";
  }

  function downloadCsv(dataset, block) {
    var stem = dataset.filename.replace(/\.[^.]+$/, "");
    var name = (stem + (dataset.blocks.length > 1 ? "-" + block.name : ""))
      .replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80) + ".csv";
    var url = URL.createObjectURL(new Blob([tableCsv(block)], { type: "text/csv" }));
    var link = el("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 1000);
  }

  // The table's numbers, laid out like a spreadsheet: a header that stays in view naming each
  // column and its unit, numbered rows, and the columns the app worked out set apart.
  function dataTable(dataset, block) {
    var wrap = el("div", "ds-table");
    var scroller = el("div", "ds-table-scroll");
    scroller.tabIndex = 0;
    scroller.setAttribute("role", "region");
    scroller.setAttribute("aria-label", "The numbers in " + block.name);
    var table = el("table", "ds-grid");
    var head = el("thead");
    var row = el("tr");
    var corner = el("th", "ds-grid-index", "#");
    corner.scope = "col";
    row.appendChild(corner);
    block.columns.forEach(function (column) {
      var th = el("th", column.derived ? "is-derived" : "");
      th.scope = "col";
      th.title = column.derived ? "Worked out from the measured columns"
        : "Read from “" + column.label + "”" +
          (column.source_unit ? " in " + column.source_unit : "");
      th.appendChild(el("span", "ds-grid-name", columnName(column)));
      th.appendChild(el("span", "ds-grid-unit",
        (column.unit || "no unit") + (column.derived ? " · worked out" : "")));
      row.appendChild(th);
    });
    head.appendChild(row);
    table.appendChild(head);
    var body = el("tbody");
    var shown = Math.min(openTables[block.id] || TABLE_STEP, block.n_points);
    addRows(body, block, 0, shown);
    table.appendChild(body);
    scroller.appendChild(table);
    wrap.appendChild(scroller);

    var foot = el("div", "ds-table-foot");
    var count = el("span", "ds-table-count");
    var more = el("button", "quiet-button", "Show " + TABLE_STEP + " more rows");
    more.type = "button";
    var update = function () {
      count.textContent = shown >= block.n_points
        ? "All " + block.n_points + " rows."
        : "Rows 1 to " + shown + " of " + block.n_points + ".";
      more.hidden = shown >= block.n_points;
    };
    more.addEventListener("click", function () {
      var next = Math.min(shown + TABLE_STEP, block.n_points);
      addRows(body, block, shown, next);
      shown = next;
      openTables[block.id] = shown;
      update();
    });
    update();
    foot.appendChild(count);
    foot.appendChild(more);
    var csv = el("button", "quiet-button", "Download as CSV");
    csv.type = "button";
    csv.addEventListener("click", function () {
      downloadCsv(dataset, block);
    });
    foot.appendChild(csv);
    wrap.appendChild(foot);
    return wrap;
  }

  function option(value, label, selected) {
    var node = el("option", "", label);
    node.value = value;
    node.selected = !!selected;
    return node;
  }

  // The same spelling rules as quantities.parse_unit on the server: case kept, outer
  // brackets dropped, "·" read as ".". A spelling that only matches with its case ignored is
  // taken when one unit fits and no SI prefix flips meaning ("MPa" is never "mPa").
  function unitKey(text) {
    return String(text).normalize("NFKC").replace(/\s+/g, " ").replace(/^[\[\]() ]+|[\[\]() ]+$/g, "")
      .replace(/[·⋅]/g, ".").replace(/μ/g, "µ");
  }

  function allUnitKeys() {
    var keys = {};
    (registry ? registry.quantities : []).forEach(function (q) {
      q.units.forEach(function (u) {
        [u.symbol].concat(u.aliases || []).forEach(function (name) {
          var key = unitKey(name);
          if (key) {
            keys[key.toLowerCase()] = true;
          }
        });
      });
    });
    return keys;
  }

  function prefixFlipped(given, name, known) {
    var first = name.search(/\p{L}/u);
    if (first < 0 || "mM".indexOf(name[first]) < 0 || given[first] === name[first]) {
      return false;
    }
    return !!known[name.slice(first + 1).toLowerCase()];
  }

  // The unit of `units` that `text` spells, or null. An empty text is no unit given: the
  // server assumed the canonical unit then, so that is the one to show (never "no unit").
  function matchUnit(units, text) {
    if (text === null || text === undefined || !units.length) {
      return null;
    }
    if (String(text).trim() === "") {
      return units[0];
    }
    var wanted = unitKey(text);
    var names = [];
    units.forEach(function (u) {
      [u.symbol].concat(u.aliases || []).forEach(function (name) {
        names.push([u, unitKey(name)]);
      });
    });
    var exact = names.filter(function (pair) {
      return pair[1] === wanted;
    })[0];
    if (exact) {
      return exact[0];
    }
    var known = allUnitKeys();
    var hits = [];
    names.forEach(function (pair) {
      if (pair[1].toLowerCase() === wanted.toLowerCase() &&
          !prefixFlipped(wanted, pair[1], known) && hits.indexOf(pair[0]) < 0) {
        hits.push(pair[0]);
      }
    });
    return hits.length === 1 ? hits[0] : null;
  }

  function unitOptions(select, quantityKey, current) {
    select.textContent = "";
    var spec = byQuantity[quantityKey];
    var units = spec ? spec.units : [];
    var chosen = matchUnit(units, current) || units[0];
    units.forEach(function (u) {
      select.appendChild(option(u.symbol, u.symbol || "no unit", u === chosen));
    });
    select.disabled = !units.length;
    // No quantity, no unit to choose: hide the field rather than show an empty box.
    if (select.parentNode) {
      select.parentNode.hidden = !units.length;
    }
  }

  function editor(dataset, block, column) {
    var form = el("form", "col-editor");
    var title = el("p", "col-editor-title");
    title.appendChild(document.createTextNode("The column “"));
    title.appendChild(el("strong", "", column.label));
    title.appendChild(document.createTextNode("” holds"));
    form.appendChild(title);

    var row = el("div", "col-editor-row");
    var quantityLabel = el("label", "", "Quantity");
    var quantity = el("select");
    quantity.appendChild(option("", "Not a rheology quantity", !column.quantity));
    registry.quantities.forEach(function (q) {
      quantity.appendChild(option(q.key, q.name + " " + q.symbol, q.key === column.quantity));
    });
    quantityLabel.appendChild(quantity);

    var unitLabel = el("label", "", "Unit in the file");
    var unit = el("select");
    unitLabel.appendChild(unit);
    unitOptions(unit, column.quantity, column.source_unit);
    unitLabel.hidden = !column.quantity;
    quantity.addEventListener("change", function () {
      // Keep the file's own unit when the newly chosen quantity has it (mPa·s stays mPa·s).
      unitOptions(unit, quantity.value, column.source_unit);
    });

    row.appendChild(quantityLabel);
    row.appendChild(unitLabel);
    form.appendChild(row);

    var actions = el("div", "col-editor-actions");
    var apply = el("button", "", "Apply");
    apply.type = "submit";
    var cancel = el("button", "quiet-button", "Cancel");
    cancel.type = "button";
    cancel.addEventListener("click", function () {
      editing = null;
      render();
    });
    actions.appendChild(apply);
    actions.appendChild(cancel);
    form.appendChild(actions);

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      apply.disabled = true;
      var override = {
        key: column.key,
        quantity: quantity.value || null,
        unit: quantity.value ? unit.value : column.source_unit
      };
      fetch("/api/rheometry/rederive", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ block: block, overrides: [override] })
      })
        .then(function (r) {
          return readJson(r, "The correction could not be applied.");
        })
        .then(function (updated) {
          return store.updateBlock(dataset.id, updated);
        })
        .then(function () {
          editing = null;
          status("");
          return render();
        })
        .catch(function (error) {
          apply.disabled = false;
          status(["The correction was not applied.", String(error.message || error)], "error");
        });
    });
    setTimeout(function () {
      quantity.focus();
    }, 0);
    return form;
  }

  function blockSection(dataset, block) {
    var page = PAGES[block.test_type] || PAGES.unknown;
    var testName = registry.test_types[block.test_type] || "Not recognised";
    var section = el("section", "ds-block");
    section.dataset.section = page[1];

    var head = el("div", "ds-block-head");
    head.appendChild(el("h3", "ds-block-name", block.name));

    var facts = el("p", "ds-block-facts");
    var type = el("span", "test-chip");
    var bead = el("span", "bead");
    bead.setAttribute("aria-hidden", "true");
    type.appendChild(bead);
    type.appendChild(document.createTextNode(testName));
    facts.appendChild(type);
    var bits = [block.n_points + (block.n_points === 1 ? " point" : " points")];
    if (block.temperature !== null && block.temperature !== undefined) {
      bits.push(block.temperature + " °C");
    }
    facts.appendChild(el("span", "ds-block-meta", bits.join(", ")));
    head.appendChild(facts);

    var custom = page[0] === PAGES.unknown[0];
    var open = el("a", "ds-block-open",
      custom ? "Plot in Custom plot" : "Plot as " + testName.toLowerCase());
    open.href = page[0];
    var tableOpen = !!openTables[block.id];
    var view = el("button", "quiet-button ds-block-view", tableOpen ? "Hide data" : "View data");
    view.type = "button";
    view.setAttribute("aria-expanded", tableOpen ? "true" : "false");
    view.addEventListener("click", function () {
      if (openTables[block.id]) {
        delete openTables[block.id];
      } else {
        openTables[block.id] = TABLE_STEP;
      }
      render();
    });
    var actions = el("div", "ds-block-actions");
    actions.appendChild(view);
    actions.appendChild(open);
    head.appendChild(actions);
    section.appendChild(head);

    var chips = el("div", "col-chips");
    var derived = [];
    block.columns.forEach(function (column) {
      if (column.derived) {
        derived.push(column);
      } else {
        chips.appendChild(chip(dataset, block, column));
      }
    });
    if (derived.length) {
      chips.appendChild(el("span", "col-chips-label", "Worked out:"));
      derived.forEach(function (column) {
        chips.appendChild(chip(dataset, block, column));
      });
    }
    section.appendChild(chips);
    if (openTables[block.id]) {
      section.appendChild(dataTable(dataset, block));
    }

    if (editing && editing.datasetId === dataset.id && editing.blockId === block.id) {
      var column = block.columns.filter(function (c) {
        return c.key === editing.columnKey;
      })[0];
      if (column) {
        section.appendChild(editor(dataset, block, column));
      }
    }
    return section;
  }

  function card(dataset) {
    var article = el("article", "ds-card");
    var head = el("header", "ds-card-head");
    var title = el("div", "ds-card-title");
    title.appendChild(el("h2", "", dataset.filename));
    title.appendChild(el("span", "needs-chip", dataset.format_label));
    head.appendChild(title);

    var remove = el("button", "quiet-button", "Remove");
    remove.type = "button";
    remove.setAttribute("aria-label", "Remove " + dataset.filename);
    remove.addEventListener("click", function () {
      store.remove(dataset.id).then(function () {
        status("Removed " + dataset.filename + ".");
        return render();
      });
    });
    head.appendChild(remove);
    article.appendChild(head);

    if (dataset.warnings && dataset.warnings.length) {
      var warnings = el("ul", "ds-warnings");
      dataset.warnings.forEach(function (w) {
        warnings.appendChild(el("li", "", w));
      });
      article.appendChild(warnings);
    }
    var details = metadataList(dataset.metadata);
    if (details) {
      article.appendChild(details);
    }
    dataset.blocks.forEach(function (block) {
      article.appendChild(blockSection(dataset, block));
    });
    return article;
  }

  // What the file says about the test (operator, start, geometry, notes …), folded away so the
  // tables stay the first thing seen. Only plain text values are shown, always as text.
  function metadataList(metadata) {
    if (!metadata || typeof metadata !== "object") {
      return null;
    }
    var keys = Object.keys(metadata).filter(function (key) {
      return isText(metadata[key]) && metadata[key] !== "";
    });
    if (!keys.length) {
      return null;
    }
    var box = el("details", "ds-details");
    box.appendChild(el("summary", "", "Test details"));
    var list = el("dl", "ds-details-list");
    keys.slice(0, 40).forEach(function (key) {
      list.appendChild(el("dt", "", key));
      list.appendChild(el("dd", "", metadata[key]));
    });
    box.appendChild(list);
    return box;
  }

  // A stored entry that cannot be shown (saved by an older page, or damaged) gets a card
  // that only offers to remove it, so it never stops the rest of the page from rendering.
  function brokenCard(dataset) {
    var article = el("article", "ds-card");
    var head = el("header", "ds-card-head");
    var name = dataset && isText(dataset.filename) ? dataset.filename : "An imported file";
    head.appendChild(el("h2", "", name));
    var remove = el("button", "quiet-button", "Remove");
    remove.type = "button";
    remove.addEventListener("click", function () {
      store.remove(dataset.id).then(render);
    });
    head.appendChild(remove);
    article.appendChild(head);
    article.appendChild(el("p", "ds-warnings",
      "This entry could not be read and is not offered for plotting. Remove it and import " +
      "the file again."));
    return article;
  }

  function render() {
    return store.all().then(function (datasets) {
      var list = $("ds-list");
      Array.prototype.slice.call(list.querySelectorAll(".ds-card")).forEach(function (node) {
        node.remove();
      });
      $("ds-empty").hidden = datasets.length > 0;
      datasets.forEach(function (dataset) {
        var node = null;
        if (validDataset(dataset)) {
          try {
            node = card(dataset);
          } catch (e) {
            node = null;
          }
        }
        list.appendChild(node || brokenCard(dataset));
      });
      var blocks = datasets.reduce(function (n, d) {
        return n + (d && Array.isArray(d.blocks) ? d.blocks.length : 0);
      }, 0);
      $("mydata-count").textContent = datasets.length
        ? datasets.length + (datasets.length === 1 ? " file, " : " files, ") +
          blocks + (blocks === 1 ? " table" : " tables")
        : "";
      $("project-save").disabled = !datasets.length;
      $("data-clear").disabled = !datasets.length;
    });
  }

  // --- Importing ------------------------------------------------------------------------

  function importFiles(files) {
    if (!files || !files.length) {
      return;
    }
    var form = new FormData();
    Array.prototype.forEach.call(files, function (file) {
      form.append("files", file);
    });
    status("Reading " + files.length + (files.length === 1 ? " file…" : " files…"), "busy");
    fetch("/api/rheometry/parse", { method: "POST", body: form })
      .then(function (r) {
        return readJson(r, "The files could not be read.");
      })
      .then(function (body) {
        var now = Date.now();
        body.datasets.forEach(function (dataset, i) {
          dataset.addedAt = now + i;
        });
        return store.putMany(body.datasets).then(function () {
          return body;
        });
      })
      .then(function (body) {
        var lines = [];
        if (body.datasets.length) {
          lines.push(
            "Added " + body.datasets.length + (body.datasets.length === 1 ? " file." : " files.")
          );
        } else {
          lines.push("No file could be read.");
        }
        body.errors.forEach(function (error) {
          lines.push(error.filename + ": " + error.message);
        });
        // Some read and some not: say what was added plainly, and only the failures in red.
        var kind = !body.errors.length ? "ok" : body.datasets.length ? "partial" : "error";
        status(lines, kind);
        return render();
      })
      .catch(function (error) {
        status(["The files were not imported.", String(error.message || error)], "error");
      })
      .then(function () {
        $("rheo-files").value = "";
      });
  }

  function setupDropzone() {
    var zone = $("dropzone");
    var input = $("rheo-files");
    input.addEventListener("change", function () {
      importFiles(input.files);
    });
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.remove("is-over");
      });
    });
    zone.addEventListener("drop", function (event) {
      importFiles(event.dataTransfer && event.dataTransfer.files);
    });
  }

  // --- Projects -------------------------------------------------------------------------

  function saveProject() {
    store.all().then(function (datasets) {
      var project = {
        kind: PROJECT_KIND,
        version: 1,
        saved: new Date().toISOString(),
        datasets: datasets
      };
      var blob = new Blob([JSON.stringify(project)], { type: "application/json" });
      var link = el("a");
      link.href = URL.createObjectURL(blob);
      link.download = "rheometer-project.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(function () {
        URL.revokeObjectURL(link.href);
      }, 1000);
    });
  }

  // --- Validation of what a project file or the store holds -----------------------------

  function isText(value) {
    return typeof value === "string";
  }

  // The check itself lives in store.js, so the graph pages skip exactly what this page does.
  function validDataset(dataset) {
    return store.validDataset(dataset, registry ? function (key) {
      return !!byQuantity[key];
    } : null);
  }

  function openProject(file) {
    file
      .text()
      .then(function (text) {
        var project = JSON.parse(text);
        if (!project || project.kind !== PROJECT_KIND || !Array.isArray(project.datasets)) {
          throw new Error("This file is not a project saved from this page.");
        }
        var valid = project.datasets.filter(validDataset);
        var skipped = project.datasets.length - valid.length;
        if (!valid.length && skipped) {
          throw new Error("None of the files in it could be read; it may be damaged.");
        }
        return store.putMany(valid).then(function () {
          return { count: valid.length, skipped: skipped };
        });
      })
      .then(function (result) {
        var lines = ["Opened the project: " + result.count +
          (result.count === 1 ? " file." : " files.")];
        if (result.skipped) {
          lines.push(result.skipped + (result.skipped === 1 ? " entry was" : " entries were") +
            " left out because it could not be read.");
        }
        status(lines.length > 1 ? lines : lines[0], result.skipped ? "partial" : "ok");
        return render();
      })
      .catch(function (error) {
        var message = error instanceof SyntaxError ? "The file is not valid JSON." : error.message;
        status(["The project was not opened.", message], "error");
      })
      .then(function () {
        $("project-open").value = "";
      });
  }

  function setupProjects() {
    $("project-save").addEventListener("click", saveProject);
    $("project-open").addEventListener("change", function (event) {
      if (event.target.files[0]) {
        openProject(event.target.files[0]);
      }
    });
    $("data-clear").addEventListener("click", function () {
      if (!window.confirm("Remove every imported file from this browser? This cannot be undone.")) {
        return;
      }
      store.clear().then(function () {
        status("All files were removed from this browser.");
        return render();
      });
    });
  }

  function start() {
    setupDropzone();
    setupProjects();
    Promise.all([loadRegistry(), store.ready])
      .then(function () {
        if (!store.persistent) {
          var note = $("mydata-note");
          note.textContent =
            "This browser is not letting the page save data, so imported files are kept only " +
            "until you leave this page. Use Save project to keep them.";
          note.classList.add("is-warning");
        }
        return render();
      })
      .catch(function () {
        status("The page could not reach the server. Check that it is running, then reload.",
          "error");
      });
  }

  // Pure helpers, for the Node tests in tests/test_rheometry_import_audit.py.
  window.ladaMyDataHooks = {
    matchUnit: matchUnit,
    validDataset: validDataset,
    detailText: detailText,
    setRegistry: function (data) {
      registry = data;
      byQuantity = {};
      data.quantities.forEach(function (q) {
        byQuantity[q.key] = q;
      });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
