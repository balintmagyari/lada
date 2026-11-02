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

  // Where each kind of test is drawn, and the section colour it wears.
  var PAGES = {
    frequency_sweep: ["/rheometer/frequency-sweep", "oscillation"],
    amplitude_sweep: ["/rheometer/amplitude-sweep", "oscillation"],
    temperature_sweep: ["/rheometer/temperature-sweep", "oscillation"],
    time_sweep: ["/rheometer/custom-plot", "oscillation"],
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

  function readJson(response, fallback) {
    return response.json().then(
      function (body) {
        if (!response.ok) {
          throw new Error(body.detail || fallback);
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

  function option(value, label, selected) {
    var node = el("option", "", label);
    node.value = value;
    node.selected = !!selected;
    return node;
  }

  function unitOptions(select, quantityKey, current) {
    select.textContent = "";
    var spec = byQuantity[quantityKey];
    var units = spec ? spec.units : [];
    var matched = false;
    units.forEach(function (u) {
      var same = current !== null && current !== undefined &&
        u.symbol.toLowerCase() === String(current).toLowerCase();
      matched = matched || same;
      select.appendChild(option(u.symbol, u.symbol || "no unit", same));
    });
    if (!matched && units.length) {
      select.options[0].selected = true;
    }
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
      unitOptions(unit, quantity.value, null);
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
    head.appendChild(open);
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
    dataset.blocks.forEach(function (block) {
      article.appendChild(blockSection(dataset, block));
    });
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
        list.appendChild(card(dataset));
      });
      var blocks = datasets.reduce(function (n, d) {
        return n + d.blocks.length;
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

  function openProject(file) {
    file
      .text()
      .then(function (text) {
        var project = JSON.parse(text);
        if (!project || project.kind !== PROJECT_KIND || !Array.isArray(project.datasets)) {
          throw new Error("This file is not a project saved from this page.");
        }
        var valid = project.datasets.filter(function (d) {
          return d && typeof d.id === "string" && Array.isArray(d.blocks);
        });
        return store.putMany(valid).then(function () {
          return valid.length;
        });
      })
      .then(function (count) {
        status("Opened the project: " + count + (count === 1 ? " file." : " files."), "ok");
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
