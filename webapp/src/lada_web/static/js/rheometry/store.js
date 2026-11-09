/* The Rheometer workspace's datasets, kept in this browser only.

   IndexedDB, so every page of the workspace sees the same files and a reload loses nothing.
   Where IndexedDB is unavailable (some private windows, blocked site data) the store keeps
   the datasets in memory for this page instead and says so through `persistent`.

   window.ladaRheoStore: all(), get(id), put(dataset), putMany(list), remove(id), clear(),
   updateBlock(datasetId, block), validDataset(dataset, knownQuantity), persistent (boolean,
   settled once ready resolves), ready. */

(function () {
  "use strict";

  var DB_NAME = "lada-rheometry";
  var STORE = "datasets";
  var memory = new Map();
  var db = null;

  function openDb() {
    return new Promise(function (resolve) {
      var request;
      try {
        request = window.indexedDB.open(DB_NAME, 1);
      } catch (e) {
        resolve(null);
        return;
      }
      request.onupgradeneeded = function () {
        if (!request.result.objectStoreNames.contains(STORE)) {
          request.result.createObjectStore(STORE, { keyPath: "id" });
        }
      };
      request.onsuccess = function () {
        resolve(request.result);
      };
      request.onerror = function () {
        resolve(null);
      };
      request.onblocked = function () {
        resolve(null);
      };
    });
  }

  function run(mode, work) {
    return new Promise(function (resolve, reject) {
      var tx = db.transaction(STORE, mode);
      var request = work(tx.objectStore(STORE));
      tx.oncomplete = function () {
        resolve(request && "result" in request ? request.result : undefined);
      };
      tx.onerror = function () {
        reject(tx.error);
      };
      tx.onabort = function () {
        reject(tx.error || new Error("The browser refused to save the data."));
      };
    });
  }

  function byAdded(list) {
    return list.sort(function (a, b) {
      return (a.addedAt || 0) - (b.addedAt || 0);
    });
  }

  // --- What a stored dataset must look like ----------------------------------------------
  // Shared by My data and every graph page: one malformed entry in IndexedDB (from an old
  // version, or a damaged project file) must be skipped, never break the page that reads it.
  // `knownQuantity(key)`, when given, says whether a quantity key is in the server's registry.

  function isText(value) {
    return typeof value === "string";
  }

  function validColumn(column, points, knownQuantity) {
    return !!column && typeof column === "object" && isText(column.key) &&
      isText(column.label) && Array.isArray(column.values) &&
      column.values.length === points &&
      column.values.every(function (v) {
        return v === null || (typeof v === "number" && isFinite(v));
      }) &&
      (column.quantity === null || column.quantity === undefined ||
        (isText(column.quantity) && (!knownQuantity || !!knownQuantity(column.quantity)))) &&
      (column.unit === undefined || isText(column.unit)) &&
      (column.source_unit === undefined || isText(column.source_unit));
  }

  function validBlock(block, knownQuantity) {
    return !!block && typeof block === "object" && isText(block.id) && isText(block.name) &&
      typeof block.n_points === "number" && block.n_points >= 0 &&
      Array.isArray(block.columns) &&
      block.columns.every(function (c) {
        return validColumn(c, block.n_points, knownQuantity);
      }) &&
      (block.temperature === null || block.temperature === undefined ||
        typeof block.temperature === "number") &&
      (block.metadata === undefined || (!!block.metadata && typeof block.metadata === "object"));
  }

  function validDataset(dataset, knownQuantity) {
    return !!dataset && typeof dataset === "object" && isText(dataset.id) && dataset.id !== "" &&
      isText(dataset.filename) && isText(dataset.format_label) &&
      Array.isArray(dataset.blocks) &&
      dataset.blocks.every(function (b) {
        return validBlock(b, knownQuantity);
      }) &&
      (dataset.warnings === undefined ||
        (Array.isArray(dataset.warnings) && dataset.warnings.every(isText)));
  }

  var api = {
    persistent: false,
    validDataset: validDataset,
    ready: openDb().then(function (opened) {
      db = opened;
      api.persistent = !!opened;
    }),

    all: function () {
      return api.ready.then(function () {
        if (!db) {
          return byAdded(Array.from(memory.values()));
        }
        return run("readonly", function (store) {
          return store.getAll();
        }).then(byAdded);
      });
    },

    get: function (id) {
      return api.ready.then(function () {
        if (!db) {
          return memory.get(id);
        }
        return run("readonly", function (store) {
          return store.get(id);
        });
      });
    },

    putMany: function (list) {
      return api.ready.then(function () {
        if (!db) {
          list.forEach(function (dataset) {
            memory.set(dataset.id, dataset);
          });
          return undefined;
        }
        return run("readwrite", function (store) {
          list.forEach(function (dataset) {
            store.put(dataset);
          });
          return null;
        });
      });
    },

    put: function (dataset) {
      return api.putMany([dataset]);
    },

    remove: function (id) {
      return api.ready.then(function () {
        if (!db) {
          memory.delete(id);
          return undefined;
        }
        return run("readwrite", function (store) {
          return store.delete(id);
        });
      });
    },

    clear: function () {
      return api.ready.then(function () {
        if (!db) {
          memory.clear();
          return undefined;
        }
        return run("readwrite", function (store) {
          return store.clear();
        });
      });
    },

    updateBlock: function (datasetId, block) {
      return api.get(datasetId).then(function (dataset) {
        if (!dataset) {
          return null;
        }
        dataset.blocks = dataset.blocks.map(function (b) {
          return b.id === block.id ? block : b;
        });
        return api.put(dataset).then(function () {
          return dataset;
        });
      });
    }
  };

  window.ladaRheoStore = api;
})();
