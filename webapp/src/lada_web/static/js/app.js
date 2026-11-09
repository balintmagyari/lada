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
})();
