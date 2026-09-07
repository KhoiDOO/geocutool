/* Conquer3D docs -- theme, search, and scrollspy.
 *
 * No framework and no build step: the site is static HTML served straight from
 * docs/, so everything here degrades to "still readable" if JS never runs.
 */
(function () {
  "use strict";

  var root = document.documentElement;
  var BASE = root.getAttribute("data-base") || "";

  /* ------------------------------------------------------------ theme --- */
  // Applied inline in <head> before paint to avoid a flash; this only wires
  // up the toggle and keeps the OS preference in sync when unset.
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("c3d-theme", next); } catch (e) {}
    });
  }

  /* ----------------------------------------------------------- search --- */
  var modal = document.getElementById("search-modal");
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var trigger = document.getElementById("search-trigger");
  var index = null, loading = false, selected = 0, current = [];

  function loadIndex() {
    if (index || loading) return;
    loading = true;
    fetch(BASE + "search-index.json")
      .then(function (r) { return r.json(); })
      .then(function (data) { index = data; loading = false; if (input && input.value) run(input.value); })
      .catch(function () { loading = false; });
  }

  function openSearch() {
    if (!modal) return;
    loadIndex();
    modal.classList.add("open");
    input.value = "";
    results.innerHTML = "";
    input.focus();
  }
  function closeSearch() { if (modal) modal.classList.remove("open"); }

  if (trigger) trigger.addEventListener("click", openSearch);
  if (modal) {
    modal.addEventListener("click", function (e) { if (e.target === modal) closeSearch(); });
  }

  document.addEventListener("keydown", function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) {
      e.preventDefault(); openSearch(); return;
    }
    if (!modal || !modal.classList.contains("open")) return;
    if (e.key === "Escape") { closeSearch(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      var el = results.querySelector(".sr-item.sel");
      if (el) { window.location.href = el.getAttribute("data-href"); }
    }
  });

  function move(delta) {
    var items = results.querySelectorAll(".sr-item");
    if (!items.length) return;
    selected = (selected + delta + items.length) % items.length;
    items.forEach(function (el, i) { el.classList.toggle("sel", i === selected); });
    items[selected].scrollIntoView({ block: "nearest" });
  }

  // Subsequence scoring: rewards a contiguous prefix hit, then substring, then
  // scattered-character matches, so "dmc" still finds dual_marching_cubes.
  function score(needle, hay) {
    var h = hay.toLowerCase();
    var i = h.indexOf(needle);
    if (i === 0) return 1000 - hay.length;
    if (i > 0) return 700 - i - hay.length * 0.1;
    var ni = 0;
    for (var k = 0; k < h.length && ni < needle.length; k++) {
      if (h[k] === needle[ni]) ni++;
    }
    return ni === needle.length ? 300 - hay.length * 0.1 : -1;
  }

  function run(query) {
    var q = query.trim().toLowerCase();
    if (!q) { results.innerHTML = ""; current = []; return; }
    if (!index) { results.innerHTML = '<div class="sr-empty">Loading index…</div>'; return; }

    var scored = [];
    for (var i = 0; i < index.length; i++) {
      var e = index[i];
      var s = score(q, e.n);
      if (s < 0) s = score(q, e.q) - 120;
      if (s < 0 && e.s && e.s.toLowerCase().indexOf(q) >= 0) s = 90;
      if (s >= 0) scored.push([s, e]);
    }
    scored.sort(function (a, b) { return b[0] - a[0]; });
    current = scored.slice(0, 40).map(function (x) { return x[1]; });
    selected = 0;

    if (!current.length) {
      results.innerHTML = '<div class="sr-empty">No matches for “' + esc(query) + '”</div>';
      return;
    }
    results.innerHTML = current.map(function (e, i) {
      return '<a class="sr-item' + (i === 0 ? " sel" : "") + '" data-href="' + BASE + e.u + '" href="' + BASE + e.u + '">'
        + '<div class="sr-name">' + esc(e.q || e.n) + "</div>"
        + '<div class="sr-meta"><span class="badge badge-' + e.t.toLowerCase() + '">' + e.t + "</span>"
        + "<span>" + esc(e.k) + "</span>"
        + (e.s ? "<span>· " + esc(e.s.slice(0, 74)) + "</span>" : "")
        + "</div></a>";
    }).join("");
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  if (input) {
    input.addEventListener("input", function () { run(input.value); });
  }

  /* -------------------------------------------------------- scrollspy --- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (links.length && "IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var seen = {};
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { seen[en.target.id] = en.isIntersecting; });
      var active = null;
      for (var id in byId) { if (seen[id]) { active = id; break; } }
      links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + active); });
    }, { rootMargin: "-72px 0px -72% 0px" });
    Object.keys(byId).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) obs.observe(el);
    });
  }

  /* ------------------------------------------------------------ lightbox */
  // Figures carry fine detail (crease crops, curvature speckle) that only reads
  // at full size, so every figure is click-to-zoom.
  var lb = document.getElementById("lightbox");
  if (lb) {
    var lbImg = lb.querySelector("img");
    var lbCap = lb.querySelector(".lightbox-cap");
    document.querySelectorAll(".fig-media").forEach(function (el) {
      el.addEventListener("click", function () {
        var img = el.querySelector("img");
        if (!img) return;
        lbImg.src = img.src;
        lbImg.alt = img.alt;
        lbCap.textContent = img.alt;
        lb.classList.add("open");
      });
    });
    function closeLb() { lb.classList.remove("open"); lbImg.src = ""; }
    lb.addEventListener("click", function (e) {
      if (e.target === lb || e.target.classList.contains("lightbox-close")) closeLb();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && lb.classList.contains("open")) closeLb();
    });
  }

  /* ------------------------------------------------- copy code buttons --- */
  document.querySelectorAll("pre").forEach(function (pre) {
    pre.addEventListener("dblclick", function () {
      var text = pre.innerText;
      if (navigator.clipboard) navigator.clipboard.writeText(text);
    });
  });
})();
