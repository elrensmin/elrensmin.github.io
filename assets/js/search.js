(function() {
  var data = window.searchData;
  if (!data) return;

  var input = document.getElementById(data.collection + "-search");
  if (!input) return;

  var resultsEl = document.getElementById(data.collection + "-search-results");
  var listEl = document.getElementById(data.collection + "-list");
  var docs = data.documents || [];
  var filter = "all";

  var filterBtns = document.querySelectorAll("#search-filters .filter-btn");
  if (filterBtns.length) {
    filterBtns.forEach(function(btn) {
      btn.addEventListener("click", function() {
        filterBtns.forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        filter = btn.getAttribute("data-filter");
        update();
      });
    });
  }

  function filterMatches(d) {
    if (filter === "all") return true;
    var tags = d.tags || [];
    return tags.indexOf(filter) !== -1;
  }

  function search(term) {
    term = term.trim().toLowerCase();
    var out = [];
    for (var i = 0; i < docs.length; i++) {
      var d = docs[i];
      if (!filterMatches(d)) continue;
      if (!term) { out.push(d); continue; }
      var hay = [d.title, (d.tags || []).join(" "), d.date, d.content]
        .join(" ").toLowerCase();
      if (hay.indexOf(term) !== -1) out.push(d);
    }
    out.sort(function(a, b) {
      return (b.dateSort || b.date || "").localeCompare(a.dateSort || a.date || "");
    });
    return out;
  }

  function render(matches) {
    if (!matches.length) {
      resultsEl.innerHTML = '<p class="search-none">no matches</p>';
      resultsEl.hidden = false;
      listEl.hidden = true;
      return;
    }
    var html = '<ul class="log-list">';
    for (var i = 0; i < matches.length; i++) {
      var m = matches[i];
      html += '<li><a class="log-link" href="' + m.url + '">' + escapeHtml(m.title) + '</a>' +
        '<div class="log-date">' + escapeHtml(m.date) + '</div>';
      if (m.excerpt) {
        html += '<div class="log-excerpt">' + escapeHtml(m.excerpt) + '</div>';
      }
      html += '</li>';
    }
    html += '</ul>';
    resultsEl.innerHTML = html;
    resultsEl.hidden = false;
    listEl.hidden = true;
  }

  function update() {
    var matches = search(input.value);
    render(matches);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  input.addEventListener("input", function() {
    if (input.value.trim() || filter !== "all") {
      update();
    } else {
      resultsEl.hidden = true;
      resultsEl.innerHTML = "";
      listEl.hidden = false;
    }
  });
})();
