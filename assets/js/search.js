(function() {
  var data = window.searchData;
  if (!data) return;

  var input = document.getElementById(data.collection + "-search");
  if (!input) return;

  var resultsEl = document.getElementById(data.collection + "-search-results");
  var pagerEl = document.getElementById(data.collection + "-pagination");
  var docs = data.documents || [];

  var perPage = 5;
  var page = 1;
  var filter = "all";

  var filterBtns = document.querySelectorAll("#search-filters .filter-btn");
  if (filterBtns.length) {
    filterBtns.forEach(function(btn) {
      btn.addEventListener("click", function() {
        filterBtns.forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        filter = btn.getAttribute("data-filter");
        page = 1;
        render();
      });
    });
  }

  function filterMatches(d) {
    if (filter === "all") return true;
    var tags = d.tags || [];
    return tags.indexOf(filter) !== -1;
  }

  function getFiltered() {
    var term = input.value.trim().toLowerCase();
    var out = [];
    for (var i = 0; i < docs.length; i++) {
      var d = docs[i];
      if (!filterMatches(d)) continue;
      if (term) {
        var hay = [d.title, (d.tags || []).join(" "), d.date, d.content]
          .join(" ").toLowerCase();
        if (hay.indexOf(term) === -1) continue;
      }
      out.push(d);
    }
    out.sort(function(a, b) {
      return (b.dateSort || b.date || "").localeCompare(a.dateSort || a.date || "");
    });
    return out;
  }

  function render() {
    var all = getFiltered();
    var totalPages = Math.max(1, Math.ceil(all.length / perPage));
    if (page > totalPages) page = totalPages;

    if (!all.length) {
      resultsEl.innerHTML = '<p class="search-none">no matches</p>';
      resultsEl.hidden = false;
      pagerEl.hidden = true;
      pagerEl.innerHTML = "";
      return;
    }

    var start = (page - 1) * perPage;
    var slice = all.slice(start, start + perPage);

    var html = '<ul class="log-list">';
    for (var i = 0; i < slice.length; i++) {
      var m = slice[i];
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

    renderPager(totalPages);
  }

  function renderPager(totalPages) {
    if (totalPages <= 1) {
      pagerEl.hidden = true;
      pagerEl.innerHTML = "";
      return;
    }
    var prev = page > 1 ? '<a class="page-link" href="#" data-page="' + (page - 1) + '">&larr; newer</a>'
                        : '<span class="page-link disabled">&larr; newer</span>';
    var next = page < totalPages ? '<a class="page-link" href="#" data-page="' + (page + 1) + '">older &rarr;</a>'
                        : '<span class="page-link disabled">older &rarr;</span>';
    pagerEl.innerHTML = prev + '<span class="page-info">page ' + page + ' of ' + totalPages + '</span>' + next;
    pagerEl.hidden = false;
  }

  pagerEl.addEventListener("click", function(e) {
    var link = e.target.closest("a.page-link");
    if (!link) return;
    e.preventDefault();
    page = parseInt(link.getAttribute("data-page"), 10);
    render();
    window.scrollTo(0, 0);
  });

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  input.addEventListener("input", function() {
    page = 1;
    render();
  });

  render();
})();
