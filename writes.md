---
layout: page
title: Writes
permalink: /writing/
hide_title: true
---

<div class="writes-bar">
  <input type="search" id="writes-search" class="search-box" placeholder="search titles..." aria-label="Search writing">
</div>

<div class="tabs" role="tablist">
  <button class="tab active" type="button" data-tab="blog" role="tab" aria-selected="true">blog</button>
  <button class="tab" type="button" data-tab="study-notes" role="tab" aria-selected="false">study-notes</button>
</div>

<div id="writes-list">
  {% assign notes = site.writing | where_exp: "p", "p.tags contains 'study-notes'" | sort: "date" | reverse %}
  {% assign posts = site.writing | where_exp: "p", "p.tags contains 'blog'" | sort: "date" | reverse %}

  <div id="tab-study-notes" class="tab-panel" hidden>
    <ul class="w-list">
      {% for post in notes %}
      <li class="w-item" data-s="{{ post.title | escape }} {{ post.tags | join: ' ' | escape }}">
        <time class="w-date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %d, %Y" }}</time>
        <a class="w-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </li>
      {% endfor %}
    </ul>
  </div>

  <div id="tab-blog" class="tab-panel">
    <ul class="w-list">
      {% for post in posts %}
      <li class="w-item" data-s="{{ post.title | escape }} {{ post.tags | join: ' ' | escape }}">
        <time class="w-date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %d, %Y" }}</time>
        <a class="w-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </li>
      {% endfor %}
    </ul>
  </div>
</div>

<script>
(function () {
  var tabs = document.querySelectorAll('.tabs .tab');
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) {
        t.classList.toggle('active', t === tab);
        t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
        var panel = document.getElementById('tab-' + t.getAttribute('data-tab'));
        if (panel) panel.hidden = (t !== tab);
      });
    });
  });

  var q = document.getElementById('writes-search');
  if (!q) return;
  q.addEventListener('input', function () {
    var term = q.value.trim().toLowerCase();
    document.querySelectorAll('#writes-list .w-item').forEach(function (li) {
      var hit = !term || (li.getAttribute('data-s') || '').toLowerCase().indexOf(term) !== -1;
      li.hidden = !hit;
    });
  });
})();
</script>
