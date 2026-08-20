---
layout: page
title: Writing
permalink: /writing/
---

[← home](/)

<div class="search-bar">
  <input type="search" id="writing-search" class="search-box" placeholder="search writing..." aria-label="Search writing">
  <div class="search-filters" id="search-filters">
    <button class="filter-btn active" data-filter="all" type="button">all</button>
    <button class="filter-btn" data-filter="logs" type="button">logs</button>
    <button class="filter-btn" data-filter="notes" type="button">notes</button>
    <button class="filter-btn" data-filter="annotations" type="button">annotations</button>
  </div>
</div>
<div id="writing-search-results" class="search-results" hidden></div>
<nav id="writing-pagination" class="pagination" hidden></nav>

<script>
window.searchData = {
  collection: "writing",
  documents: [
    {% for post in site.writing %}
    {
      title: {{ post.title | jsonify }},
      url: {{ post.url | relative_url | jsonify }},
      date: {{ post.date | date: "%b %d, %Y" | jsonify }},
      dateSort: {{ post.date | date_to_xmlschema | jsonify }},
      excerpt: {{ post.excerpt | strip_html | strip_newlines | jsonify }},
      tags: {{ post.tags | jsonify | default: "[]" }},
      content: {{ post.content | strip_html | strip_newlines | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
};
</script>
<script src="{{ '/assets/js/search.js' | relative_url }}"></script>
