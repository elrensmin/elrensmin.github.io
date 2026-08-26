---
layout: page
title: Artifacts
permalink: /artifacts/
hide_title: true
---

[← home](/)

<div class="reading-section" markdown="1">

## artifacts

<div class="search-bar">
  <input type="search" id="artifacts-search" class="search-box" placeholder="search artifacts..." aria-label="Search artifacts">
</div>
<div id="artifacts-search-results" class="search-results" hidden></div>
<nav id="artifacts-pagination" class="pagination" hidden></nav>

<script>
window.searchData = {
  collection: "artifacts",
  documents: [
    {% for post in site.artifacts | sort: "date" | reverse %}
    {
      title: {{ post.title | jsonify }},
      url: {{ post.github | jsonify }},
      date: {{ post.date | date: "%b %d, %Y" | jsonify }},
      dateSort: {{ post.date | date_to_xmlschema | jsonify }},
      excerpt: {{ post.description | jsonify }},
      tags: {{ post.tags | jsonify | default: "[]" }},
      content: {{ post.content | strip_html | strip_newlines | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
};
</script>
<script src="{{ '/assets/js/search.js' | relative_url }}"></script>
</div>
