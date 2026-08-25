---
layout: page
title: Experiments
permalink: /experiments/
hide_title: true
---

[← home](/)

<div class="search-bar">
  <input type="search" id="experiments-search" class="search-box" placeholder="search experiments..." aria-label="Search experiments">
</div>
<div id="experiments-search-results" class="search-results" hidden></div>
<nav id="experiments-pagination" class="pagination" hidden></nav>

<script>
window.searchData = {
  collection: "experiments",
  documents: [
    {% for post in site.experiments %}
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
