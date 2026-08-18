---
layout: page
title: Papers
permalink: /papers/
---

[← home](/)

<div class="search-bar">
  <input type="search" id="papers-search" class="search-box" placeholder="search papers..." aria-label="Search papers">
</div>
<div id="papers-search-results" class="search-results" hidden></div>
<nav id="papers-pagination" class="pagination" hidden></nav>

<script>
window.searchData = {
  collection: "papers",
  documents: [
    {% for post in site.papers %}
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
