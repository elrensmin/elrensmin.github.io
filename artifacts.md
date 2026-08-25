---
layout: page
title: Artifacts
permalink: /artifacts/
hide_title: true
---

[← home](/)

<div class="reading-section" markdown="1">

## artifacts

<ul class="home-list">
  {% assign items = site.artifacts | sort: "date" | reverse %}
  {% for post in items %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
</div>
