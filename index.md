---
layout: home
title: welcome
---

<div class="home-grid">

<section class="home-section">
  <h3>Notes</h3>
  <ul class="home-list">
    {% assign recent = site.writing | sort: "date" | reverse | slice: 0, 5 %}
    {% for post in recent %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
    {% if site.writing.size == 0 %}
    <li>coming soon</li>
    {% elsif site.writing.size > 5 %}
    <li><a href="{{ '/writing/' | relative_url }}">...more</a></li>
    {% endif %}
  </ul>
</section>

<section class="home-section">
  <h3>Experiments</h3>
  <ul class="home-list">
    {% assign recent = site.experiments | sort: "date" | reverse | slice: 0, 5 %}
    {% for post in recent %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
    {% if site.experiments.size == 0 %}
    <li>coming soon</li>
    {% elsif site.experiments.size > 5 %}
    <li><a href="{{ '/experiments/' | relative_url }}">...more</a></li>
    {% endif %}
  </ul>
</section>

<section class="home-section">
  <h3>Reports</h3>
  <ul class="home-list">
    {% assign recent = site.reports | sort: "date" | reverse | slice: 0, 5 %}
    {% for post in recent %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
    {% if site.reports.size == 0 %}
    <li>coming soon</li>
    {% elsif site.reports.size > 5 %}
    <li><a href="{{ '/reports/' | relative_url }}">...more</a></li>
    {% endif %}
  </ul>
</section>

<section class="home-section">
  <h3>Artifacts</h3>
  <ul class="home-list">
    {% assign recent = site.artifacts | sort: "date" | reverse | slice: 0, 5 %}
    {% for post in recent %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
    {% if site.artifacts.size == 0 %}
    <li>coming soon</li>
    {% elsif site.artifacts.size > 5 %}
    <li><a href="{{ '/artifacts/' | relative_url }}">...more</a></li>
    {% endif %}
  </ul>
</section>

<section class="home-section">
  <h3>Others</h3>
  <ul class="home-list">
    <li><a href="{{ '/reading/' | relative_url }}">Books</a></li>
    <li><a href="{{ '/blogs/' | relative_url }}">Blogs</a></li>
  </ul>
</section>

</div>
