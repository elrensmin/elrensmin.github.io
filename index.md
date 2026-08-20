---
layout: home
title: welcome
---

<img class="home-image" src="/images/sunrise.png" alt="sunrise">

<section class="home-section">
  <h3>Notes</h3>
  <ul class="home-list">
    {% assign recent = site.writing | sort: "date" | reverse | slice: 0, 3 %}
    {% for post in recent %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
    <li><a href="{{ '/writing/' | relative_url }}">...more</a></li>
  </ul>
</section>

<section class="home-section">
  <h3>Others</h3>
  <ul class="home-list">
    <li><a href="{{ '/reading/' | relative_url }}">Books</a></li>
  </ul>
</section>
