---
layout: page
title: Logs
permalink: /logs/
---

{% assign log_posts = site.logs | sort: 'date' | reverse %}
<ul class="log-list">
  {% for log in log_posts %}
  <li>
    <a class="log-link" href="{{ log.url | relative_url }}">{{ log.title }}</a>
    <div class="log-date">{{ log.date | date: "%b %d, %Y" }}</div>
    {% if log.excerpt %}
    <div class="log-excerpt">{{ log.excerpt | strip_html | truncate: 200 }}</div>
    {% endif %}
  </li>
  {% endfor %}
</ul>
