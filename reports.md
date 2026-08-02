---
layout: page
title: Reports
permalink: /reports/
---

{% assign report_pages = site.reports | sort: 'date' | reverse %}
<ul class="log-list">
  {% for report in report_pages %}
  <li>
    <a class="log-link" href="{{ report.url | relative_url }}">{{ report.title }}</a>
    <div class="log-date">{{ report.date | date: "%b %d, %Y" }}</div>
    {% if report.excerpt %}
    <div class="log-excerpt">{{ report.excerpt | strip_html | truncate: 200 }}</div>
    {% endif %}
  </li>
  {% endfor %}
</ul>
