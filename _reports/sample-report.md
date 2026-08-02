---
title: Sample Report
date: 2026-08-01
description: A template for the reports section — longer-form writing, findings, and deep dives.
tags: [template, meta]
---

This is a sample report. Use this file as a template for future reports.

Reports are for longer-form writing: paper reviews, project retrospectives, research findings, or anything that doesn't fit the worklog format of the logs section.

## Structure

Each report is a markdown file in `_reports/` with YAML front matter:

```yaml
---
title: Your Title Here
date: YYYY-MM-DD
description: A short summary shown on the listing page.
tags: [tag1, tag2]
---
```

## Front matter fields

| Field       | Required | Description                              |
|-------------|----------|------------------------------------------|
| `title`     | yes      | Page title and link text on listing      |
| `date`      | yes      | Display date and sort order (descending) |
| `description` | no    | Excerpt shown below the link on listing  |
| `tags`      | no       | List of tags for categorization          |

## Linking

Reports are accessible at `/reports/:title/` and listed at `/reports/`.
