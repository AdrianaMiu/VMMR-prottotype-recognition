---
status: accepted
date: 2026-11-06
---

# Freeze the project context as protocol v1.0

`PROTOCOL.md` is approved as protocol v1.0. Any change to a dataset protocol, split, label definition, metric definition, encoder checkpoint, preprocessing variant, hypothesis, or statistical procedure requires a new ADR and a protocol version increment before the behavior changes. Engineer-owned defaults listed in §26 of the context (manifest format, config framework, exact-search library, tie-breaking, UMAP parameters, module boundaries) may be fixed in ADRs without new research approval.

We chose a hard freeze over an evolving document because the article's comparability claims depend on splits and metrics that were fixed before test results were seen.
