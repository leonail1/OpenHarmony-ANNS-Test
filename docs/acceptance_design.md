# Acceptance Design

## Purpose

The harness validates whether an ANN search system can satisfy filtered-search acceptance metrics without constraining its internal architecture.

## Public Contract

Adapters expose five command-template interfaces:

- `ann_build_index`
- `ann_filter_search`
- `ann_apply_insert`
- `ann_apply_delete`
- `ann_label_selectivity`

The harness records wall time, max RSS, CPU, IO, recall, latency, space ratio, and failures externally. It does not trust the implementation to self-report acceptance metrics except for raw search results and label selectivity counts.

## Labels

The harness creates a merged CSV label file for build and insert. The first version supports uniform labels only. Equality, range, and intersect workloads cover fixed target selectivities:

`0.01% / 0.1% / 1% / 5% / 10% / 25% / 50% / 100%`

Selectors with fewer than `k` live candidates are marked invalid and are not treated as algorithm failures.

## Tests

- Build Space Test: `index_bytes / raw_bytes < 2.0`.
- Label Selectivity Test: equality/range counts must match the harness live label table.
- Static Filtered Search Test: all datasets, selector types, and target selectivities; pilot skip at 20 ms; full run at up to 1000 queries.
- Dynamic Update Chain Test: from 0 vectors to target size, then 5 cycles of 60% delete and insert back to target size with foreground searches during mutations.
- Single Query Resource Test: one-query filtered search with no groundtruth, recording latency and max RSS.

