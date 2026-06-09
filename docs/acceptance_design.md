# PipeANN C++ Acceptance Design

The acceptance suite is integrated into PipeANN instead of wrapping PipeANN with
a Python black-box adapter. This keeps update tests close to the official
examples: one process holds `DynamicIndex`, performs updates, and searches the
live index while updates run.

## Data And Labels

`oh_generate_labels` creates PipeANN-native attribute files:

- `base_labels.spmat` for equality and intersect selectors.
- `base_range.bin` for range selectors.
- one JSON selector config per selector/selectivity pair.
- `selector_manifest.csv` listing every selector case.

The fixed selectivities are `0.01%, 0.1%, 1%, 5%, 10%, 25%, 50%, 100%`.
Intersect selectors use PipeANN's `label_and` selector. Range selectors use
PipeANN's `[lower, upper)` range query format.

## Groundtruth

The only groundtruth path is PipeANN's official C++ tool:

```bash
build/tests/utils/compute_groundtruth \
  <type> <metric> <base.bin> <query.bin> <K> <gt.bin> <tag_file|null> <label_config.json|null>
```

Static search uses the initial base vectors. Dynamic checkpoint search first
materializes the current logical 1M vector version with
`oh_materialize_cycle_vectors`, then calls `compute_groundtruth` for every
selector.

## Dynamic Update Semantics

The runner alternates two 60% continuous delete ranges:

```text
cycle1: [400k, 1M)
cycle2: [0, 600k)
cycle3: [400k, 1M)
cycle4: [0, 600k)
cycle5: [400k, 1M)
```

Each cycle is:

```text
mark delete -> save/merge_deletes -> insert new vectors with the same tags
```

Same-tag insertion only happens after merge clears tombstones. Delete timing
therefore measures only mark-delete, while merge/save is reported separately.

`oh_dynamic_chain` keeps one `DynamicIndex` alive for the whole run. It loads
the official label and range attr indexes once with `load_attr_index_from_file`
and constructs native selectors directly. It intentionally does not call
`load_filter_from_json` during dynamic testing, because that convenience loader
replaces the in-memory attr index map used by insert and merge.

## Metrics

- Space: core index bytes divided by raw vector bytes.
- Static and checkpoint search: recall, avg/p50/p95/p99 latency, IO stats.
- Dynamic: delete time, merge time, insert time, foreground latency.
- Single query: runner latency plus `/usr/bin/time -v` max RSS.
- Summary: `oh_summarize_results` writes `acceptance_summary.json` with the
  final pass/fail decision and compact failure details.
