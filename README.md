# OpenHarmony-ANNS-Test

PipeANN C++ acceptance tests for filtered vector search, space usage, dynamic
updates, and single-query resource measurement.

This repository is not a standalone ANN framework. Copy `openharmony_acceptance/`
into a PipeANN source tree and build it with PipeANN so the tests can call
PipeANN C++ APIs directly.

## What This Tests

- Core index space expansion is below `2.0x` raw vector bytes.
- Static filtered search covers `match_all`, equality, intersect, and range
  selectors at `0.01%, 0.1%, 1%, 5%, 10%, 25%, 50%, 100%`.
- Recall is computed with PipeANN's official `compute_groundtruth` output.
- Dynamic update uses PipeANN's native `DynamicIndex` in one long-lived process:
  mark-delete, merge/save, then insert new vectors into the same tag range.
- The 5-cycle delete ranges alternate between `[400k, 1M)` and `[0, 600k)`.
- Single-query latency is emitted by the C++ runner; max RSS is measured by
  `/usr/bin/time -v`.

## Integrate Into PipeANN

From a PipeANN checkout:

```bash
cp -r /path/to/OpenHarmony-ANNS-Test/openharmony_acceptance .
printf '\nadd_subdirectory(openharmony_acceptance)\n' >> CMakeLists.txt
cmake -S . -B build -DIO_ENGINE=uring -DUSE_TCMALLOC=OFF
cmake --build build -j"$(nproc)" --target \
  oh_generate_labels oh_make_synthetic oh_materialize_cycle_vectors \
  oh_build_space oh_static_filtered oh_dynamic_chain oh_single_query \
  oh_summarize_results \
  compute_groundtruth build_disk_index_filtered
```

## Smoke

```bash
PIPEANN_ROOT=/path/to/PipeANN \
  /path/to/PipeANN/openharmony_acceptance/scripts/run_smoke.sh
```

The smoke run creates a small synthetic dataset, builds a filtered disk index,
generates official filtered groundtruth, runs static search, runs two dynamic
cycles, records single-query resource output, and writes a real pass/fail
summary. Smoke uses relaxed default thresholds because tiny synthetic indexes
are dominated by fixed metadata overhead.

## Full Run

```bash
PIPEANN_ROOT=/path/to/PipeANN \
BASE_BIN=/data/sift1m/base.bin \
UPDATES_BIN=/data/sift1m/updates_3m.bin \
QUERY_BIN=/data/sift1m/query.bin \
TYPE=float \
NPOINTS=1000000 \
NQUERIES=1000 \
R=96 \
R_DENSE=1000 \
BUILD_L=128 \
PQ_BYTES=32 \
MEM_GB=64 \
SEARCH_L=100 \
  /path/to/PipeANN/openharmony_acceptance/scripts/run_full_acceptance.sh
```

`UPDATES_BIN` must contain at least `cycles * 600k` rows. For the default
5-cycle run this is at least 3 million update vectors.

## Outputs

The scripts write compact artifacts only:

- `space_audit.json/csv`
- `static_filtered.jsonl`
- `dynamic_chain.jsonl`
- `foreground_latency.jsonl`
- `dynamic_checkpoint_search.jsonl`
- `single_query_resource.jsonl`
- `single_query_time.txt`
- `acceptance_summary.json`

Large datasets, index files, generated GT files, and full experiment results are
not committed.

## Dynamic Filter Path

The dynamic runner does not call `DynamicIndex::load_filter_from_json()`.
Instead it loads the official label/range attr indexes once with
`load_attr_index_from_file()` and constructs PipeANN native selectors directly:

- equality: `LabelOrSelector`
- intersect: `LabelAndSelector`
- range: `RangeSelector`

This keeps the live attr index map stable while `remove()`, `save()`, and
`insert()` update the same label/range indexes.
