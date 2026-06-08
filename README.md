# OpenHarmony ANNS Acceptance Tests

Generic acceptance tests for filtered ANN systems. The harness tests only external behavior: build, filtered search, insert, delete, and label selectivity. It does not assume PipeANN, PQ, prefilter routing, or any specific index layout.

## Interfaces

Adapters are configured with command templates in an adapter manifest. Every command must support a `threads` variable.

- `ann_build_index`: build an index from vectors, ids, and externally supplied labels. It writes `build_manifest.json` with `raw_data_paths` and `index_output_paths`.
- `ann_filter_search`: the only search interface. Selectors are `match_all`, `equality`, `intersect`, and `range`.
- `ann_apply_insert`: insert externally supplied vectors, ids, and labels. It returns the current live vector count.
- `ann_apply_delete`: delete externally supplied ids. It returns the current live vector count.
- `ann_label_selectivity`: return `matched_count`, `total_live_count`, and `selectivity` for equality/range selectors.

## Label Workload

The test side generates uniform labels for:

`0.01% / 0.1% / 1% / 5% / 10% / 25% / 50% / 100%`

Build and insert commands receive the generated label CSV. Delete commands receive ids only; the system must invalidate labels internally with the deleted vectors.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./tools/build_groundtruth.sh
```

`tools/build_groundtruth.sh` builds the harness-owned C++ exact L2 top-k
groundtruth binary. Set `ANNS_GT_BINARY` only when you intentionally want to use
another compatible binary.

## Run

```bash
python -m anns_acceptance.cli run --config acceptance_config.yaml
```

Direct pytest is also supported:

```bash
pytest --config acceptance_config.yaml
pytest -m space --config acceptance_config.yaml
pytest -m selectivity --config acceptance_config.yaml
pytest -m static --config acceptance_config.yaml
pytest -m "static or dynamic" --config acceptance_config.yaml
pytest -m single_query --config acceptance_config.yaml
```

`dynamic` intentionally depends on the worst foreground selector selected by the
`static` test in the same `results_dir`. Running `pytest -m dynamic` alone is
invalid and should fail before mutation work starts.

## Smoke

```bash
python mock_adapter/create_smoke_dataset.py --out /tmp/anns-smoke
pytest --config /tmp/anns-smoke/acceptance_config.positive.yaml
```

A negative smoke config is also generated and should fail because the mock search latency is intentionally too high:

```bash
pytest --config /tmp/anns-smoke/acceptance_config.negative.yaml
```

A second negative smoke config should fail because the mock search process intentionally exceeds the single-query max RSS threshold:

```bash
pytest -m single_query --config /tmp/anns-smoke/acceptance_config.rss_negative.yaml
```

Single-query RSS is measured through two channels. The official acceptance RSS
is GNU `/usr/bin/time -v`; result rows report it as both
`time_v_max_rss_bytes` and `max_rss_bytes`. The harness also samples the adapter
process tree with `psutil` and reports `psutil_max_rss_bytes`,
`rss_measurement_delta_bytes`, and `rss_measurement_ratio` as a sanity check. By
default the single-query test requires time-v RSS and uses only that value for
the RSS threshold. Set `require_psutil_rss_sanity: true` when debugging the
measurement path and you want the test to fail on missing or divergent psutil
samples.

## Outputs

The results directory contains JSON/JSONL/CSV artifacts, including space audit, label selectivity, static search, dynamic update chain, foreground latency, mutation timing, single-query resources, and `acceptance_summary.json`.
