# Result Artifacts

Expected result files:

- `space_audit.json`
- `space_audit.csv`
- `label_selectivity_results.jsonl`
- `label_selectivity_failures.csv`
- `static_filtered_search_results.jsonl`
- `static_filtered_search_failures.csv`
- `static_filtered_search_resource.jsonl`
- `static_foreground_worst_selectors.jsonl`
- `dynamic_foreground_worst_selectors.jsonl`
- `dynamic_update_chain_results.jsonl`
- `dynamic_update_foreground_latency.jsonl`
- `insert_api_timing.jsonl`
- `delete_api_timing.jsonl`
- `dynamic_update_failures.csv`
- `single_query_latency.jsonl`
- `single_query_resource.jsonl`
- `single_query_resource_failures.csv`
- `acceptance_summary.json`

`acceptance_summary.json` records the pytest exit status, aggregate pass/fail, produced artifacts, and failure files.

`single_query_resource.jsonl` includes `max_rss_bytes` as the official GNU
`/usr/bin/time -v` RSS. It also records `time_v_max_rss_bytes`,
`psutil_max_rss_bytes`, `rss_measurement_delta_bytes`, and
`rss_measurement_ratio`; the psutil channel is a sanity check, not the official
RSS source.
