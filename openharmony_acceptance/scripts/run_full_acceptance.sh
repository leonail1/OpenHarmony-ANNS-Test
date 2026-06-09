#!/usr/bin/env bash
set -euo pipefail

PIPEANN_ROOT=${PIPEANN_ROOT:-$(pwd)}
BUILD_DIR=${BUILD_DIR:-"${PIPEANN_ROOT}/build"}
OH_BIN_DIR=${OH_BIN_DIR:-"${BUILD_DIR}/openharmony_acceptance"}
TEST_BIN_DIR=${TEST_BIN_DIR:-"${BUILD_DIR}/tests"}
UTIL_BIN_DIR=${UTIL_BIN_DIR:-"${BUILD_DIR}/tests/utils"}
WORK_DIR=${WORK_DIR:-"${PIPEANN_ROOT}/acceptance_work/full"}
RESULTS_DIR=${RESULTS_DIR:-"${PIPEANN_ROOT}/acceptance_results/full"}
THREADS=${THREADS:-$(nproc)}

: "${BASE_BIN:?Set BASE_BIN to the 1M base .bin file}"
: "${UPDATES_BIN:?Set UPDATES_BIN to at least 3M update vectors in .bin format}"
: "${QUERY_BIN:?Set QUERY_BIN to the query .bin file}"

TYPE=${TYPE:-float}
METRIC=${METRIC:-l2}
NPOINTS=${NPOINTS:-1000000}
NQUERIES=${NQUERIES:-1000}
CYCLES=${CYCLES:-5}
R=${R:-96}
R_DENSE=${R_DENSE:-0}
BUILD_L=${BUILD_L:-128}
PQ_BYTES=${PQ_BYTES:-32}
MEM_GB=${MEM_GB:-64}
SEARCH_L=${SEARCH_L:-100}
L_CANDIDATES=${L_CANDIDATES:-20,40,60,80,100,150,200,300,400,600,800}
K=${K:-10}
INDEX_PREFIX=${INDEX_PREFIX:-"${WORK_DIR}/index/sift1m"}

safe_reset_dir() {
  local dir="$1"
  if [[ -z "${dir}" || "${dir}" == "/" ]]; then
    echo "Refusing to remove unsafe directory: ${dir}" >&2
    exit 1
  fi
  case "${dir}" in
    *acceptance_work|*acceptance_work/*|*acceptance_results|*acceptance_results/*)
      rm -rf "${dir}"
      ;;
    *)
      if [[ "${ALLOW_ACCEPTANCE_RM:-0}" == "1" ]]; then
        rm -rf "${dir}"
      else
        echo "Refusing to remove ${dir}; set ALLOW_ACCEPTANCE_RM=1 to override" >&2
        exit 1
      fi
      ;;
  esac
}

safe_reset_dir "${WORK_DIR}"
safe_reset_dir "${RESULTS_DIR}"
mkdir -p "${WORK_DIR}/gt" "${RESULTS_DIR}"
mkdir -p "$(dirname "${INDEX_PREFIX}")"

"${OH_BIN_DIR}/oh_generate_labels" \
  --npoints "${NPOINTS}" \
  --nqueries "${NQUERIES}" \
  --index-prefix "${INDEX_PREFIX}" \
  --out-dir "${WORK_DIR}/labels"

QUERY_ACTIVE="${WORK_DIR}/query_${NQUERIES}.bin"
"${OH_BIN_DIR}/oh_materialize_cycle_vectors" \
  --type "${TYPE}" \
  --base "${QUERY_BIN}" \
  --updates "${QUERY_BIN}" \
  --cycle 0 \
  --npoints "${NQUERIES}" \
  --out "${QUERY_ACTIVE}"

"${TEST_BIN_DIR}/build_disk_index_filtered" \
  "${TYPE}" "${BASE_BIN}" "${INDEX_PREFIX}" \
  "${R}" "${R_DENSE}" "${BUILD_L}" "${PQ_BYTES}" "${MEM_GB}" "${THREADS}" "${METRIC}" pq \
  label_spmat "${WORK_DIR}/labels/base_labels.spmat" \
  range "${WORK_DIR}/labels/base_range.bin"

"${OH_BIN_DIR}/oh_build_space" \
  --raw "${BASE_BIN}" \
  --index-prefix "${INDEX_PREFIX}" \
  --out-json "${RESULTS_DIR}/space_audit.json" \
  --out-csv "${RESULTS_DIR}/space_audit.csv"

while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
  [[ "${selector_id}" == "selector_id" ]] && continue
  GT="${WORK_DIR}/gt/cycle0_${selector_id}.bin"
  if [[ "${selector_type}" == "match_all" ]]; then
    "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${BASE_BIN}" "${QUERY_ACTIVE}" "${K}" "${GT}" null null
    LABEL_ARG="null"
  else
    "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${BASE_BIN}" "${QUERY_ACTIVE}" "${K}" "${GT}" null "${label_config}"
    LABEL_ARG="${label_config}"
  fi
  "${OH_BIN_DIR}/oh_static_filtered" \
    --type "${TYPE}" \
    --metric "${METRIC}" \
    --index-prefix "${INDEX_PREFIX}" \
    --query "${QUERY_ACTIVE}" \
    --gt "${GT}" \
    --label-config "${LABEL_ARG}" \
    --selector-id "${selector_id}" \
    --threads "${THREADS}" \
    --L "${SEARCH_L}" \
    --L-candidates "${L_CANDIDATES}" \
    --recall-min "${RECALL_MIN:-98.0}" \
    --k "${K}" \
    --out-jsonl "${RESULTS_DIR}/static_filtered.jsonl"
done < "${WORK_DIR}/labels/selector_manifest.csv"

UPDATE_ROWS_PER_CYCLE=$((NPOINTS * 6 / 10))
for cycle in $(seq 1 "${CYCLES}"); do
  CYCLE_BIN="${WORK_DIR}/cycle${cycle}.bin"
  "${OH_BIN_DIR}/oh_materialize_cycle_vectors" \
    --type "${TYPE}" \
    --base "${BASE_BIN}" \
    --updates "${UPDATES_BIN}" \
    --cycle "${cycle}" \
    --npoints "${NPOINTS}" \
    --update-rows-per-cycle "${UPDATE_ROWS_PER_CYCLE}" \
    --out "${CYCLE_BIN}"
  while IFS=, read -r selector_id selector_type target_selectivity candidate_count query_file label_config; do
    [[ "${selector_id}" == "selector_id" ]] && continue
    GT="${WORK_DIR}/gt/cycle${cycle}_${selector_id}.bin"
    if [[ "${selector_type}" == "match_all" ]]; then
      "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${CYCLE_BIN}" "${QUERY_ACTIVE}" "${K}" "${GT}" null null
    else
      "${UTIL_BIN_DIR}/compute_groundtruth" "${TYPE}" "${METRIC}" "${CYCLE_BIN}" "${QUERY_ACTIVE}" "${K}" "${GT}" null "${label_config}"
    fi
  done < "${WORK_DIR}/labels/selector_manifest.csv"
done

FOREGROUND_CONFIG=${FOREGROUND_CONFIG:-"${WORK_DIR}/labels/intersect_s25.json"}
"${OH_BIN_DIR}/oh_dynamic_chain" \
  --type "${TYPE}" \
  --metric "${METRIC}" \
  --index-prefix "${INDEX_PREFIX}" \
  --updates "${UPDATES_BIN}" \
  --query "${QUERY_ACTIVE}" \
  --label-config "${FOREGROUND_CONFIG}" \
  --label-index "${INDEX_PREFIX}.label.0" \
  --range-index "${INDEX_PREFIX}.label.1" \
  --npoints "${NPOINTS}" \
  --cycles "${CYCLES}" \
  --insert-threads "${THREADS}" \
  --search-threads "${THREADS}" \
  --merge-threads "${THREADS}" \
  --L "${SEARCH_L}" \
  --L-candidates "${L_CANDIDATES}" \
  --recall-min "${RECALL_MIN:-98.0}" \
  --selector-manifest "${WORK_DIR}/labels/selector_manifest.csv" \
  --gt-dir "${WORK_DIR}/gt" \
  --out-jsonl "${RESULTS_DIR}/dynamic_chain.jsonl" \
  --out-foreground-jsonl "${RESULTS_DIR}/foreground_latency.jsonl" \
  --out-checkpoint-jsonl "${RESULTS_DIR}/dynamic_checkpoint_search.jsonl"

/usr/bin/time -v "${OH_BIN_DIR}/oh_single_query" \
  --type "${TYPE}" \
  --metric "${METRIC}" \
  --index-prefix "${INDEX_PREFIX}" \
  --query "${QUERY_ACTIVE}" \
  --label-config "${WORK_DIR}/labels/range_s10.json" \
  --selector-id range_s10 \
  --L "${SEARCH_L}" \
  --k "${K}" \
  --out-jsonl "${RESULTS_DIR}/single_query_resource.jsonl" \
  2> "${RESULTS_DIR}/single_query_time.txt"

"${OH_BIN_DIR}/oh_summarize_results" \
  --results-dir "${RESULTS_DIR}" \
  --out-json "${RESULTS_DIR}/acceptance_summary.json" \
  --space-expansion-lt "${SPACE_EXPANSION_LT:-2.0}" \
  --recall-min "${RECALL_MIN:-98.0}" \
  --latency-lt "${LATENCY_LT:-10.0}" \
  --delete-ms-per-vector-lte "${DELETE_MS_PER_VECTOR_LTE:-0.5}" \
  --single-query-max-rss-bytes-lt "${SINGLE_QUERY_MAX_RSS_BYTES_LT:-30000000}"

echo "Full acceptance complete: ${RESULTS_DIR}"
