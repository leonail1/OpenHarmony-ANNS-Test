#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Matrix {
  int32_t rows = 0;
  int32_t cols = 0;
  std::vector<float> values;
};

struct Tags {
  int32_t rows = 0;
  int32_t cols = 0;
  std::vector<uint32_t> values;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

template <typename T>
void read_exact(std::ifstream &input, T *target, size_t count, const std::string &path) {
  input.read(reinterpret_cast<char *>(target), static_cast<std::streamsize>(sizeof(T) * count));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(T) * count)) {
    fail("truncated file: " + path);
  }
}

Matrix read_fbin(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("cannot open " + path);
  }
  Matrix matrix;
  read_exact(input, &matrix.rows, 1, path);
  read_exact(input, &matrix.cols, 1, path);
  if (matrix.rows < 0 || matrix.cols <= 0) {
    fail("invalid fbin header: " + path);
  }
  const size_t total = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
  matrix.values.resize(total);
  read_exact(input, matrix.values.data(), total, path);
  return matrix;
}

Tags read_u32bin_or_empty(const std::string &path, int32_t expected_rows) {
  Tags tags;
  if (path == "null" || path.empty()) {
    tags.rows = expected_rows;
    tags.cols = 0;
    return tags;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("cannot open " + path);
  }
  read_exact(input, &tags.rows, 1, path);
  read_exact(input, &tags.cols, 1, path);
  if (tags.rows != expected_rows || tags.cols < 0) {
    fail("invalid tag header: " + path);
  }
  const size_t total = static_cast<size_t>(tags.rows) * static_cast<size_t>(tags.cols);
  tags.values.resize(total);
  if (total != 0) {
    read_exact(input, tags.values.data(), total, path);
  }
  return tags;
}

int env_threads() {
  const char *anns = std::getenv("ANNS_GT_THREADS");
  const char *omp = std::getenv("OMP_NUM_THREADS");
  const char *raw = anns != nullptr ? anns : omp;
  if (raw != nullptr) {
    char *end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end != raw && value > 0 && value < 4096) {
      return static_cast<int>(value);
    }
  }
  unsigned int hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1 : static_cast<int>(hc);
}

inline float l2sqr(const float *a, const float *b, int32_t dims) {
  float acc = 0.0f;
  for (int32_t i = 0; i < dims; ++i) {
    const float diff = a[i] - b[i];
    acc += diff * diff;
  }
  return acc;
}

void compute_range(const Matrix &base, const Matrix &queries, const Tags &tags, int k, int begin, int end,
                   std::vector<int32_t> &out_ids, std::vector<float> &out_dists,
                   std::vector<uint32_t> &out_tags) {
  using HeapItem = std::pair<float, int32_t>;
  for (int qi = begin; qi < end; ++qi) {
    std::priority_queue<HeapItem> heap;
    const float *query = queries.values.data() + static_cast<size_t>(qi) * queries.cols;
    for (int32_t bi = 0; bi < base.rows; ++bi) {
      const float *vector = base.values.data() + static_cast<size_t>(bi) * base.cols;
      const float dist = l2sqr(query, vector, base.cols);
      if (static_cast<int>(heap.size()) < k) {
        heap.emplace(dist, bi);
      } else if (dist < heap.top().first || (dist == heap.top().first && bi < heap.top().second)) {
        heap.pop();
        heap.emplace(dist, bi);
      }
    }
    std::vector<HeapItem> best;
    best.reserve(static_cast<size_t>(k));
    while (!heap.empty()) {
      best.push_back(heap.top());
      heap.pop();
    }
    std::sort(best.begin(), best.end(), [](const HeapItem &a, const HeapItem &b) {
      if (a.first != b.first) {
        return a.first < b.first;
      }
      return a.second < b.second;
    });
    for (int rank = 0; rank < k; ++rank) {
      const size_t offset = static_cast<size_t>(qi) * static_cast<size_t>(k) + static_cast<size_t>(rank);
      if (rank >= static_cast<int>(best.size())) {
        out_ids[offset] = -1;
        out_dists[offset] = std::numeric_limits<float>::infinity();
        out_tags[offset] = std::numeric_limits<uint32_t>::max();
        continue;
      }
      const int32_t local_id = best[rank].second;
      out_ids[offset] = local_id;
      out_dists[offset] = best[rank].first;
      if (tags.cols > 0) {
        out_tags[offset] = tags.values[static_cast<size_t>(local_id) * tags.cols];
      } else {
        out_tags[offset] = static_cast<uint32_t>(local_id);
      }
    }
  }
}

template <typename T>
void write_all(std::ofstream &output, const T *data, size_t count, const std::string &path) {
  output.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(sizeof(T) * count));
  if (!output) {
    fail("failed writing " + path);
  }
}

void write_truthset(const std::string &path, int32_t nqueries, int32_t k, const std::vector<int32_t> &ids,
                    const std::vector<float> &dists, const std::vector<uint32_t> &tags) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    fail("cannot create " + path);
  }
  write_all(output, &nqueries, 1, path);
  write_all(output, &k, 1, path);
  write_all(output, ids.data(), ids.size(), path);
  write_all(output, dists.data(), dists.size(), path);
  write_all(output, tags.data(), tags.size(), path);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 9) {
      std::cerr << "usage: compute_groundtruth float l2 BASE.fbin QUERY.fbin K OUT.bin TAGS.u32bin null\n";
      return 2;
    }
    const std::string dtype = argv[1];
    const std::string metric = argv[2];
    if (dtype != "float" || metric != "l2") {
      fail("only float l2 groundtruth is supported");
    }
    Matrix base = read_fbin(argv[3]);
    Matrix queries = read_fbin(argv[4]);
    if (base.cols != queries.cols) {
      fail("base/query dimension mismatch");
    }
    const int k = std::stoi(argv[5]);
    if (k <= 0 || k > base.rows) {
      fail("invalid k");
    }
    Tags tags = read_u32bin_or_empty(argv[7], base.rows);
    std::vector<int32_t> ids(static_cast<size_t>(queries.rows) * k, -1);
    std::vector<float> dists(static_cast<size_t>(queries.rows) * k, std::numeric_limits<float>::infinity());
    std::vector<uint32_t> out_tags(static_cast<size_t>(queries.rows) * k, std::numeric_limits<uint32_t>::max());

    const int threads = std::max(1, std::min(env_threads(), static_cast<int>(queries.rows)));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int ti = 0; ti < threads; ++ti) {
      const int begin = static_cast<int>((static_cast<int64_t>(queries.rows) * ti) / threads);
      const int end = static_cast<int>((static_cast<int64_t>(queries.rows) * (ti + 1)) / threads);
      workers.emplace_back(compute_range, std::cref(base), std::cref(queries), std::cref(tags), k, begin, end,
                           std::ref(ids), std::ref(dists), std::ref(out_tags));
    }
    for (auto &worker : workers) {
      worker.join();
    }
    write_truthset(argv[6], queries.rows, static_cast<int32_t>(k), ids, dists, out_tags);
  } catch (const std::exception &ex) {
    std::cerr << "compute_groundtruth: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
