#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$repo_root/tools/cpp/compute_groundtruth.cpp"
out="$repo_root/tools/bin/compute_groundtruth"

mkdir -p "$(dirname "$out")"

cxx="${CXX:-g++}"
"$cxx" -O3 -std=c++17 -pthread -march=native "$src" -o "$out"
echo "$out"
