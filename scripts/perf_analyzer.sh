#!/usr/bin/env bash
set -euo pipefail
destination=${1:-/results/perf_analyzer/baseline}
mkdir -p "$destination"
command -v perf_analyzer > "$destination/executable.txt"
for concurrency in 1 2 4 8 16 32; do
  perf_analyzer -m mobilenet -i http -u "${TRITON_URL:-localhost:8000}" \
    -b 1 --concurrency-range "$concurrency" --percentile=99 \
    --measurement-interval=5000 --stability-percentage=10 --max-trials=10 \
    -f "$destination/concurrency_${concurrency}.csv" \
    2>&1 | tee "$destination/concurrency_${concurrency}.log"
done
