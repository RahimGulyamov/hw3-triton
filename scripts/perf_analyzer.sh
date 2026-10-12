#!/usr/bin/env bash
set -euo pipefail
destination=${1:-/results/perf_analyzer/baseline}
mkdir -p "$destination"
command -v perf_analyzer > "$destination/executable.txt"
printf '%s\n' 'measurement_interval_ms=15000' 'stability_percentage=10' \
  'max_trials=20' 'stability_latency=average' > "$destination/settings.txt"
for concurrency in 1 2 4 8 16 32; do
  perf_analyzer -m mobilenet -i http -u "${TRITON_URL:-localhost:8000}" \
    -b 1 --concurrency-range "$concurrency" \
    --measurement-interval=15000 --stability-percentage=10 --max-trials=20 \
    -f "$destination/concurrency_${concurrency}.csv" \
    2>&1 | tee "$destination/concurrency_${concurrency}.log"
done
