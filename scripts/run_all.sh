#!/usr/bin/env bash
# Run outside the agent sandbox if it cannot access Docker Desktop's socket.
set -euo pipefail
project_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_directory"

# Check access before building images or creating measurement directories.
docker info > /dev/null
if [[ -d results/model_analyzer/model_repository || -d results/perf_analyzer/baseline ]]; then
  printf 'Existing benchmark results found. Move results to another directory before a fresh run.\n' >&2
  exit 1
fi
mkdir -p results/checks
uname -sm > results/host.txt
docker version >> results/host.txt
docker compose version >> results/host.txt

# Validate both images before starting the measurements.
docker compose --profile tools build triton analyzer
docker compose up -d --wait triton
docker compose --profile tools run --rm client clients/smoke.py \
  2>&1 | tee results/checks/smoke.log
docker compose --profile tools run --rm client clients/infer.py \
  --protocol http --url triton:8000 --verify \
  2>&1 | tee results/checks/http.log
docker compose --profile tools run --rm client clients/infer.py \
  --protocol grpc --url triton:8001 --batch-size 4 --verify \
  2>&1 | tee results/checks/grpc.log
docker compose --profile tools run --rm client clients/check_metrics.py \
  --url triton:8000 --metrics-url http://triton:8002/metrics \
  2>&1 | tee results/checks/metrics.log
docker compose --profile tools run --rm perf
docker compose logs --no-color triton > results/checks/triton.log

docker compose stop triton
docker compose --profile tools run --rm analyzer
printf 'Measurements saved to %s/results. Compare configurations before choosing the final one.\n' "$project_directory"
