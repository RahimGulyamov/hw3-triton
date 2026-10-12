#!/usr/bin/env bash
# Inspect the two exact image manifests through SSH without pulling layers.
set -euo pipefail
project_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_directory"
mkdir -p results
ssh -o ConnectTimeout=10 "${1:-one-vm}" 'bash -s' <<'REMOTE' 2>&1 | tee results/registry_check.txt
set -u
failures=0
for image in nvcr.io/nvidia/tritonserver:23.12-py3 nvcr.io/nvidia/tritonserver:23.12-py3-sdk; do
  printf '\nChecking manifest: %s\n' "$image"
  if docker manifest inspect "$image" > /dev/null; then
    printf 'OK: manifest accessible\n'
  else
    printf 'FAILED: manifest unavailable\n'
    failures=$((failures + 1))
  fi
done
printf '\nFailed manifest checks: %s\n' "$failures"
exit "$((failures > 0))"
REMOTE
