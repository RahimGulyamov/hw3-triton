#!/usr/bin/env bash
set -euo pipefail
project_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_directory"
mkdir -p results
ssh -o ConnectTimeout=10 "${1:-one-vm}" 'bash -s' < scripts/check_host.sh \
  2>&1 | tee results/vm_preflight.txt
