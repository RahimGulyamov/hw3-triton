#!/usr/bin/env bash
# Run from a regular Mac terminal: transfer images if necessary, run on Linux,
# and collect measured results. No existing remote directory is overwritten.
set -euo pipefail
project_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_directory"
target=${1:-one-vm}
run_id=$(date -u +%Y%m%d-%H%M%S)-$$
remote_directory="/tmp/hw3-run-$run_id"
local_directory="$project_directory/results/vm_run_$run_id"

# Establish SSH and verify the required platform before downloading images.
remote_platform=$(ssh -o ConnectTimeout=10 "$target" 'uname -sm')
if [[ "$remote_platform" != 'Linux x86_64' ]]; then
  printf 'Expected Linux x86_64; received: %s\n' "$remote_platform" >&2
  exit 1
fi
ssh "$target" 'docker info > /dev/null && docker compose version'
mkdir -p "$local_directory"
printf '%s\n' "$remote_directory" > "$local_directory/remote_directory.txt"

for image in nvcr.io/nvidia/tritonserver:23.12-py3 nvcr.io/nvidia/tritonserver:23.12-py3-sdk; do
  printf '\nPreparing %s\n' "$image" | tee -a "$local_directory/preparation.log"
  architecture=$(ssh "$target" "docker image inspect --format '{{.Architecture}}' '$image'" 2>/dev/null || true)
  if [[ "$architecture" == amd64 ]]; then
    printf 'Image already present on the VM\n'
    continue
  fi
  if ssh "$target" "docker manifest inspect '$image' > /dev/null" 2>> "$local_directory/preparation.log"; then
    ssh "$target" "docker pull --platform linux/amd64 '$image'" \
      2>&1 | tee -a "$local_directory/preparation.log"
  else
    printf 'VM registry access failed; attempting download through local Docker Desktop\n'
    docker info > /dev/null 2>> "$local_directory/preparation.log"
    local_architecture=$(docker image inspect --format '{{.Architecture}}' "$image" 2>/dev/null || true)
    if [[ "$local_architecture" == amd64 ]]; then
      printf 'Using the image already present in local Docker Desktop\n' \
        | tee -a "$local_directory/preparation.log"
    else
      docker pull --platform linux/amd64 "$image" \
        2>&1 | tee -a "$local_directory/preparation.log"
    fi
    docker image save --platform linux/amd64 "$image" 2>> "$local_directory/preparation.log" | gzip -1 \
      | ssh "$target" 'bash -o pipefail -c "gzip -d | docker image load"' \
      2>&1 | tee -a "$local_directory/preparation.log"
  fi
  architecture=$(ssh "$target" "docker image inspect --format '{{.Architecture}}' '$image'")
  if [[ "$architecture" != amd64 ]]; then
    printf 'Wrong architecture for %s: %s\n' "$image" "$architecture" >&2
    exit 1
  fi
done

ssh "$target" "mkdir -m 700 '$remote_directory'"
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' -czf - \
  Dockerfile compose.yaml requirements.txt requirements-analyzer.txt .dockerignore .gitignore \
  model_repository clients scripts profiling examples README.md docs tests \
  | ssh "$target" "tar -xzf - -C '$remote_directory'"

# Keep the run status while collecting diagnostics even if a build/test fails.
remote_status=0
if ssh "$target" "bash -s -- '$remote_directory' 'hw3-cpu-$run_id'" <<'REMOTE' 2>&1 | tee "$local_directory/run.log"
set -euo pipefail
cd "$1"
export COMPOSE_PROJECT_NAME="$2"
bash scripts/run_all.sh
REMOTE
then
  printf '\nVM profiling completed\n'
else
  remote_status=$?
  printf '\nVM run failed with status %s; collecting any available results\n' "$remote_status" >&2
fi
ssh "$target" "tar -C '$remote_directory' --exclude='results/model_analyzer/model_repository' --exclude='results/model_analyzer/checkpoints' -czf - results" \
  | tar -xzf - -C "$local_directory"
printf '\nLogs and reports saved to %s\n' "$local_directory"
printf 'Remote workspace retained at %s:%s\n' "$target" "$remote_directory"
exit "$remote_status"
