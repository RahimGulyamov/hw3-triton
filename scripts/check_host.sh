#!/usr/bin/env bash
# Read-only preflight; can be streamed to a remote machine through SSH.
set -u
failures=0

printf '\nArchitecture and CPU\n'
uname -sm
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'ATTENTION: the current solution targets Linux x86_64.\n'
  failures=$((failures + 1))
fi
getconf _NPROCESSORS_ONLN
printf '\nMemory (8 GiB recommended)\n'
free -h
printf '\nDisk (30 GiB free recommended for images and results)\n'
df -h /

printf '\nDocker access\n'
if command -v docker >/dev/null 2>&1; then
  if docker version; then
    docker_directory=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    if [[ -n "$docker_directory" && -d "$docker_directory" ]]; then
      df -h "$docker_directory"
    fi
  else
    printf 'ATTENTION: Docker daemon is unavailable to this SSH user.\n'
    failures=$((failures + 1))
  fi
  if ! docker compose version; then
    printf 'ATTENTION: Docker Compose v2 is required.\n'
    failures=$((failures + 1))
  fi
else
  printf 'ATTENTION: Docker is not installed or is not in PATH.\n'
  failures=$((failures + 1))
fi

printf '\nHTTPS connectivity (does not pull images or packages)\n'
if command -v curl >/dev/null 2>&1; then
  for url in https://nvcr.io/v2/ https://pypi.org/simple/numpy/ https://download.pytorch.org/whl/cpu/; do
    printf '%s: ' "$url"
    # A registry HTTP 401 is normal here: the anonymous token flow is not run.
    if http_status=$(curl --silent --show-error --location --output /dev/null \
      --connect-timeout 10 --max-time 20 --write-out '%{http_code}' "$url"); then
      printf 'HTTP %s\n' "$http_status"
      if [[ "$http_status" != 200 && !( "$url" == https://nvcr.io/v2/ && "$http_status" == 401 ) ]]; then
        printf 'ATTENTION: unexpected HTTP response from build dependency source.\n'
        failures=$((failures + 1))
      fi
    else
      printf 'Connection failed\n'
      failures=$((failures + 1))
    fi
  done
else
  printf 'curl unavailable; HTTPS connectivity has not been checked.\n'
fi

printf '\nPreflight completed; blocking checks: %s\n' "$failures"
printf 'No packages were installed and no containers were started.\n'
exit "$((failures > 0))"
