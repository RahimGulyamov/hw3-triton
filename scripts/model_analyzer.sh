#!/usr/bin/env bash
set -euo pipefail
mkdir -p /results/model_analyzer
python3 -m pip freeze > /results/model_analyzer/environment.txt
uname -a > /results/model_analyzer/system.txt
if command -v lscpu >/dev/null; then
  lscpu >> /results/model_analyzer/system.txt
fi
model-analyzer profile -f /workspace/profiling/model_analyzer.yaml \
  2>&1 | tee /results/model_analyzer/profile.log
# Generate detailed reports for the actual configurations from this run.
python3 /workspace/scripts/list_configs.py > /results/model_analyzer/config_names.txt
configuration_names=$(paste -sd ',' /results/model_analyzer/config_names.txt)
if [[ -z "$configuration_names" ]]; then
  printf 'Model Analyzer did not generate any configurations\n' >&2
  exit 1
fi
model-analyzer report --report-model-configs "$configuration_names" \
  --checkpoint-directory /results/model_analyzer/checkpoints \
  --export-path /results/model_analyzer \
  2>&1 | tee /results/model_analyzer/report.log
printf 'Configuration names saved to /results/model_analyzer/config_names.txt\n'
