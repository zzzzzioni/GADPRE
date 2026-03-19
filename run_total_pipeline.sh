#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_total_pipeline.sh
#   ./run_total_pipeline.sh "2024-05-10 00:00:00" 주중 밤
#   ./run_total_pipeline.sh --date "2024-05-10 00:00:00" --week 주중 --time 밤
#   ./run_total_pipeline.sh --help
#
# Args:
#   $1: date (default: 2024-05-10 00:00:00)
#   $2: week (default: 주중)  -> 주중 | 주말
#   $3: time (default: 밤)    -> 밤 | 새벽

if [[ "${1:-}" == -* ]]; then
  # Forward option-style args directly.
  python3 "./total_pipeline.py" "$@"
else
  DATE_ARG="${1:-2024-05-10 00:00:00}"
  WEEK_ARG="${2:-주중}"
  TIME_ARG="${3:-밤}"

  python3 "./total_pipeline.py" \
    --date "${DATE_ARG}" \
    --week "${WEEK_ARG}" \
    --time "${TIME_ARG}"
fi
