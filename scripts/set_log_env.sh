#!/usr/bin/env bash
# Helper to export logging environment variables for parquet_segmenter
# Usage (in your shell):
#   source scripts/set_log_env.sh [--enable-file] [--path /tmp/ps.log] [--level DEBUG]
# Example:
#   source scripts/set_log_env.sh --enable-file --path /tmp/ps.log --level DEBUG

set -euo pipefail

# defaults
LOG_TO_FILE=0
LOG_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_PATH="$REPO_ROOT/parquet_segmenter.log"

# simple arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-file|--file|--to-file)
      LOG_TO_FILE=1
      shift
      ;;
    --disable-file|--no-file)
      LOG_TO_FILE=0
      shift
      ;;
    --path)
      LOG_PATH="$2"
      shift 2
      ;;
    --level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help|-h)
      sed -n '1,120p' "${BASH_SOURCE[0]}"
      return 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      return 2
      ;;
  esac
done

export PARQUET_SEGMENTER_LOG_TO_FILE="$LOG_TO_FILE"
export PARQUET_SEGMENTER_LOG_PATH="$LOG_PATH"
export PARQUET_SEGMENTER_LOG_LEVEL="$LOG_LEVEL"

cat <<EOF
Exported:
  PARQUET_SEGMENTER_LOG_TO_FILE=$PARQUET_SEGMENTER_LOG_TO_FILE
  PARQUET_SEGMENTER_LOG_PATH=$PARQUET_SEGMENTER_LOG_PATH
  PARQUET_SEGMENTER_LOG_LEVEL=$PARQUET_SEGMENTER_LOG_LEVEL

Note: run this with 'source' to export into your current shell, e.g.: 
  source scripts/set_log_env.sh --enable-file --path /tmp/ps.log --level DEBUG
EOF
