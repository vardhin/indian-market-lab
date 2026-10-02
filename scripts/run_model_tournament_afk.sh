#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-tabular}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

mkdir -p reports/ml/model_tournament/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="reports/ml/model_tournament/logs/${MODE}_${STAMP}.log"

run_tabular() {
  echo
  echo "===== INSTALL TABULAR TOURNAMENT EXTRAS ====="
  uv sync --extra tournament

  echo
  echo "===== TABULAR SELF TEST ====="
  uv run python src/ml/model_tournament.py --self-test

  echo
  echo "===== TABULAR MODEL TOURNAMENT ====="
  uv run python src/ml/model_tournament.py

  echo
  echo "===== TABULAR LEADERBOARD ====="
  if [[ -f reports/ml/model_tournament/leaderboard.csv ]]; then
    cat reports/ml/model_tournament/leaderboard.csv
  fi

  echo
  echo "===== TABULAR FAILURES ====="
  if [[ -f reports/ml/model_tournament/failures.csv ]]; then
    cat reports/ml/model_tournament/failures.csv
  fi
}

run_temporal() {
  echo
  echo "===== INSTALL DEEP TOURNAMENT EXTRAS ====="
  uv sync --extra deep

  echo
  echo "===== TEMPORAL SELF TEST ====="
  uv run python src/ml/temporal_tournament.py --self-test

  echo
  echo "===== TEMPORAL MODEL TOURNAMENT ====="
  uv run python src/ml/temporal_tournament.py

  echo
  echo "===== TEMPORAL LEADERBOARD ====="
  if [[ -f reports/ml/model_tournament/temporal/leaderboard.csv ]]; then
    cat reports/ml/model_tournament/temporal/leaderboard.csv
  fi

  echo
  echo "===== TEMPORAL FAILURES ====="
  if [[ -f reports/ml/model_tournament/temporal/failures.csv ]]; then
    cat reports/ml/model_tournament/temporal/failures.csv
  fi
}

{
  echo "============================================================"
  echo "INDIA MODEL TOURNAMENT"
  echo "Mode: $MODE"
  echo "Started: $(date -Is)"
  echo "Repo: $ROOT"
  echo "============================================================"

  case "$MODE" in
    tabular)
      run_tabular
      ;;
    temporal)
      run_temporal
      ;;
    all)
      run_tabular
      run_temporal
      ;;
    *)
      echo "Unknown mode: $MODE"
      echo "Usage:"
      echo "  bash scripts/run_model_tournament_afk.sh tabular"
      echo "  bash scripts/run_model_tournament_afk.sh temporal"
      echo "  bash scripts/run_model_tournament_afk.sh all"
      false
      ;;
  esac

  STATUS=$?

  echo
  echo "============================================================"
  echo "Finished: $(date -Is)"
  echo "Status: $STATUS"
  echo "Log: $LOG"
  echo "============================================================"

  # Do not exit the parent shell. This script simply returns after printing
  # the log location.
} 2>&1 | tee "$LOG"

PIPE_STATUS=( "${PIPESTATUS[@]}" )
STATUS="${PIPE_STATUS[0]:-0}"

echo
echo "FINAL LOG: $LOG"
echo "SCRIPT STATUS: $STATUS"
