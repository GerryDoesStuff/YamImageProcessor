#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v poetry >/dev/null 2>&1; then
    run_cmd=(poetry run)
else
    run_cmd=()
fi

run() {
    echo "→ $*"
    "$@"
}

if [[ ${#run_cmd[@]} -gt 0 ]]; then
    run "${run_cmd[@]}" black .
    run "${run_cmd[@]}" flake8 .
    run "${run_cmd[@]}" mypy .
else
    run black .
    run flake8 .
    run mypy .
fi

