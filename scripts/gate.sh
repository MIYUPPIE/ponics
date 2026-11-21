#!/usr/bin/env bash
# Gate tests: deterministic, local, a few seconds. Run by the pre-commit hook.
# PYTHONPATH is cleared so a sourced ROS 2 install cannot inject its pytest plugins.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env -u PYTHONPATH uv run --frozen pytest "$@"
