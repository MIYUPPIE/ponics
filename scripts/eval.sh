#!/usr/bin/env bash
# Periodic eval: trains any missing learned model (train suite, seeds >= 1000), then runs
# 100 seeded 35-day episodes and checks them against evals/baseline.json and
# evals/requirements.json. Run before every push that touches sim, chem, control or learning.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { env -u PYTHONPATH uv run --frozen "$@"; }
[ -f models/gp.pkl ] || run ponics-train gp
[ -f models/ppo.zip ] || run ponics-train ppo
exec env -u PYTHONPATH uv run --frozen ponics-eval --check "$@"
