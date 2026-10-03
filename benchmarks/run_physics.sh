#!/bin/bash
# Run benchmarks/physics.py one task per process: run_physics.sh <tasks file or "all"> <physics settings...>
set -u
cd "$(dirname "$0")/.."
tasks=$1; shift
if [ "$tasks" = all ]; then
  tasks=$(uv run python -c 'from metaworld.env import TASKS; print(" ".join(sorted(TASKS)))')
fi
for t in $tasks; do
  uv run python benchmarks/physics.py --tasks "$t" --out "${OUT:-benchmarks/results/physics.json}" --physics "$@" || echo "failed: $t"
done
