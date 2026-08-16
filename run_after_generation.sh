#!/bin/bash
# Waits for the generation run to exit, then judges via OpenRouter and builds
# the annotation sheet. Safe to re-run: rejudge always rewrites its output dir.
set -u
cd "$(dirname "$0")"

echo "[$(date +%H:%M)] waiting for generation to finish..."
while pgrep -f 'p2e run --config configs/models.ollama.yaml' > /dev/null; do sleep 60; done

N=$(wc -l < results/live/responses.jsonl | tr -d ' ')
echo "[$(date +%H:%M)] generation done: $N responses"
if [ "$N" -lt 2000 ]; then
  echo "WARNING: expected 2000 responses, found $N — the run may have died early."
  echo "Resume with: .venv/bin/p2e run --config configs/models.ollama.yaml --out results/live --workers 1 --resume"
fi

echo "[$(date +%H:%M)] judging via OpenRouter..."
.venv/bin/p2e rejudge --run results/live --config configs/judge.openrouter.yaml \
    --out results/live_judged --workers 8 || { echo "rejudge FAILED"; exit 1; }

echo "[$(date +%H:%M)] building annotation sheet..."
.venv/bin/p2e annotate --run results/live_judged --sample \
    --out data/annotations/human_labels_live.csv || { echo "annotate FAILED"; exit 1; }

echo "[$(date +%H:%M)] DONE — hand-label data/annotations/human_labels_live.csv, then:"
echo "  .venv/bin/p2e report --run results/live_judged --annotations data/annotations/human_labels_live.csv --out reports/findings_live.md --charts reports/charts_live"
