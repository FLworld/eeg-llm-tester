#!/usr/bin/env bash
# One-step launcher for eeg-llm — no commands to type for day-to-day starts.
# Idempotent: the ~18 GB model pull and the image build run ONLY if not already done.
# macOS users can double-click "Start eeg-llm.command" (which calls this).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

CHAT_MODEL="${CHAT_MODEL:-eeg-qwen}"
IMAGE="${IMAGE:-eeg-llm}"

echo "== eeg-llm launcher =="

# 1) Models (once): pull + build eeg-qwen only if missing.
if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | grep -q "^${CHAT_MODEL}"; then
  echo "  ✓ model ${CHAT_MODEL} present"
else
  echo "  → first-time model setup (~18 GB, one-time)…"
  make setup || { echo "  ✗ make setup failed — is Ollama installed & running?"; exit 1; }
fi

# 2) Image (once): build only if missing.
if docker image inspect "${IMAGE}:latest" >/dev/null 2>&1; then
  echo "  ✓ app image present"
else
  echo "  → building app image (first time, ~5 min)…"
  make build || { echo "  ✗ make build failed — is Docker Desktop running?"; exit 1; }
fi

# 3) Preflight, then run.
echo "  → checking prerequisites (make doctor)…"
make doctor || echo "  ! doctor reported issues above — attempting to start anyway"
echo "  → starting the app at http://127.0.0.1:8001 (Ctrl-C to stop)"
make run
