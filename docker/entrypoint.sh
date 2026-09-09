#!/usr/bin/env bash
# Container entrypoint. The LLM runs in Ollama on the HOST; this only starts the app once the
# host is reachable and the required models exist. FAIL LOUD — never launch a half-working UI.
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
CHAT_MODEL="${CHAT_MODEL:-eeg-qwen}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

STATE="${STATE_DIR:-/state}"
mkdir -p "$STATE/cache" "$STATE/sessions" "$STATE/recipes" "$STATE/batches"
# The knowledge base ships empty (the model falls back to its own EEG expertise). A tester builds
# their own by dropping PDFs into ./docs and running `make ingest`; it persists in $STATE.

die() { echo ""; echo "  ✗ $*" >&2; echo "" >&2; exit 1; }

echo "→ eeg-llm: waiting for Ollama at ${OLLAMA_HOST} ..."
tags=""
for i in $(seq 1 30); do
  if tags="$(curl --connect-timeout 2 --max-time 3 -fsS "${OLLAMA_HOST}/api/tags" 2>/dev/null)"; then
    echo "  ✓ Ollama reachable"
    break
  fi
  sleep 1
done
[ -n "$tags" ] || die "Ollama not reachable at ${OLLAMA_HOST}.
    Start Ollama on your machine (\`ollama serve\` or open the Ollama app), then re-run.
    On Linux, ensure the container can resolve host.docker.internal (compose sets extra_hosts)."

for m in "$CHAT_MODEL" "$EMBED_MODEL"; do
  # Untagged model names mean :latest, not any available tag.
  case "$m" in *:*) ;; *) m="${m}:latest" ;; esac
  echo "$tags" | grep -Fq "\"${m}\"" \
    || die "Model '${m}' is not present in Ollama on the host.
    Run the one-time bootstrap on your machine:  make setup
    (pulls qwen2.5-coder:14b + nomic-embed-text and builds eeg-qwen — ~10 GB, once)."
done
echo "  ✓ models present: ${CHAT_MODEL}, ${EMBED_MODEL}"

echo "→ launching Chainlit on 0.0.0.0:8001"
exec chainlit run app.py --host 0.0.0.0 --port 8001 -h
