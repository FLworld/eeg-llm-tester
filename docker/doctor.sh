#!/usr/bin/env bash
# Preflight for testers: is everything in place to run eeg-llm? Prints a PASS/FAIL per check.
# Runs on the HOST (Ollama + Docker live here). Non-zero exit if a hard requirement is missing.
set -uo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
CHAT_MODEL="${CHAT_MODEL:-eeg-qwen}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
fail=0
pass() { echo "  ✓ $*"; }
warn() { echo "  ! $*"; }
bad()  { echo "  ✗ $*"; fail=1; }

echo "eeg-llm doctor:"

# 1. Docker
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  pass "Docker is installed and the daemon is running"
else
  bad "Docker not available — install Docker Desktop and start it (https://docker.com)"
fi

# 2. Ollama reachable
tags="$(curl -fsS "${OLLAMA_HOST}/api/tags" 2>/dev/null)"
if [ -n "$tags" ]; then
  pass "Ollama reachable at ${OLLAMA_HOST}"
  # 3. Models present
  for m in "$CHAT_MODEL" "$EMBED_MODEL"; do
    if echo "$tags" | grep -q "\"${m}\(:[^\"]*\)\?\""; then
      pass "model present: ${m}"
    else
      bad "model missing: ${m} — run: make setup"
    fi
  done
else
  bad "Ollama not reachable at ${OLLAMA_HOST} — start it (\`ollama serve\` or the Ollama app)"
fi

# 3b. Can a CONTAINER reach the host's Ollama? The #1 trap: Ollama bound to 127.0.0.1 answers on
# the host but REFUSES the container. Fix = bind it to 0.0.0.0. Only meaningful if docker is up.
if [ -n "$tags" ] && command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  if docker run --rm --add-host=host.docker.internal:host-gateway curlimages/curl:latest \
       -fsS http://host.docker.internal:11434/api/tags >/dev/null 2>&1; then
    pass "the app container can reach host Ollama"
  else
    bad "Ollama is not reachable from inside a container (it's bound to 127.0.0.1).
       Make it listen on all interfaces, then restart it:
         macOS:   launchctl setenv OLLAMA_HOST 0.0.0.0:11434
                  osascript -e 'quit app \"Ollama\"'; open -a Ollama
         Linux:   OLLAMA_HOST=0.0.0.0:11434 ollama serve
         Windows: setx OLLAMA_HOST 0.0.0.0:11434   (then quit + reopen Ollama)"
  fi
fi

# 4. RAM (>=16 GB recommended for the 14B model)
mem_gb=0
if [ "$(uname)" = "Darwin" ]; then
  mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
elif [ -r /proc/meminfo ]; then
  mem_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
fi
if [ "$mem_gb" -ge 16 ]; then
  pass "RAM: ${mem_gb} GB"
elif [ "$mem_gb" -gt 0 ]; then
  warn "RAM: ${mem_gb} GB — 16 GB+ recommended; the 14B model will swap and run slowly"
else
  warn "RAM: could not determine"
fi

# 5. Data mount dir
if [ -d data-in ] || mkdir -p data-in 2>/dev/null; then
  pass "data-in/ present (mounts to /data in the app)"
else
  bad "cannot create data-in/ here — run make from the tester package directory"
fi

# 6. In-image runtime smoke (needs the image built; skip cleanly if not)
if docker image inspect eeg-llm:latest >/dev/null 2>&1 || [ -f Dockerfile ]; then
  echo "→ runtime smoke (in image):"
  if docker compose run --rm --entrypoint python eeg-llm /usr/local/bin/smoke.py 2>/dev/null; then
    pass "runtime smoke passed"
  else
    warn "runtime smoke did not pass yet — build first with: make build"
  fi
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "DOCTOR: OK — you're ready.  make run"
else
  echo "DOCTOR: FAIL — fix the ✗ items above, then re-run: make doctor"
fi
exit "$fail"
