#!/usr/bin/env bash
# macOS/Linux launcher: no host Python, make, Homebrew or compiler required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$PATH:/usr/local/bin:/opt/homebrew/bin:$HOME/.docker/bin:/Applications/Docker.app/Contents/Resources/bin:/Applications/Ollama.app/Contents/Resources"
die() { printf '\nSetup stopped: %s\nSee START-HERE.html, then double-click Start again.\n' "$*" >&2; exit 1; }
trap 'echo "Setup did not finish. Read the error above; retrying Start reuses completed downloads." >&2' ERR
case "${1:-}" in ""|--check|--stop) ;; *) die "Usage: bash start-eeg-llm.sh [--check|--stop]" ;; esac

echo "eeg-llm setup - $(uname -s) / $(uname -m)"
echo "First start needs internet. Allow 40 GB free disk; 16 GB RAM recommended, 32 GB preferred."
echo "[1/5] Checking Docker Desktop"
command -v docker >/dev/null || die "Install Docker Desktop: https://docs.docker.com/desktop/ . Open it and finish setup."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing. Update Docker Desktop."
docker info >/dev/null 2>&1 || die "Open Docker Desktop and wait for its engine to run."
[ "$(docker info --format '{{.OSType}}')" = linux ] || die "Docker must be using Linux containers."
if [ "${1:-}" = --stop ]; then
  docker compose stop
  echo "eeg-llm stopped. Recordings and saved results are kept."
  exit 0
fi

echo "[2/5] Checking Ollama"
command -v ollama >/dev/null || die "Install and open Ollama: https://ollama.com/download . Then close this window and start again."
# Host CLI address is separate from the container's OLLAMA_HOST in Compose/.env.
export OLLAMA_HOST="${EEG_HOST_OLLAMA_URL:-http://127.0.0.1:11434}"
ollama list >/dev/null 2>&1 || die "Open Ollama and wait for it to start. For a custom server, set EEG_HOST_OLLAMA_URL."
mkdir -p data-in data-out docs
if [ "${1:-}" != --check ]; then
  echo "[3/5] Preparing models (roughly 10 GB on first start; keep this window open)"
  ollama show nomic-embed-text:latest >/dev/null 2>&1 || ollama pull nomic-embed-text:latest
  # Reuses cached layers and applies changes to the supplied Modelfile.
  ollama create eeg-qwen -f Modelfile
  echo "[4/5] Preparing app (first build 5-20+ minutes; later builds use cache)"
fi
unset OLLAMA_HOST
if [ "${1:-}" != --check ]; then
  image=$(docker compose config --images)
  if [ "$image" != eeg-llm:latest ]; then
    docker image inspect "$image" >/dev/null 2>&1 || docker compose pull eeg-llm
    echo "Using prebuilt image $image (docker compose pull to update it)."
  else
    docker compose build eeg-llm
  fi
fi
echo "[5/5] Checking the app's connection, both models, storage and scientific libraries"
docker compose run --rm --no-deps --entrypoint python eeg-llm /usr/local/bin/preflight.py
if [ "${1:-}" = --check ]; then echo "CHECK: PASS"; exit 0; fi
if ! docker compose up -d --no-build --wait --wait-timeout 180; then
  docker compose logs --tail 60 eeg-llm
  die "The app did not become ready. If port 8001 is occupied, set APP_PORT=8011 in .env and retry."
fi
binding=$(docker compose port eeg-llm 8001)
url="http://${binding}"
echo "Ready: $url"
echo "The app runs in the background. You can close this window."
echo "Stop it in Docker Desktop, or run: bash start-eeg-llm.sh --stop"
if [ "${EEG_NO_BROWSER:-0}" != 1 ]; then
  if [ "$(uname -s)" = Darwin ]; then open "$url" || true
  elif command -v xdg-open >/dev/null; then xdg-open "$url" >/dev/null 2>&1 || true
  fi
fi
