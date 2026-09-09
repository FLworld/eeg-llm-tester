# Install / launch verification

This records what has been tested about the setup and launch process, so you know what to expect.
Last updated: 2026-09-09.

## Automated (per commit, GitHub Actions — `.github/workflows/launch-tests.yml`)

- **Windows launcher (`windows-latest`, real Windows PowerShell):** parses `start-eeg-llm.ps1` and
  exercises every launch scenario with Docker/Ollama mocked — `success, embedding-missing,
  docker-off, ollama-off, build-fail, preflight-fail, up-fail, windows-containers, check, stop,
  prebuilt`. All pass; every failure scenario exits non-zero and never reports "Ready".
- **Linux launcher + build (`ubuntu-latest`):** the bash launcher's failure paths (10 tests), then a
  `linux/amd64` image build and the in-image runtime smoke test (`SMOKE: PASS`, incl. the scipy pin).

## Manual / assisted verification (2026-09-09)

- **macOS/Linux launcher, real end-to-end:** from a clean checkout, `start-eeg-llm.sh` builds the
  image, runs the in-image preflight, `docker compose up --wait`, and serves **HTTP 200**; re-running
  is idempotent (no rebuild); `--check` mutates nothing; `--stop` stops cleanly; missing-Ollama and
  port-in-use both fail loud with actionable messages.
- **Windows launcher logic:** the `.bat` → `.ps1` chain and all mocked scenarios pass under native
  PowerShell; `preflight.py` is present in the image at the path the launcher invokes.
- **Cross-architecture:** the image builds and passes the in-image smoke test on **both**
  `linux/arm64` (Apple Silicon) and `linux/amd64` (Intel/AMD, Windows/WSL).
- **Fresh-clone integrity:** a clean clone imports all runtime modules with no missing files.

## Known limitation

Full Docker Desktop + Ollama **end-to-end on physical Windows** is not run in CI (hosted runners have
no GPU/engine); it is a manual pre-release check. The per-commit Windows job validates the real
PowerShell launcher logic (paths, quoting, exit codes) with Docker/Ollama mocked.

## What to expect on your machine

First run downloads ~40 GB (models + image) and can take 5–20 minutes plus download time; later runs
reuse everything. If the app reports it cannot reach Ollama, apply **Step 0** in `README.md` (bind
Ollama to `0.0.0.0:11434`) and start again.
