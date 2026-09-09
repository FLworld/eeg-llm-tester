# eeg-llm — tester quickstart

A local, chat-driven EEG analysis assistant. You describe a pipeline in plain language; it drafts
the exact MNE-Python steps and runs them only after you approve. **Everything runs on your own
machine — your recordings never leave it.**

The app runs in Docker; the language model runs in **Ollama on your machine** (Docker can't use the
GPU directly, so Ollama stays on the host and the app connects to it).

---

## Prerequisites

- **macOS (Apple Silicon or Intel), Linux, or Windows.** The double-click launcher works natively on
  each; only the manual `make` path on **Windows** needs WSL2 (Docker Desktop uses it as its backend
  either way, but you don't have to work inside an Ubuntu terminal).
- **Docker Desktop** — https://docker.com (installed, and opened at least once so its engine runs)
- **Ollama** — https://ollama.com (installed and running)
- **~40 GB free disk** (models + image build space) and **16 GB+ RAM** (the model is a 14B; less will swap and be slow)

## Setup

**Step 0 — let the app reach Ollama (do this once).** By default Ollama listens only on `127.0.0.1`,
which a container can't reach, so this one step avoids the most common snag. You don't have to do it
up front — if you skip it, the launcher's preflight (and `make doctor`) reports the exact fix — but
doing it once now is simplest. Pick your OS:

- **macOS:**
  ```bash
  launchctl setenv OLLAMA_HOST 0.0.0.0:11434
  osascript -e 'quit app "Ollama"'; open -a Ollama
  ```
- **Linux:** run Ollama bound to all interfaces — `OLLAMA_HOST=0.0.0.0:11434 ollama serve` (or, if
  Ollama runs under systemd, add `Environment="OLLAMA_HOST=0.0.0.0:11434"` via
  `sudo systemctl edit ollama`, then `sudo systemctl restart ollama`).
- **Windows (WSL2):** in PowerShell, `setx OLLAMA_HOST 0.0.0.0:11434`, then quit and reopen Ollama.
  (`host.docker.internal` is automatic on Docker Desktop.)

**Easiest — double-click (recommended).** One step does everything (model setup, image build,
prerequisite checks, launch) and is idempotent, so it's also how you start day-to-day:

- **macOS:** double-click **`Start eeg-llm.command`**
- **Windows:** double-click **`Start eeg-llm.bat`**
- **Linux:** run **`./start-eeg-llm.sh`**

The first run pulls the models and builds the image (~40 GB, one-time, 5–20 min); later runs reuse
them. When it prints `Ready:` the browser opens at **http://127.0.0.1:8001**. See `START-HERE.html`
for the full illustrated walkthrough. (Diagnostics only: add `--check` on Mac/Linux or `-Check` on
Windows; stop with `--stop` / `-Stop`.)

**Or, step by step with `make`** (any platform; on Windows run these inside WSL2):

```bash
git clone <repo-url> && cd eeg-llm-tester
make setup     # one-time: pulls the models into Ollama + builds eeg-qwen (~18 GB, slow once)
make build     # builds the app image locally (~5 min the first time)
make doctor    # checks Docker, Ollama, models, RAM, mounts — fix any ✗ it reports
make run       # starts the app
```

Then open **http://127.0.0.1:8001**. (If `make doctor` is all green, you're ready.)

## Try the bundled sample

```bash
make fetch-sample   # puts an ERP CORE N170 subject + its codebook into ./data-in
```
In the app:
```
/scope sub-002.set
/plan load it, band-pass 0.1-30 Hz, average reference, epoch faces vs cars, measure the N170 at PO8
```
Review the drafted plan, then `/run`.

## Use your own data

1. Get your recordings in, either way:
   - **Drag & drop** them straight into the chat — the app saves them to `data-in/` and scopes the
     recording automatically. For multi-file formats drag the whole set together (a `.set` with its
     `.fdt`; a `.vhdr` with its `.eeg` + `.vmrk`).
   - Or put them in the **`data-in/`** folder next to this file and refer to them **by name** —
     e.g. `/scope my-recording.set` (not the full path on your computer).
2. If your event codes aren't described by a BIDS `events.tsv`, tell it what they mean:
   ```
   /codebook {"conditions": {"target": [[1,40]], "standard": [[41,80]]}, "responses": {"correct": 201}}
   ```
3. `/plan …` or `/sweep …`, review, `/run`.

Useful commands: `/params <tool>` (what a tool takes), `/sweep` (parameter sweeps),
`/batch <dataset-dir>` (a whole cohort), `/engine mne|erplab`, `/save-plan <name>` + `/recipes`
(save and reuse a pipeline). Reference PDFs can be dropped in `docs/` and indexed with `make ingest`.

## Updating

When I push a new version: `git pull && make build && make run`.

## Troubleshooting

- **App says "Ollama not reachable" even though Ollama is running** — you skipped Step 0. Run it and
  restart Ollama. `make doctor` detects this and prints the exact fix.
- **"Ollama not reachable" and it's genuinely off** — start Ollama (open the app, or `ollama serve`), then `make run` again.
- **"port is already in use" / 8001 taken** — run on another port: `APP_PORT=8011 make run`, then open http://127.0.0.1:8011.
- **"model missing"** — run `make setup`.
- **"file not found" after `/scope name.set`** — the file must be inside `data-in/`; reference it by
  name, not by its path on your computer.
- **Slow / laggy** — expected on 16 GB RAM with a 14B model; close other heavy apps.
- **Password prompt** — only if you set `APP_PASSWORD`; unset it for no login.

## Not in this build

This is the **runtime** package. The genuine-`runica.m`/Octave validation path is a developer tool and
is intentionally absent, so `run_ica(engine='eeglab')` / `'eeglab-python'` will error — use the default
MNE engine (`run_ica`, or `/engine mne`). Everything a normal analysis needs is included.
