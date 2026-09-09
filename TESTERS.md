# eeg-llm — tester quickstart

**Double-click [START-HERE.html](START-HERE.html) for the step-by-step setup guide and download links.**

1. Install **Docker Desktop** for your chip (Apple silicon/ARM64 or Intel/AMD64), open it,
   and wait for the engine. Windows uses the WSL 2 backend and Linux containers.
2. Install and open **[Ollama](https://ollama.com/download)**. No manual model selection is needed.
3. Extract the entire project ZIP to a writable folder.
4. Double-click **Start eeg-llm.command** on Mac or **Start eeg-llm.bat** on Windows.
   It opens a terminal, downloads missing models, builds the app, runs checks, then opens the browser.
5. When it says **Ready**, you can close the terminal. Stop the app using Docker Desktop.

No Git, Python, make, Homebrew or host compiler is needed. Linux users can install Docker Engine
with Compose v2 and Ollama, then run `bash start-eeg-llm.sh`.

Allow **40 GB free disk** for models and build space, plus recordings. **16 GB RAM recommended;
32 GB preferred**. First start needs internet and can take 5–20+ minutes plus download time.
Intel Macs use CPU inference. Windows ARM remains experimental and needs hardware acceptance
testing; see [verification status](INSTALL-VERIFICATION.md).

Both project copies use the same launchers. If running both simultaneously, put `APP_PORT=8011`
in the tester folder's `.env` file. The browser opens the configured port automatically.

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

Replace the application files with the new version (or `git pull` if using Git), keep your
`data-in`, `data-out`, `.env`, and project folder name, then double-click Start again.
It rebuilds changed app files and refreshes the model definition using cached downloads.
See **START-HERE.html** for troubleshooting and platform requirements.

## Runtime package

The developer-only Octave/EEGLAB validation engines are absent; use the MNE ICA engine.
