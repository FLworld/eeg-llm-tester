# Publishing the eeg-llm image (maintainer)

Two ways to get the built image onto a tester's machine. Both assume the image builds locally
(`make build`) and that Docker Desktop is running. Pick per audience; you can use both.

| | Container registry (GHCR) | Tarball |
|---|---|---|
| Tester setup | one-time `docker login` | none |
| Transfer | `docker compose pull` | you send a ~1–2 GB file |
| Updates | push new tag, tester pulls | resend the file |
| Multi-arch | **clean** (one ref serves amd64 + arm64) | **one tarball per arch** |

The **image contains only the app + frozen Python stack** — no models. Every tester still runs
`make setup` once to pull the ~18 GB of models into their own Ollama. Nothing here changes that.

**Both amd64 and arm64 are supported.** Testers build locally (`make build`), which produces an image
for their own architecture, so Intel Macs / Linux / Windows-WSL (amd64) and Apple Silicon (arm64) are
all covered with no per-arch work. The amd64 build has been validated under emulation. `PLATFORMS` in
the Makefile is `linux/arm64` and only matters for **prebuilt** publishing: to push a prebuilt image to
amd64 testers, set `PLATFORMS = linux/amd64,linux/arm64` for a multi-arch GHCR push, or `make save` a
separate amd64 tarball (build it with `docker buildx build --platform linux/amd64 ... --load` first).

**Use both methods** (this is the chosen plan): GHCR for testers comfortable with a one-time
`docker login`, and a tarball for anyone else. The two are independent — publish once to each.

---

## Option 0 — GitHub repo, tester builds locally (recommended for this round)

Simplest path for a few Apple-Silicon testers: they clone a small repo and build the image on their
own machine. **No registry, no `docker login`, no multi-GB file to send** — the pinned `Dockerfile`
makes the build deterministic, so "build locally" still yields the same frozen stack.

**The repo is runtime-only** — carved out of the dev workspace so testers never receive `tests/`,
`data/`, the dev docs, or the copyrighted PDFs. It contains: the 13 runtime `.py` modules,
`.chainlit/config.toml`, `chainlit.md`, `Modelfile`, `Dockerfile`, `.dockerignore`,
`docker-compose.yml`, `Makefile`, `requirements-runtime.txt`, the `docker/` folder, a `.gitignore`,
and `README.md` (= the tester guide).

**Publish it (one-time):**
```bash
cd ~/eeg-llm-tester            # the carved runtime-only folder
git init && git add -A && git commit -m "eeg-llm tester package"
gh repo create eeg-llm-tester --private --source=. --push
# then add the tester on the repo's Settings → Collaborators, and send them the URL
```
(Web alternative: create an empty private repo, then `git remote add origin <url> && git push -u origin main`.)

**Tester side:** exactly the steps in `README.md` — install Docker + Ollama, the one Ollama-bind
command, then `git clone` → `make setup` → `make build` → `make doctor` → `make run`.

**Updating:** `git push`; testers `git pull && make build && make run`.

---

## Option A — GitHub Container Registry (GHCR)

**1. Make a token.** GitHub → Settings → Developer settings → Personal access tokens (classic) → a
token with the **`write:packages`** scope.

**2. Log in and push** (from `eeg-llm/`):

```bash
export GHCR_USER=<your-github-username>
export GHCR_TOKEN=<the-token>
make login
make push REGISTRY=ghcr.io/$GHCR_USER            # multi-arch build + push :latest
# versioned release:
make push REGISTRY=ghcr.io/$GHCR_USER TAG=2026-09-06
```

(Or set `REGISTRY := ghcr.io/<you>` once in the `Makefile` and just `make push`.)

**3. Access.** New GHCR packages are **private** by default — fine for trusted testers. On the
package's GitHub page you can either keep it private and add each tester as a collaborator, or set it
public (only do that when you're ready for it to be world-readable).

**4. Tester consumption.** In the tester package they set the image ref and pull:

```bash
export EEG_IMAGE=ghcr.io/<you>/eeg-llm:latest
echo <their-token> | docker login ghcr.io -u <their-github-user> --password-stdin   # once
docker compose pull
make setup && make doctor && make run
```

`docker-compose.yml` already reads `${EEG_IMAGE:-eeg-llm:latest}`, so setting `EEG_IMAGE` makes
compose run the pulled image instead of building. A private image requires the tester's own `docker
login` (add them as collaborators on the package).

**Updating:** `make push TAG=…` a new version; testers `docker compose pull && make run`.

---

## Option B — Tarball (no registry, no accounts)

Build for the tester's architecture and save a gzipped image file:

```bash
make save                       # builds + saves for THIS machine's arch (arm64 on your Mac)
# for Intel/Windows/Linux testers, build the other arch explicitly:
docker buildx build --platform linux/amd64 -t eeg-llm:latest --load .
docker save eeg-llm:latest | gzip > eeg-llm-latest-x86_64.tar.gz
```

Send the right file (Apple-Silicon testers → the arm64/`arm64` tarball; Intel/Windows/Linux →
`x86_64`). **A tarball is single-arch** — sending the wrong one fails to run, so label them.

**Tester consumption** (in the package dir, with the tarball beside it):

```bash
gunzip -c eeg-llm-latest-*.tar.gz | docker load     # imports eeg-llm:latest
make setup && make doctor && make run
```

`make load-help` prints these commands.

**Updating:** rebuild, resend the file, tester re-`docker load`.

---

## Which to use

- **Registry** if testers are OK with a one-time `docker login` and you expect to iterate — updates
  are a one-line pull.
- **Tarball** if you want zero accounts and infrequent updates — but you manage arch-matched files by
  hand.

For a handful of mixed-hardware testers who won't update often, tarballs are the least-moving-parts
choice; switch to GHCR once you're iterating.
