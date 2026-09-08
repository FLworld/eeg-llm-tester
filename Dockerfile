# eeg-llm — runtime image for trusted testers.
# Ships ONLY what the app needs to run (see the allowlist COPY below). The development / audit
# layer (ICM docs, TROUBLESHOOTING, HANDOFFs, reproduce/, tests/, scripts/, ica_eeglab.py,
# erpcore_compare.py) is never copied in. The LLM itself runs in Ollama on the HOST — this image
# only holds the Python app + the frozen scientific stack.
#
# Build (multi-arch):
#   docker buildx build --platform linux/amd64,linux/arm64 -t eeg-llm:latest --load .

# ---------- builder: resolve the pinned scientific stack once ---------- #
FROM python:3.11-slim AS builder

# Build deps: chroma-hnswlib compiles on some arches; libgl/glib are matplotlib/mne runtime libs
# that a few wheels probe at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && python -c "import scipy, sys; v=tuple(map(int, scipy.__version__.split('.')[:2])); \
sys.exit('FATAL: scipy pin blown (%s >= 1.14) — sph_harm removed, ICLabel/autoreject would break' % scipy.__version__) if v >= (1,14) else print('scipy pin OK:', scipy.__version__)"

# ---------- runtime: slim image with just the venv + allowlisted app files ---------- #
FROM python:3.11-slim AS runtime

# Runtime shared libs the science stack loads at import (not build).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
    STATE_DIR=/state \
    EEG_DATA_DIR=/data \
    EEG_OUTPUT_DIR=/output \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    XDG_CACHE_HOME=/state/cache \
    MPLCONFIGDIR=/state/cache/matplotlib \
    MNE_DATA=/state/cache/mne

WORKDIR /app

# --- RUNTIME ALLOWLIST: the ONLY app files that ship. Nothing else can enter the image. --- #
COPY app.py pipeline.py engines.py tools.py prompt.py rag.py ingest.py exports.py \
     batch.py qc.py recipes.py sweep.py \
     artifact_break_removal.py artifact_continuous_detect.py artifact_epoch_reject.py \
     Modelfile chainlit.md ./
COPY .chainlit/config.toml ./.chainlit/config.toml
COPY docker/entrypoint.sh docker/smoke.py /usr/local/bin/

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /state /data /output /app/docs

EXPOSE 8001
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
