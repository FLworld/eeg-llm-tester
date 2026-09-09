# eeg-llm — tester control surface. Run `make help` for the list.
# The LLM runs in Ollama on your HOST; the app runs in Docker and connects to it.
SHELL := /bin/bash
COMPOSE := docker compose
CHAT_MODEL := eeg-qwen
BASE_MODEL := qwen2.5-coder:14b
EMBED_MODEL := nomic-embed-text
DATA_DIR := data-in
SAMPLE_URL ?=

# Publishing (see docker/PUBLISHING.md). Edit REGISTRY to your namespace before pushing.
IMAGE := eeg-llm
TAG ?= latest
REGISTRY ?= ghcr.io/CHANGE-ME
# Prebuilt releases include both Intel/AMD and ARM.
PLATFORMS ?= linux/amd64,linux/arm64
ARCH := $(shell uname -m)

.DEFAULT_GOAL := help
.PHONY: help setup build run up down stop logs doctor smoke ingest fetch-sample \
        login push save load-help

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

setup:  ## One-time: pull the LLM models on the HOST + build eeg-qwen (~18 GB, once)
	@command -v ollama >/dev/null || { echo "✗ Ollama CLI not found — install Ollama first: https://ollama.com"; exit 1; }
	@echo "→ pulling base + embedding models (this is the big, one-time download)…"
	ollama pull $(BASE_MODEL)
	ollama pull $(EMBED_MODEL)
	@echo "→ building $(CHAT_MODEL) from Modelfile…"
	ollama create $(CHAT_MODEL) -f Modelfile
	@echo "✓ setup complete. Next:  make doctor  then  make run"

build:  ## Build the app image locally
	$(COMPOSE) build

run up:  ## Start the app  ->  http://127.0.0.1:8001
	$(COMPOSE) up

down:  ## Stop and remove the container
	$(COMPOSE) down

stop:  ## Stop the container (keep it)
	$(COMPOSE) stop

logs:  ## Tail app logs
	$(COMPOSE) logs -f

doctor:  ## Preflight: Docker, Ollama, models, RAM, mounts, runtime smoke
	./docker/doctor.sh

smoke:  ## Run the in-image runtime smoke test (no Ollama needed)
	$(COMPOSE) run --rm --entrypoint python eeg-llm /usr/local/bin/smoke.py

ingest:  ## Build the RAG knowledge base from PDFs in ./docs
	$(COMPOSE) run --rm --entrypoint python eeg-llm ingest.py

fetch-sample:  ## Put the ERP CORE N170 sub-002 sample + codebook into ./data-in
	@mkdir -p $(DATA_DIR)
	@cp docker/sample/sub-002.codebook.json $(DATA_DIR)/
	@if [ -n "$(SAMPLE_URL)" ]; then \
	  echo "→ downloading sample recording…"; \
	  curl -fL "$(SAMPLE_URL)" -o $(DATA_DIR)/sub-002.set && echo "✓ sample ready:  /scope sub-002.set"; \
	else \
	  echo "✓ codebook placed at $(DATA_DIR)/sub-002.codebook.json"; \
	  echo "  Add the recording: drop the ERP CORE N170 sub-002 file at $(DATA_DIR)/sub-002.set"; \
	  echo "  (or:  make fetch-sample SAMPLE_URL=<direct-download-url>)"; \
	fi

# ---- Publishing (maintainer). Details + tester consumption: docker/PUBLISHING.md ---- #

login:  ## Log in to the registry (expects GHCR_TOKEN + GHCR_USER in the environment)
	@echo "$${GHCR_TOKEN:?set GHCR_TOKEN to a GitHub PAT with write:packages}" \
	  | docker login ghcr.io -u "$${GHCR_USER:?set GHCR_USER to your GitHub username}" --password-stdin

push:  ## Build multi-arch and push to $(REGISTRY)/$(IMAGE):$(TAG)
	@[ "$(REGISTRY)" != "ghcr.io/CHANGE-ME" ] || { echo "✗ edit REGISTRY in the Makefile (or pass REGISTRY=…) first"; exit 1; }
	docker buildx build --platform $(PLATFORMS) -t $(REGISTRY)/$(IMAGE):$(TAG) --push .

save:  ## Save a single-arch ($(ARCH)) image tarball to send offline
	docker build -t $(IMAGE):$(TAG) .
	docker save $(IMAGE):$(TAG) | gzip > $(IMAGE)-$(TAG)-$(ARCH).tar.gz
	@echo "✓ wrote $(IMAGE)-$(TAG)-$(ARCH).tar.gz — send it; the tester runs: make load-help"

load-help:  ## Show the command a tester runs to load a tarball you sent
	@echo "On the tester machine, in the package dir:"
	@echo "  gunzip -c $(IMAGE)-$(TAG)-*.tar.gz | docker load"
	@echo "  make doctor && make run"
