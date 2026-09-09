# Reproducibility container for the free, offline path (baselines + retrieval
# quality + selective-prediction metrics) -- the same numbers `make baselines`
# produces locally, in an environment nobody has to configure by hand.
#
# This deliberately does NOT try to containerize the LLM-agent path: that
# needs a real API key at run time (never at build time -- see .dockerignore),
# and the point of a base image is a numbers-you-can-trust smoke test that
# works for anyone, with or without a key. Run the LLM agent the same way
# whether you're in the container or not: pass GEMINI_API_KEY/ANTHROPIC_API_KEY
# at `docker run` time (see the bottom of this file).
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so this layer only rebuilds when requirements.txt changes,
# not on every source edit.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Bake the retrieval index at build time. This is a pure cache hit against the
# COMMITTED embedding cache (data/interim/emb_cache/*.npz) -- no network, no
# Ollama, a few seconds -- so every `docker run` of the resulting image starts
# instantly instead of re-doing this deterministic step on every container start.
RUN python -m support_agent.retrieval.index \
      --cases data/interim/hulu_support_cases.jsonl \
      --out-prefix artifacts/precedent_index

# Default: the free, offline path. No API key, no network, ~1 minute.
CMD ["python", "scripts/run_baselines.py"]

# --- other targets (all documented in README.md / Makefile) ---
# Tests:            docker run --rm <image> python -m pytest tests/ -v
# LLM agent (real):  docker run --rm --env-file .env -e LLM_BACKEND=gemini <image> \
#                      python scripts/run_llm_agent.py
# LLM agent (cached, needs llm_cache/ committed): docker run --rm <image> \
#                      env LLM_BACKEND=cached python scripts/run_llm_agent.py
