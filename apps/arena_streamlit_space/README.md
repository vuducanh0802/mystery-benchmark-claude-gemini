---
title: MysteryArena
sdk: docker
app_port: 7860
pinned: false
---

# MysteryArena

Full-stack MysteryArena Space:

- `/` serves the Streamlit viewer for published results.
- `/api/*` proxies to the FastAPI Arena backend for live sessions, jobs, and publishing.
- `/app/start.sh` starts `uvicorn api_app:app` on `127.0.0.1:8000`,
  Streamlit on `127.0.0.1:8501`, and nginx on public port `7860`.

## Configuration

Set the Space variable:

```text
ARENA_DATASET_REPO=org/mystery-arena-results
ARENA_HF_DATASET=org/mystery-arena-results
ARENA_DEFAULT_REVISION=main
ARENA_DATASET_BASE_URL=
ARENA_API_URL=https://<space-subdomain>.hf.space
ARENA_API_PUBLIC_URL=https://<space-subdomain>.hf.space
ARENA_ROOT=/data/arena/results
```

Set Space secrets when backend publishing or registered LLM runs are needed:

```text
HF_TOKEN=...
LLM_GATEWAY_URL=...
LLM_GATEWAY_API_KEY=...
```

The deploy script uses `HF_TOKEN` locally for authentication. It does not sync
secrets into the Space unless `--sync-secrets` is passed explicitly.

`ARENA_DATASET_BASE_URL` is optional. It can point at a local or custom HTTP
root that serves the same `index/` and `runs/` layout.
