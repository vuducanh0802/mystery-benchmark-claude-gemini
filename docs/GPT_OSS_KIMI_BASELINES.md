# GPT-OSS / Kimi: Vanilla vs. Guarded Evaluation

This branch evaluates GPT-OSS and Kimi detective agents using the same protocol
as the Claude/Gemini experiment. It supports OpenAI-compatible Chat Completions
endpoints provided by vLLM, SGLang, or a hosted inference service.

The repository contains the evaluation runner and a frozen benchmark suite.
Model weights and inference servers are managed separately.

Default matrix: **GPT-OSS-120B + Kimi-K2.5**, each with **vanilla + guarded**, on
1,000 frozen case entries (200 per difficulty): **4,000 episodes**.
GPT-OSS-20B and original Kimi-K2 are configurable below.

## 1. Clone and install

```bash
git clone --branch experiments/gpt-oss-kimi-vanilla-guarded --single-branch \
  https://github.com/vuducanh0802/mystery-benchmark-claude-gemini.git \
  mystery-benchmark-gpt-oss-kimi
cd mystery-benchmark-gpt-oss-kimi

python3 -m venv .venv
.venv/bin/pip install -r requirements-experiment.txt
cp .env.example .env
```

Python 3.10 or newer is required. The full application has a separate Python 3.13
environment in `pyproject.toml`; it is not required for this experiment runner.
Keep model-serving dependencies such as vLLM in a separate environment.

## 2. Configure model endpoints

Edit `.env` with the endpoint settings used for the experiment:

```dotenv
GPT_OSS_BASE_URL=http://localhost:8000/v1
GPT_OSS_MODEL=openai/gpt-oss-120b
GPT_OSS_API_KEY=EMPTY
GPT_OSS_REVISION=checkpoint-quantization-engine-version
GPT_OSS_EXTRA_BODY='{"reasoning_effort":"medium"}'
GPT_OSS_TEMPERATURE=

KIMI_BASE_URL=http://localhost:8001/v1
KIMI_MODEL=moonshotai/Kimi-K2.5
KIMI_API_KEY=EMPTY
KIMI_REVISION=checkpoint-quantization-engine-version
KIMI_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false},"top_p":0.95}'
KIMI_TEMPERATURE=0.6
```

- `MODEL` must match the endpoint's served name. A provider may use a different
  name from the Hugging Face repository.
- Replace localhost with the server hostname when using a remote server.
- Use the real provider key for authenticated endpoints; use `EMPTY` explicitly
  for an unauthenticated server. Keys are read from `.env`, not CLI arguments.
- `provider=openai` in reports means **wire protocol**, not that OpenAI serves
  GPT-OSS or Kimi. The endpoint and exact model are recorded in `run_config.json`.
- Set `*_REVISION` to record checkpoint revision, quantization, inference engine
  version, and server context limit. The client cannot verify remote weights.
- Reasoning/decoding options are backend-specific. Unsupported options fail
  loudly; they are never silently dropped. Preflight before a long run.

### Other model/backend choices

**GPT-OSS-20B:** set `GPT_OSS_MODEL=openai/gpt-oss-20b` (or its served alias).

**Kimi-K2 original, non-thinking:** set the served model name, for example
`moonshotai/Kimi-K2-Instruct`, set `KIMI_EXTRA_BODY='{}'`, and choose the decoding
settings for that checkpoint. Do not apply K2.5-specific options automatically.

**Kimi-K2.5 Thinking on a compatible local server:** use
`KIMI_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":true},"top_p":0.95}'`
and `KIMI_TEMPERATURE=1.0`.

**Moonshot hosted K2.5:** use the account API base URL, `KIMI_MODEL=kimi-k2.5`,
and `KIMI_API_KEY=<api-key>`. For Instant mode, use
`KIMI_EXTRA_BODY='{"thinking":{"type":"disabled"},"top_p":0.95}'` and
`KIMI_TEMPERATURE=0.6`; for Thinking use `type=enabled` and temperature `1.0`.

The server must return final JSON in `choices[0].message.content` and nonzero
`usage.prompt_tokens` / `usage.completion_tokens`. Reasoning-only output is not
an action. Configure the server's reasoning parser for the selected model. Increase
`MAX_TOKENS` if thinking consumes the completion budget; context limits must
accommodate the prompt **plus** this allowance. Defaults are 16,384 output tokens
and a 10-observation history for both policies.

References for serving (commands depend on the hardware and engine version):
[GPT-OSS vLLM recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md),
[Kimi-K2.5 model card](https://huggingface.co/moonshotai/Kimi-K2.5),
[vLLM reasoning output configuration](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

## 3. Validate without model calls

```bash
VALIDATE_ONLY=1 bash scripts/run_gpt_oss_kimi_baselines.sh
```

The launcher restores the bundled, checksummed **Claude/Gemini suite** and checks
all selected serialized worlds. It never regenerates cases. Existing different
files are rejected rather than overwritten. All four cells share these cases.

`benchmark_suites/claude_gemini_cases_v1.json` contains the source information and
per-file hashes. This is the historical 1,000-entry suite, **not a deduplicated or
new held-out test set**. It is not assumed identical to another checkout's suite.
For another explicitly chosen dataset, set `BENCHMARK_DIR=/path/to/suite` and
`PREPARE_BENCHMARK=0`; the runner fingerprints its actual manifest and instances.

## 4. Preflight, then a small pilot

Preflight makes **one real completion per selected model**, so a hosted API may
charge for it. No investigation episodes are run in this mode:

```bash
PREFLIGHT_ONLY=1 bash scripts/run_gpt_oss_kimi_baselines.sh
```

Run one case per level for each model/policy (20 episodes):

```bash
PER_LEVEL=1 OUTPUT_DIR=results/pilot \
  bash scripts/run_gpt_oss_kimi_baselines.sh
```

Run just GPT-OSS first if only that server is ready:

```bash
MODELS=gpt-oss PER_LEVEL=1 OUTPUT_DIR=results/gpt_oss_pilot \
  bash scripts/run_gpt_oss_kimi_baselines.sh
```

## 5. Full experiment and resume

```bash
bash scripts/run_gpt_oss_kimi_baselines.sh
```

Or run the models sequentially, on separate servers/machines, with distinct outputs:

```bash
MODELS=gpt-oss OUTPUT_DIR=results/gpt_oss_full \
  bash scripts/run_gpt_oss_kimi_baselines.sh
MODELS=kimi OUTPUT_DIR=results/kimi_full \
  bash scripts/run_gpt_oss_kimi_baselines.sh
```

Defaults are one concurrent episode per endpoint. Set `GPT_OSS_WORKERS=2` and/or
`KIMI_WORKERS=2` only when the servers have sufficient capacity. The same command
and configuration resume completed runs; incomplete/error trajectories are retried. Do not launch
two client processes into the same output directory.

Changing model, endpoint, decoding, selected cases/policies, or relevant Python
source requires a **new OUTPUT_DIR**. Worker count and whole-episode retry rounds
can change without invalidating already completed trajectories. Environment
assignments inside `.env` override prefixed shell variables; keep run controls
commented there if using the examples above.

## 6. Outputs

```text
results/gpt_oss_kimi_vanilla_guarded/
  run_config.json
  summary.csv
  summary.json
  validation.json
  trajectories/<model_identity>/<policy>/<level>/*.jsonl
```

`validation.json` is complete only when every expected episode has a valid header,
footer, matching case/config hashes, model output, and nonzero token usage.
`summary.csv` reports completed valid episode scores and invalid/missing counts.
Its inherited `attempted_n` is the **scheduled cell size**, including jobs that
were never started after a fatal provider error; it is not an API-call count.
A failed accusation is a valid task outcome; an API/parser failure is not.

`REPORT_ONLY=1` rebuilds reports without calls; use the same model/settings and
selection as the run. Partial runs return exit code 1. Configuration errors return 2.

To share results, archive only the experiment directory:

```bash
tar -czf mystery-gpt-oss-kimi-results.tar.gz \
  -C results gpt_oss_kimi_vanilla_guarded
```

Archive the model-server configuration and version separately. Do not include `.env`.

## Protocol and tests

The world, scoring, vanilla prompt, and guarded logic are inherited from the
Claude/Gemini branch. Culprit is passive and interviews use the fixed deterministic
fallback. Guarded traces retain proposed actions, executed actions, and intervention
reasons. This branch does not claim to improve or redesign the guard.

Offline tests use a fake HTTP endpoint, **not real model inference**:

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q tests/test_gpt_oss_kimi_baselines.py \
  tests/test_claude_gemini_baselines.py
```
