# GPT-OSS and Gemma-4 guarded ablations

Run **GPT-OSS-120B × 3 variants × 1,000 fixed cases = 3,000 episodes**.
The runner connects to an existing OpenAI-compatible model endpoint.
Weights and model serving are managed separately.

## Gemma-4-31B

`scripts/run_gemma4_ablations.py` runs the same three variants and the same frozen
1,000 cases. Its Gemma decoding profile is temperature 0, history 4, at most 768
output tokens, strict detective JSON schema, and 300-second request timeout.
Disable thinking on the model server. `--base-urls` accepts multiple replicas;
`--workers` is the total concurrent episode count across all replicas.

For cached Q4_K_M weights and a compatible llama.cpp executable, this launcher
starts one replica on each specified GPU, checks all endpoints, completes a
15-episode pilot, and then runs/resumes the 3,000 episodes:

~~~bash
.venv/bin/python -u scripts/launch_gemma4_ablations.py \
  --llama-server /path/to/llama-server --gguf /path/to/gemma-4-31B-it-Q4_K_M.gguf \
  --gpus 0 2 3 --ports 8332 8333 8334 \
  --output-dir results/gemma4_guarded_ablations
~~~

The launcher verifies the cached checkpoint checksum and writes serving
provenance, per-server logs, launcher status, and the same episode/summary reports
as GPT-OSS. Pilot trajectories are stored separately. Only servers created by
the launcher are stopped after completion. Rerun with the identical settings to
resume; `--pilot-per-level 0` skips the pilot on resume.

**The historical Gemma Full Guarded run used different serialized worlds and a
different guard implementation.** To obtain a matched control for paper
comparisons, add `--include-guarded-reference` to either Gemma command. This adds
1,000 Full Guarded episodes, for 4,000 total and a 20-episode pilot. The historical
Gemma solve rate must not be presented as a paired ablation control for this run.

Offline validation against existing cases:

~~~bash
.venv/bin/python scripts/run_gemma4_ablations.py --validate-only
~~~

## GPT-OSS variants

| Variant | Guard prompt + ledger | Talk redirect | Object redirect | Accusation interception |
| --- | --- | --- | --- | --- |
| full_minus_accuse | Yes | Yes | Yes | None |
| delay_control | Yes | No | No | Delay the first two proposed accusations, regardless of evidence |
| prompt_ledger (P+L) | Yes | No | No | None |

Every variant inherits the original full guarded prompt, history, ledger,
parsing and proposed/executed action tracing. The command runs three new
variants for comparison with the existing full guarded and vanilla baselines.

## Clone and install

~~~bash
git clone --branch experiments/gpt-oss-guarded-ablations --single-branch \
  https://github.com/vuducanh0802/mystery-benchmark-claude-gemini.git \
  mystery-benchmark-gpt-oss-ablations
cd mystery-benchmark-gpt-oss-ablations

python3 -m venv .venv
.venv/bin/pip install -r requirements-experiment.txt
cp .env.example .env
~~~

Python 3.10+ is sufficient for the client. Use a separate environment for vLLM
or another model-serving engine.

## Configure the endpoint

Edit `.env`:

~~~dotenv
GPT_OSS_BASE_URL=http://localhost:8000/v1
GPT_OSS_MODEL=gpt-oss-120b
GPT_OSS_API_KEY=EMPTY
GPT_OSS_EXTRA_BODY='{"reasoning_effort":"low"}'
GPT_OSS_TEMPERATURE=0
GPT_OSS_REVISION=unspecified
~~~

`GPT_OSS_MODEL` must match the server's served name. For a remote authenticated
endpoint, set its URL and key. `provider=openai` identifies the wire protocol.

The released GPT-OSS baseline used checkpoint revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, low reasoning, temperature 0,
JSON-object output, at most **8,192 output tokens**, and **10 observations** of
history. The ablation defaults reproduce those decoding controls. Set
`GPT_OSS_REVISION` to the checkpoint and serving configuration actually used;
the client cannot verify remote weights. Configure the server's reasoning
parser to return final action JSON in `message.content`.

The wrapper requests JSON-object output for every variant. Explicit overrides
`--reasoning-effort` and `--temperature` take precedence over endpoint environment
settings. Additional decoding options in `GPT_OSS_EXTRA_BODY` are recorded;
use the released profile when comparing against the published baseline.

## Validate, preflight, pilot

Restore and check the bundled suite without endpoint calls:

~~~bash
VALIDATE_ONLY=1 bash scripts/run_gpt_oss_ablations.sh
~~~

The bundle contains the same 1,000 case identities and byte-for-byte world files
used by the published GPT-OSS run. Different existing case files are rejected.
Source hashes are also checked by the runner, including when using an explicit
`BENCHMARK_DIR`. A different manifest serialization is acceptable when case
identities and bytes match. The 1,000 nominal entries include **874 byte-distinct
worlds**.

Preflight makes one actual completion; run it yourself once the server is ready:

~~~bash
PREFLIGHT_ONLY=1 bash scripts/run_gpt_oss_ablations.sh
~~~

Pilot all variants on one case per level (15 episodes), in a separate directory:

~~~bash
PER_LEVEL=1 OUTPUT_DIR=results/gpt_oss_ablation_pilot \
  bash scripts/run_gpt_oss_ablations.sh
~~~

## Full run and resume

~~~bash
GPT_OSS_WORKERS=32 bash scripts/run_gpt_oss_ablations.sh
~~~

`GPT_OSS_WORKERS` controls concurrent episodes on the endpoint, not GPU count.
Choose a value the server can handle; the default is 1. The default output is
`results/gpt_oss_guarded_ablations`. Repeating the same command resumes valid
trajectories and retries incomplete episodes. Use one client process per output
directory.

Defaults: all five levels, 200 cases per level, three variants, two per-call
retries, two additional whole-episode rounds, 180-second call timeout.
Set `PYTHON_BIN=/path/to/python` to use an existing client environment.

To run one variant independently:

~~~bash
VARIANTS=full_minus_accuse OUTPUT_DIR=results/gpt_oss_minus_accuse \
  bash scripts/run_gpt_oss_ablations.sh
VARIANTS=delay_control OUTPUT_DIR=results/gpt_oss_delay_control \
  bash scripts/run_gpt_oss_ablations.sh
VARIANTS=prompt_ledger OUTPUT_DIR=results/gpt_oss_prompt_ledger \
  bash scripts/run_gpt_oss_ablations.sh
~~~

Model, endpoint, revision, cases, variant definitions, prompt, decoding, block
limit, and Python source are fingerprinted. Changing them requires a new output
directory. Worker counts and whole-episode retry rounds may change on resume.
A `.env` assignment overrides the same shell variable; keep run-control variables
commented in `.env` when using prefixed command examples.

`MAX_ACCUSE_BLOCKS=2` is the reference setting. The delay control passes accusations
through when budget remaining is at most 5 or its delay limit has been reached,
matching the original guard's escape conditions. It uses the original fallback
investigation action after each delayed accusation and does not inspect evidence
support to decide whether to delay.

## Outputs and comparisons

~~~text
results/gpt_oss_guarded_ablations/
  run_config.json
  validation.json
  summary.csv
  summary.json
  ablation_summary.csv
  episode_metrics.csv
  trajectories/<model_identity>/<variant>/<level>/*.jsonl
~~~

- `ablation_summary.csv`: per-level and overall solve/composite rates using the
  scheduled denominator, plus valid/invalid counts and actions/tokens/interventions.
  Invalid and pending episodes count as zero; partial rates are provisional.
- `episode_metrics.csv`: one row per scheduled case/variant, case/world hash,
  validity, scores, actions, tokens and guard reasons. Pair by `instance_id`;
  cluster duplicate worlds by `source_sha256` for uncertainty estimates.
- `summary.csv`: inherited valid-only metrics by level, with invalid counts.
- `validation.json`: completion is true only when every expected trajectory
  passes header/config/case, terminal, model-response and token checks.
- Trajectories retain the model's proposed action, executed action and intervention
  reason. Provider/parser failures stay invalid.

The inherited action budgets, deterministic NPC fallback, passive culprit and
scoring are unchanged. This branch includes the
`inventory-relocation-missing-room-v1` environment repair used in the published
GPT-OSS recovery: a move of inventory-held evidence is skipped after the original
random draws. Other invalid room identifiers still fail.

Compare **Full vs full_minus_accuse** for the accusation gate's conditional
contribution, **Full vs prompt_ledger** for all action interception, and
**delay_control vs prompt_ledger** for generic delay. Full and delay control also
differ in talk/object redirects; their difference cannot isolate evidence checking
alone. Analyze within a model and use paired cases.

Reference results and configurations:
[Hugging Face release](https://huggingface.co/datasets/Elfsong/Mystery-Benchmark-Full/tree/a4e88a7ca2fd7530e703f76a73dde87488c9a680/run/gpt_oss_120b).
Access to its gated files requires a permitted Hugging Face account. Running this
branch uses the bundled suite and requires no Hugging Face download.

Rebuild reports without model calls:

~~~bash
REPORT_ONLY=1 bash scripts/run_gpt_oss_ablations.sh
~~~

The command returns 0 for complete validation, 1 for incomplete results, and 2 for
configuration errors. Use the original run's settings when rebuilding reports.

## Offline tests

~~~bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q tests/test_gpt_oss_ablations.py \
  tests/test_gpt_oss_kimi_baselines.py tests/test_claude_gemini_baselines.py
~~~

Tests use a synthetic local HTTP endpoint and do not invoke a model.
The original paired runner remains documented in
[GPT-OSS/Kimi baseline instructions](docs/GPT_OSS_KIMI_BASELINES.md).
