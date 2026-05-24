# MysteryArena Arena

Arena 用来分别评估模型的侦探能力和凶手能力。一次 match 里 detective 和 culprit 可以是不同模型，其他 player NPC 默认使用统一的 fallback responder，也可以显式切到 LLM responder。

## 环境变量

LLM agents 通过 OpenAI-compatible gateway 调用。确认 `.env` 里有：

```bash
LLM_GATEWAY_URL=http://llmbox-global.byteintl.net/v1
LLM_GATEWAY_API_KEY=...
```

`LLM_GATEWAY_URL` 应该是 API root，不要写成 `/chat/completions` endpoint。

## Model List

当前 Arena registry 里用于这轮 ranking 的模型：

```bash
MODELS=deepseek-v4-pro,glm-4.7,glm-5,glm-5.1,gpt-5.2-codex,gpt-5.3-codex,gpt-5.4,gpt-5.4-ptu,gpt-5.5,kimi-k2.5,minimax-m2.5,minimax-m2.7
```

内置 baseline：

```bash
BASELINES=heuristic,oracle_min,oracle_max,passive
```

`heuristic` / `oracle_min` / `oracle_max` 只能做 detective；`passive` 只能做 culprit。

## Unified Run Script

推荐用统一入口 `scripts/run_arena_matches.py` 跑新增对局。它支持 Heuristic baseline、两个模型互换角色对战、自定义 detective x culprit matrix，并且默认每完成一场 match 就上传一次 Hugging Face Dataset。

上传前确认 `.env` 或 shell 里有：

```bash
HF_TOKEN=...
ARENA_HF_DATASET=Elfsong/Mystery_Arena_Results
```

如果涉及 LLM 模型，还需要：

```bash
LLM_GATEWAY_URL=http://llmbox-global.byteintl.net/v1
LLM_GATEWAY_API_KEY=...
```

### Heuristic baseline

`heuristic` detective 对 `passive` culprit：

```bash
uv run python -u scripts/run_arena_matches.py \
  --matchup heuristic \
  --levels TRIVIAL EASY MEDIUM HARD EXPERT \
  --seeds 0-9 \
  --workers 8
```

`heuristic` detective 对多个 culprit 模型：

```bash
uv run python -u scripts/run_arena_matches.py \
  --matchup heuristic \
  --culprits passive,kimi-k2.5,glm-4.7 \
  --levels TRIVIAL EASY \
  --seeds 0-2 \
  --workers 4
```

### Two-way model battle

两个模型互换 detective / culprit 角色。下面命令会跑：

- `kimi-k2.5` detective vs `glm-4.7` culprit
- `glm-4.7` detective vs `kimi-k2.5` culprit

```bash
uv run python -u scripts/run_arena_matches.py \
  --matchup two-way \
  --model-a kimi-k2.5 \
  --model-b glm-4.7 \
  --levels TRIVIAL \
  --seeds 0-2 \
  --workers 2
```

### Custom matrix

自定义所有 detective x culprit 组合：

```bash
MODELS=deepseek-v4-pro,glm-4.7,glm-5,gpt-5.5,kimi-k2.5

uv run python -u scripts/run_arena_matches.py \
  --matchup matrix \
  --detectives heuristic,$MODELS \
  --culprits passive,$MODELS \
  --levels TRIVIAL EASY \
  --seeds 0-4 \
  --workers 8
```

### Common options

- `--workers N`：并行跑 N 场 match。
- `--run-id NAME`：指定 run id；不指定会自动生成。
- `--out DIR`：指定本地输出目录；默认写到 `arena/results/<run_id>`。
- `--no-publish-hf`：只本地运行，不上传。
- `--no-resume`：忽略已有完整 trajectory，强制重跑。
- `--no-preflight`：跳过 LLM 网关可用性检查。
- `--include-model-responses`：上传 trajectory 时包含 raw model response；默认不上传。

每次完成一场 match，脚本会刷新本地：

```text
matches.jsonl
detective_leaderboard.json
culprit_leaderboard.json
role_ratings.json
duel_matrix.json
arena_summary.json
report.md
trajectories/
```

然后把当前已完成的 match 集合发布到 `ARENA_HF_DATASET`。LLM API key 只从本地环境变量读取，不会写入本地结果或上传到 Hugging Face。

## Gateway Check

先跑一个最小验证，确认 URL 和 key 没有 404 或鉴权问题：

```bash
MODELS=gpt-5.4

uv run python scripts/arena_run.py \
  --mode detective \
  --detectives "$MODELS" \
  --culprits passive \
  --levels TRIVIAL \
  --seeds 0 \
  --workers 1 \
  --out arena/results/gateway_check \
  --tui
```

检查轨迹里是否还有 API error：

```bash
rg "API error|404 Not Found" arena/results/gateway_check/trajectories
```

没有输出再跑完整 matrix。

## Full Matrix TrueSkill Run

完整 matrix 会让每个模型都分别作为 detective 和 culprit，与其他模型两两对局。下面命令会跑 13 x 13 x 1 level x 1 seed，共 169 局：

```bash
MODELS=deepseek-v4-pro,glm-4.7,glm-5,glm-5.1,gpt-5.2-codex,gpt-5.3-codex,gpt-5.4,gpt-5.4-ptu,gpt-5.5,kimi-k2,kimi-k2.5,minimax-m2.5,minimax-m2.7

uv run python scripts/arena_run.py \
  --mode matrix \
  --detectives "$MODELS" \
  --culprits "$MODELS" \
  --levels TRIVIAL \
  --seeds 0 \
  --workers 16 \
  --out arena/results/model_elo_smoke \
  --tui
```

默认 job 调度是 `--schedule balanced`，会把 detective/culprit pair 按 diagonal 顺序打散，避免一开始连续运行同一个 detective model 对所有 culprit models。旧的 detective-major 顺序可以用 `--schedule row-major` 恢复。

默认也会开启断点续跑。重新执行同一个 `--out` 目录时，runner 会先读取已有的完整 trajectory，把它们计入 TUI、leaderboard 和 TrueSkill rating，只继续跑缺失或未完成的局。只有写入了 footer 的 trajectory 才算完成；中断留下的半截 trajectory 会重新跑。

Ranking 使用 mean payoff。展示里的 `Skill` 是辅助的 role-specific TrueSkill conservative score `mu - 3 * sigma`，`Sigma` 越高表示不确定性越高。单局会先把 detective payoff 按 0.5 分界转成 TrueSkill 胜负：`payoff > 0.5` 为 detective win，`payoff < 0.5` 为 culprit win。

如果要强制重跑同一个输出目录：

```bash
uv run python scripts/arena_run.py \
  --mode matrix \
  --detectives "$MODELS" \
  --culprits "$MODELS" \
  --levels TRIVIAL \
  --seeds 0 \
  --workers 16 \
  --out arena/results/model_elo_smoke \
  --no-resume \
  --tui
```

更稳定的 ranking 建议加 seeds：

```bash
uv run python scripts/arena_run.py \
  --mode matrix \
  --detectives "$MODELS" \
  --culprits "$MODELS" \
  --levels TRIVIAL EASY \
  --seeds 0-4 \
  --workers 8 \
  --out arena/results/model_elo_run_001 \
  --tui
```

如果修过 gateway URL 或 prompt，不要复用旧输出目录；换一个新的 `--out`，避免旧 trajectory 污染结果。

## Role-Specific Runs

只评估 detective 能力，用 `passive` culprit：

```bash
uv run python scripts/arena_run.py \
  --mode detective \
  --detectives "$MODELS" \
  --culprits passive \
  --levels TRIVIAL EASY \
  --seeds 0-4 \
  --workers 8 \
  --out arena/results/detective_eval_001 \
  --tui
```

只评估 culprit 能力，用统一 detective baseline：

```bash
uv run python scripts/arena_run.py \
  --mode culprit \
  --detectives heuristic \
  --culprits "$MODELS" \
  --levels TRIVIAL EASY \
  --seeds 0-4 \
  --workers 8 \
  --out arena/results/culprit_eval_001 \
  --tui
```

## Dashboard

Arena 现在可以拆成两个进程：

- Back-end API：读取结果、托管 server-side LLM gateway key、提供交互式对局 API。
- Front-end visualization：只负责展示，通过 HTTP API 取 leaderboard / matrix / replay 数据。

启动后端：

```bash
uv run python scripts/arena_api.py \
  --arena-root arena/results \
  --port 8000
```

后端会从 `.env` 读取 `LLM_GATEWAY_URL` / `LLM_GATEWAY_API_KEY`，API response 只返回 `key_env`，不会返回 key value。

主要接口：

```text
GET  /api/health
GET  /api/models
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/episodes
GET  /api/runs/{run_id}/episodes/{episode_id}/trajectory
GET  /api/arena/jobs
GET  /api/arena/jobs/{job_id}
POST /api/arena/jobs/{job_id}/cancel
POST /api/arena/runs
POST /api/arena/matches
POST /api/arena/runs/{run_id}/publish-hf
POST /api/sessions
GET  /api/sessions/{session_id}
POST /api/sessions/{session_id}/actions
```

交互式对局示例：用户通过 API 扮演侦探，对局 Arena 上的 passive culprit。

```bash
curl -s http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "player_role": "detective",
    "detective": "human",
    "culprit": "passive",
    "level": "TRIVIAL",
    "seed": 0
  }'
```

之后把返回的 `session_id` 填到 action endpoint：

```bash
curl -s http://127.0.0.1:8000/api/sessions/<session_id>/actions \
  -H 'Content-Type: application/json' \
  -d '{"action": "EXAMINE_LOCATION", "action_args": {}}'
```

如果想让真人/API 扮演凶手、和 Arena 上的侦探模型对比：

```bash
curl -s http://127.0.0.1:8000/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "player_role": "culprit",
    "detective": "heuristic",
    "culprit": "human",
    "level": "TRIVIAL",
    "seed": 0
  }'
```

启动 API-backed 可视化 dashboard：

```bash
uv run python scripts/arena_dashboard.py \
  --api-url http://127.0.0.1:8000 \
  --run-id model_elo_run_001
```

也可以继续用旧的本地文件模式启动 dashboard：

```bash
uv run python scripts/arena_dashboard.py \
  --arena-dir arena/results/model_elo_run_001
```

如果默认端口被占用，可以指定端口：

```bash
uv run python scripts/arena_dashboard.py \
  --arena-dir arena/results/model_elo_run_001 \
  --port 7861
```

Dashboard 包含：

- Detective Leaderboard
- Culprit Leaderboard
- Role Gap
- Duel Matrix
- Episode Replay

## API-triggered Runs

后端也可以异步触发 Arena run。MVP 版本会在 API 进程内注册一个后台 job，并复用 `scripts/arena_run.py` 子进程完成对局。

单局对局：

```bash
curl -s http://127.0.0.1:8000/api/arena/matches \
  -H 'Content-Type: application/json' \
  -d '{
    "detective": "heuristic",
    "culprit": "passive",
    "level": "TRIVIAL",
    "seed": 0,
    "run_id": "api_match_smoke"
  }'
```

批量 run：

```bash
curl -s http://127.0.0.1:8000/api/arena/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "matrix",
    "detectives": "heuristic",
    "culprits": "passive",
    "levels": ["TRIVIAL"],
    "seeds": "0-2",
    "workers": 1,
    "run_id": "api_matrix_smoke"
  }'
```

查询 job：

```bash
curl -s http://127.0.0.1:8000/api/arena/jobs/<job_id>
```

## Hugging Face Publishing

发布前配置：

```bash
export HF_TOKEN=...
export ARENA_HF_DATASET=org/mystery-arena-results
```

本地打包但不上传：

```bash
uv run python scripts/arena_publish.py \
  arena/results/api_matrix_smoke \
  --package-dir /tmp/mystery_arena_hf_package \
  --no-upload
```

上传到 Hugging Face Dataset：

```bash
uv run python scripts/arena_publish.py \
  arena/results/api_matrix_smoke \
  --repo-id "$ARENA_HF_DATASET"
```

也可以通过 API 发布：

```bash
curl -s http://127.0.0.1:8000/api/arena/runs/api_matrix_smoke/publish-hf \
  -H 'Content-Type: application/json' \
  -d '{"repo_id": "org/mystery-arena-results"}'
```

发布产物结构：

```text
index/runs.json
runs/<run_id>/summary.json
runs/<run_id>/matches.jsonl.gz
runs/<run_id>/trajectories/<match_id>.jsonl.gz
README.md
```

`summary.json`、`matches.jsonl.gz` 和 trajectory 都会移除本地 `trajectory_path`，并清洗 gateway URL、key env、token、secret 等敏感字段。默认保留 `model_response` 用于 replay；如需隐藏 raw response，可使用 `--no-model-responses` 或 API 里的 `include_model_responses=false`。

## Streamlit Space Frontend

只读前端在：

```text
apps/arena_streamlit_space/
```

部署为 Hugging Face Streamlit Space，并设置：

```text
ARENA_DATASET_REPO=org/mystery-arena-results
ARENA_DEFAULT_REVISION=main
ARENA_DEFAULT_RUN=latest
```

前端直接读取 Hugging Face Dataset 的公开文件，不需要后端在线，也不需要 LLM gateway key 或 HF 写 token。

一键创建/更新 Space：

```bash
export HF_TOKEN=...
export ARENA_HF_DATASET=org/mystery-arena-results

uv run python scripts/arena_deploy_space.py \
  org/mystery-arena \
  --dataset-repo "$ARENA_HF_DATASET"
```

脚本会创建 `sdk=streamlit` 的 Space、上传 `apps/arena_streamlit_space/`，并设置 `ARENA_DATASET_REPO`、`ARENA_DEFAULT_REVISION`、`ARENA_DEFAULT_RUN` 三个 Space variable。

## End-to-end Verification

基础回归：通过 FastAPI 触发 `heuristic vs passive` 单局，生成本地结果，打包成 HF Dataset layout，并检查敏感字段不会进入公开产物。

```bash
uv run python scripts/verify_arena_system.py
```

完整回归：额外把打包目录用本地 HTTP server 暴露给 Streamlit `AppTest`，验证 Overview、Leaderboards、Duel Matrix、Episode Replay 都能从 Dataset layout 加载。

```bash
uv run \
  --with streamlit \
  --with pandas \
  --with plotly \
  --with requests \
  python scripts/verify_arena_system.py --frontend
```

## TUI Progress

`scripts/arena_run.py --tui` 会显示两层进度：

- `episodes`: 已完成的 episode 数。
- `actor steps`: episode 内部的 actor 动作预算进度。单个 detective 的预算仍然是 complexity config 里的 `max_agent_actions`；如果 culprit 是自由行动 agent，总 actor 预算会按 detective + culprit 两个 actor 估算。

`Active Episodes` 表会显示当前 worker 正在跑的 episode、detective steps / detective budget、culprit steps / culprit budget，以及最近一次动作。

## Outputs

每个 Arena run 会写入：

```text
config.json
roster.json
matches.jsonl
detective_leaderboard.json
culprit_leaderboard.json
role_ratings.json
duel_matrix.json
trajectories/
```

`detective_leaderboard.json` 和 `culprit_leaderboard.json` 是按 mean payoff 排序的分角色 ranking，包含 `trueskill.mu`、`trueskill.sigma` 和 `trueskill.skill` 作为辅助指标；`duel_matrix.json` 用于查看具体 detective vs culprit 的对局表现。
