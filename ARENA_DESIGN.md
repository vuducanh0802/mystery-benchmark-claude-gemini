# MysteryArena Arena Design

## 目标

Arena 要回答两个独立问题：

1. 一个模型做侦探时，能不能查清真相并用证据证明。
2. 一个模型做凶手时，能不能在不破坏可解性的前提下干扰侦探。

这两个能力必须分开排名。一个模型可能很会破案，但不擅长扮演凶手；也可能反过来。

其它 NPC 不进入排名。它们使用同一个固定 NPC 配置，作为环境条件，而不是参赛选手。

## 基本对局

一局 Arena 对局由三部分组成：

- Detective model: 当前被评估的侦探。
- Culprit model: 当前被评估或固定的凶手。
- NPC model: 全局固定，用于所有普通 NPC 的对话。

侦探和凶手共享同一个世界状态。侦探通过 `env.step(...)` 行动；凶手通过 `env.step_for_actor(culprit_id, ...)` 行动。普通 NPC 只在被询问时由统一 NPC responder 回答。

每一步后都保持 solvability guard：模型可以让案件变难，但不能让案件变成无解。

## 三条赛道

### 1. Detective Leaderboard

问题：哪个模型最会当侦探？

固定项：

- 凶手：passive baseline，或一个固定 reference culprit。
- NPC：统一固定模型。
- cases：所有侦探跑同一批 seed/level。

主分数：

```text
detective_payoff = composite_score
```

辅助指标：

- solve rate
- accusation accuracy
- triangle / alibi / elimination subscore
- average actions
- token cost
- failed action rate
- solvability guard blocked action rate

这个 leaderboard 不应该混入凶手能力。它只衡量“在同样环境下谁更会破案”。

### 2. Culprit Leaderboard

问题：哪个模型最会当凶手？

固定项：

- 侦探：一个或多个 reference detective。
- NPC：同一个固定 NPC 模型。
- cases：所有凶手跑同一批 seed/level。

主分数：

```text
culprit_payoff = 1 - culprit_exposure

culprit_exposure =
  0.70 * correct_suspect
  + 0.15 * correct_weapon
  + 0.15 * correct_room
```

辅助指标：

- detective failure rate: 错误指控或超时
- score drop vs passive culprit baseline
- detective actions before accusation
- culprit failed action rate
- culprit guard blocked action rate
- evidence interference count

注意：guard-blocked action 不应该作为正向奖励。否则模型会学会反复尝试非法破坏。它只作为诊断项。

`culprit_payoff` 不使用 `1 - detective_payoff`。侦探主分 `detective_payoff`
是完整 composite score，包含证据引用、alibi、elimination、效率等维度；凶手的主目标是避免在最终指控中暴露。因此侦探抓对人但证据引用不完整时，侦探 composite 可以偏低，但凶手不应获得高 payoff。

### 3. Cross-Role Matrix

问题：模型之间的攻防关系是什么？

令 `D_i` 是侦探模型，`C_j` 是凶手模型。跑完整矩阵：

```text
matrix[i, j] = mean detective_payoff(D_i vs C_j)
culprit_matrix[i, j] = 1 - matrix[i, j]
```

这个视图用于看：

- 哪些侦探对所有凶手都稳。
- 哪些凶手能普遍压低侦探分数。
- 某些模型组合是否存在特殊克制关系。
- 同一个模型做侦探和做凶手的差异。

## 排名方法

主排名用 mean payoff，同时保留 solve/failure rate 和 role-specific TrueSkill conservative score 作为辅助诊断指标。

### Mean Payoff

侦探：

```text
DetectiveMean(model) = mean(detective_payoff)
```

凶手：

```text
CulpritMean(model) = mean(culprit_payoff)
```

CI 用 case-level bootstrap。重采样单位是 episode/case，不是 step。

### Role TrueSkill

维护两个 rating pool：

- `TS_detective[model]`
- `TS_culprit[model]`

每一局 `D` vs `C` 先把连续 payoff 转成 TrueSkill 的序关系：

```text
d = detective_payoff
c = culprit_payoff
d > c  => detective win
d < c  => culprit win
d == c => draw / no-op
```

建议：

- mu: 25
- sigma: 25 / 3
- beta: 25 / 6
- tau: 25 / 300
- 每个 episode 更新一次
- dashboard 按 mean payoff 排序，同时显示 Skill、Mu、Sigma

`Skill = mu - 3 * sigma`，作为辅助的 pairwise 保守能力分。`Sigma` 越高说明样本不足或不确定性更高。

## NPC 策略

NPC 是控制变量。

官方 Arena run 应该在 config 里明确写：

```json
{
  "npc_provider": "fallback | openai | openrouter | vllm",
  "npc_model": "...",
  "npc_seed": 42,
  "npc_prompt_policy": "role_facts_only_no_strategy"
}
```

原则：

- 同一个 Arena run 内所有模型面对同一个 NPC 配置。
- NPC prompt 不指导它帮助侦探或帮助凶手。
- NPC 不进入 TrueSkill，不进入 leaderboard。
- 如果想研究 NPC 条件影响，应作为 separate condition，不和主榜混在一起。

## 数据结构

建议结果目录：

```text
arena/results/<run_id>/
  config.json
  roster.json
  matches.jsonl
  detective_leaderboard.json
  culprit_leaderboard.json
  role_ratings.json
  duel_matrix.json
  trajectories/
    <detective_name>/
      <culprit_name>/
        <level>/
          seed_<seed>.jsonl
```

### `config.json`

记录本次 Arena 的控制变量：

```json
{
  "run_id": "arena_001",
  "mode": "detective | culprit | matrix",
  "levels": ["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"],
  "seeds": [0, 1, 2],
  "npc": {},
  "detectives": [],
  "culprits": [],
  "rating": {
    "system": "trueskill",
    "trueskill_mu": 25.0,
    "trueskill_sigma": 8.3333333333,
    "trueskill_beta": 4.1666666667,
    "trueskill_tau": 0.0833333333,
    "bootstrap_samples": 10000
  }
}
```

### `matches.jsonl`

一行一局：

```json
{
  "run_id": "arena_001",
  "level": "HARD",
  "seed": 83,
  "detective": {"name": "model_a", "provider": "openai", "model": "..."},
  "culprit": {"name": "model_b", "provider": "anthropic", "model": "..."},
  "npc": {"provider": "vllm", "model": "...", "seed": 42},
  "payoff_schema": "detective_composite_v1_culprit_exposure_v1",
  "detective_payoff": 0.72,
  "culprit_payoff": 0.0,
  "solved": true,
  "accusation_correct": true,
  "score_result": {
    "composite_score": 0.72,
    "triangle_score": 2.4,
    "alibi_score": 1.0,
    "elimination_score": 0.5
  },
  "actions_taken": 21,
  "culprit_actions_taken": 21,
  "detective_failed_actions": 1,
  "culprit_failed_actions": 2,
  "guard_blocked_actions": 1,
  "trajectory_path": "trajectories/model_a/model_b/HARD/seed_83.jsonl"
}
```

### Trajectory Metadata

现有 JSONL trajectory 已经有 `actor_id` 和 `role`。Arena 需要再补充：

- `culprit_agent`
- `culprit_provider`
- `culprit_model`
- `npc_provider`
- `npc_model`
- `npc_seed`
- `arena_run_id`
- `arena_match_id`

Episode Replay 直接读 trajectory，不重新跑模型。

## 可视化界面

用 Gradio 做 dashboard。它已经是项目依赖。图表可以先用 Gradio `Dataframe` + 简单 HTML/SVG；后续如果需要更强交互，再加 Plotly。

### Tab 1: Overview

展示：

- run_id
- mode
- completed / failed / total
- case suite
- NPC config
- model roster
- 当前最佳侦探
- 当前最佳凶手

### Tab 2: Detective Leaderboard

表格列：

- rank
- model
- mean detective payoff
- 95% CI
- detective Skill / Mu / Sigma
- solve rate
- accusation accuracy
- triangle score
- alibi score
- elimination score
- avg actions
- token cost
- failed action rate

图：

- payoff by complexity
- score decomposition
- score vs token cost

### Tab 3: Culprit Leaderboard

表格列：

- rank
- model
- mean culprit payoff
- 95% CI
- culprit Skill / Mu / Sigma
- induced detective failure rate
- score drop vs passive baseline
- avg detective actions before accusation
- culprit failed action rate
- guard blocked rate

图：

- culprit payoff by complexity
- score drop by reference detective
- blocked-action diagnostics

### Tab 4: Role Gap

用于回答“同一个模型做侦探和做凶手差异多大”。

视图：

- scatter: x = detective mean/Skill, y = culprit mean/Skill
- diagonal reference line
- table: `role_gap = z(detective_mean) - z(culprit_mean)`

解释：

- 右上：两种角色都强。
- 右下：强侦探，弱凶手。
- 左上：弱侦探，强凶手。
- 左下：两种角色都弱。

### Tab 5: Duel Matrix

热力图：

- rows: detective models
- columns: culprit models
- cell: mean detective payoff

可切换：

- detective payoff / culprit payoff
- level filter
- NPC condition filter
- show confidence interval
- show episode count

点击 cell 后列出该 matchup 的所有 episodes。

### Tab 6: Episode Replay

核心调试页面。

组件：

- episode selector
- step slider
- actor timeline: detective / culprit interleaved
- current actor observation
- raw model response
- parsed action
- result observation
- success/failure
- world hash
- final score breakdown

需要特别标记：

- culprit action
- failed action
- solvability guard block
- accusation step
- discovered evidence ids

### Tab 7: Run Monitor

只做监控，不在 v1 里承担长任务调度。

展示：

- matchup progress grid
- latest completed episodes
- latest errors
- resume command
- trajectory path links

## CLI

### `scripts/arena_run.py`

```bash
uv run python scripts/arena_run.py \
  --mode matrix \
  --detectives gpt4,claude,gemini \
  --culprits gpt4,claude,gemini \
  --npc-provider vllm \
  --npc-model Qwen/Qwen2.5-27B-Instruct \
  --npc-seed 42 \
  --levels TRIVIAL,EASY,MEDIUM,HARD,EXPERT \
  --seeds 0-49 \
  --out arena/results/run_001 \
  --skip-existing
```

`arena_run.py` 默认在交互式终端显示 live progress TUI，包括总进度、ETA、实时侦探/凶手均分榜和最近 episode。非 TTY 输出会自动退回逐行日志；也可以显式使用 `--no-tui` 或 `--tui`。

Modes:

- `detective`: rank detectives under fixed culprit condition.
- `culprit`: rank culprits under fixed detective condition.
- `matrix`: full detective x culprit matrix.

### `scripts/arena_dashboard.py`

```bash
uv run python scripts/arena_dashboard.py \
  --arena-dir arena/results/run_001
```

## 实现顺序

### P0: Arena 数据层

- 新建 `arena/roster.py`
- 新建 `arena/metrics.py`
- 新建 `arena/trueskill.py`
- 新建 `arena/aggregate.py`
- 扩展 trajectory header，加入 culprit 和 arena metadata。

### P1: Runner

- 新建 `scripts/arena_run.py`
- 支持 detective / culprit / matrix 三种 mode。
- 输出 `matches.jsonl`。
- 支持 `--skip-existing`。

### P2: Dashboard

- 新建 `scripts/arena_dashboard.py`
- 实现 Overview、两个 leaderboard、Role Gap、Duel Matrix、Episode Replay。
- 先读静态结果文件，不负责启动 batch run。

### P3: Baselines

- 加 passive culprit baseline。
- 加 reference detective set。
- 加 passive-vs-free culprit score drop。

### P4: Report Export

- 导出 CSV。
- 导出静态 HTML summary。
- 固化 official suite 配置。

## 默认 official suite

建议第一版：

- levels: all five complexity levels
- seeds: 10 per level, total 50 cases
- NPC: deterministic fallback for main leaderboard
- optional NPC robustness condition: one fixed LLM NPC
- detective references for culprit track:
  - heuristic
  - oracle_min
  - one fixed LLM detective
- culprit baseline:
  - passive culprit
  - free LLM culprit candidates

最终报告同时展示：

- Detective Leaderboard
- Culprit Leaderboard
- Role Gap scatter
- Duel Matrix
- representative episode replays
