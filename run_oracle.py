from mystery_world import ComplexityLevel, COMPLEXITY_PRESETS
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing
from mystery_world.world import MysteryEnvironment
from agents.maximum_score_oracle_agent import OracleAgent

for level in [
    ComplexityLevel.TRIVIAL,
    ComplexityLevel.EASY,
    ComplexityLevel.MEDIUM,
    ComplexityLevel.HARD,
    ComplexityLevel.EXPERT,
]:
    config = COMPLEXITY_PRESETS[level]
    state  = generate_mystery(seed=42, config=config)
    env    = MysteryEnvironment(state)
    agent  = OracleAgent()

    result = agent.run(env, render_initial_briefing(env))
    score  = env.get_episode_summary().get("score_result", {})

    elim_correct = score.get("correct_eliminations", 0)
    elim_total   = score.get("total_innocents", 0)

    print(f"=== {level.name} ===")
    print(f"  accusation_correct : {result['accusation_correct']}")
    print(f"  actions_taken      : {result['actions_taken']}")
    print(f"  composite_score    : {score.get('composite_score', 0):.3f}")
    print(f"  triangle_score     : {score.get('triangle_score', 0):.2f}  "
          f"(sw={score.get('suspect_weapon_score', 0):.2f}  "
          f"wv={score.get('weapon_victim_score', 0):.2f}  "
          f"sr={score.get('suspect_room_score', 0):.2f})")
    print(f"  alibi_score        : {score.get('alibi_score', 0):.2f}  "
          f"(cited={score.get('alibi_cited')}  "
          f"found={score.get('contradiction_found')}  "
          f"valid={score.get('contradiction_valid')})")
    print(f"  elimination_score  : {score.get('elimination_score', 0):.2f}  "
          f"({elim_correct}/{elim_total} innocents)")
    print(f"  plan_summary       : {result['plan_summary']}")
    print()
