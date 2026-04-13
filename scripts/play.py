"""
Interactive human-player mode for MysteryArena.

Usage:
    uv run scripts/play.py                          # random MEDIUM case
    uv run scripts/play.py --level EASY --seed 7    # specific difficulty + seed
    uv run scripts/play.py --load path/to/world.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.narrator import render_initial_briefing, render_step_observation
from mystery_world.world import AgentAction, MysteryEnvironment, WorldState


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 70
THICK   = "═" * 70

def _print_box(text: str) -> None:
    print(f"\n{THICK}")
    print(text)
    print(THICK)

def _print_result(text: str) -> None:
    print(f"\n{DIVIDER}")
    print(text)
    print(DIVIDER)

HELP_TEXT = """
COMMANDS
  look                          — examine your surroundings
  go <location>                 — move to an adjacent room
  examine <object>              — inspect an object closely
  search                        — thorough search of the room (may reveal hidden clues)
  talk <name>                   — start / continue an interview (you will be prompted for a question)
  take <object>                 — pick up a portable object
  inventory                     — review evidence you have collected
  wait                          — let time pass (costs one action)
  accuse                        — make your final accusation (you will be prompted)
  map                           — show the estate map and your current position
  suspects                      — list all suspects
  help                          — show this message
  quit                          — exit without finishing
"""

LEVEL_NAMES = {lvl.name: lvl for lvl in ComplexityLevel}


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

def _parse_command(raw: str) -> tuple[AgentAction, dict] | None:
    """
    Convert a natural-language command into (AgentAction, kwargs).
    Returns None for special commands (help, map, quit, suspects) handled
    outside the env.step() loop.
    """
    tokens = raw.strip().split()
    if not tokens:
        return None
    verb = tokens[0].lower()
    rest = " ".join(tokens[1:])

    if verb in ("look", "l", "examine") and (not rest or rest in ("room", "around", "location")):
        return AgentAction.EXAMINE_LOCATION, {}

    if verb in ("go", "move", "walk", "run"):
        return AgentAction.MOVE, {"target_location": rest}

    if verb in ("examine", "inspect", "x") and rest:
        return AgentAction.EXAMINE_OBJECT, {"object_name": rest}

    if verb in ("search", "s"):
        return AgentAction.SEARCH_FOR_EVIDENCE, {}

    if verb in ("wait", "w"):
        return AgentAction.WAIT, {}

    if verb in ("inventory", "inv", "i"):
        return AgentAction.CHECK_INVENTORY, {}

    if verb in ("take", "grab", "pick"):
        obj = rest.removeprefix("up ").strip()
        return AgentAction.TAKE_OBJECT, {"object_name": obj}

    # "talk <name>" or "ask <name>"
    if verb in ("talk", "ask", "interview", "question"):
        return AgentAction.TALK_TO, {"character_name": rest, "_needs_question": True}

    # "accuse" — handled interactively in the main loop
    if verb in ("accuse", "arrest", "charge"):
        return AgentAction.ACCUSE, {"_interactive": True}

    return None   # unknown / handled by caller


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def play(env: MysteryEnvironment) -> None:
    state = env.state

    # Build quick-reference data
    suspect_list = [
        c for c in state.characters.values()
        if any(r.name == "SUSPECT" for r in c.roles) and c.is_alive
    ]
    location_map: dict[str, list[str]] = {
        loc.name: [state.locations[a].name for a in loc.adjacent_ids if a in state.locations]
        for loc in state.locations.values()
    }

    def _show_map() -> None:
        print(f"\n{'=== ESTATE MAP ==='}")
        for name, exits in location_map.items():
            marker = " ◄ YOU" if name == state.locations.get(env.agent_location_id, type("", (), {"name": ""})()).name else ""
            print(f"  {name}{marker}")
            if exits:
                print(f"    └─ exits: {', '.join(exits)}")

    def _show_suspects() -> None:
        print(f"\n{'=== SUSPECTS ==='}")
        for s in suspect_list:
            loc = state.locations.get(s.location_id)
            loc_name = loc.name if loc else "unknown"
            print(f"  • {s.full_name}  (last seen: {loc_name})")

    def _interactive_accuse() -> tuple[str, str, str]:
        print("\nYou are about to make your final accusation. This ends the game.")
        print("Suspects:", ", ".join(s.full_name for s in suspect_list))
        weapons = [o.name for o in state.objects.values() if o.is_weapon]
        print("Weapons: ", ", ".join(weapons))
        print("Locations:", ", ".join(l.name for l in state.locations.values()))
        suspect  = input("\nWho did it?          > ").strip()
        weapon   = input("What weapon?         > ").strip()
        location = input("Where did it happen? > ").strip()
        return suspect, weapon, location

    # ── Initial briefing ────────────────────────────────────────────────
    briefing = render_initial_briefing(env)
    _print_box(briefing)

    # ── REPL ────────────────────────────────────────────────────────────
    while not env.is_solved:
        budget = env.budget_remaining
        loc = env.get_current_location()
        loc_name = loc.name if loc else "?"

        prompt = f"\n[{loc_name} | budget: {budget}] > "
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGame aborted.")
            return

        if not raw:
            continue

        low = raw.lower()

        # Special non-action commands
        if low in ("help", "h", "?"):
            print(HELP_TEXT)
            continue

        if low == "map":
            _show_map()
            continue

        if low in ("suspects", "suspect list"):
            _show_suspects()
            continue

        if low in ("quit", "exit", "q"):
            print("You leave the case unsolved.")
            return

        # Parse into action
        parsed = _parse_command(raw)

        if parsed is None:
            print("Unknown command. Type 'help' for a list of commands.")
            continue

        action, kwargs = parsed

        # --- TALK_TO: prompt for question ---
        if action == AgentAction.TALK_TO and kwargs.pop("_needs_question", False):
            char_name = kwargs["character_name"]
            if not char_name:
                print("Who do you want to talk to?")
                continue
            question = input(f'What do you ask {char_name}? > ').strip()
            if not question:
                question = "Where were you at the time of the murder?"
            kwargs["question"] = question

        # --- ACCUSE: interactive prompt ---
        if action == AgentAction.ACCUSE and kwargs.pop("_interactive", False):
            suspect, weapon, location = _interactive_accuse()
            kwargs = {
                "suspect_name": suspect,
                "weapon_name":  weapon,
                "location_name": location,
            }
            confirm = input(
                f"\nAccuse {suspect!r} with {weapon!r} in {location!r}? [y/N] > "
            ).strip().lower()
            if confirm != "y":
                print("Accusation cancelled.")
                continue

        # --- Execute ---
        result = env.step(action, **kwargs)
        obs = render_step_observation(env, result.observation)
        _print_result(obs)

    # ── End screen ──────────────────────────────────────────────────────
    summary = env.get_episode_summary()
    culprit  = state.get_culprit()
    weapon   = state.objects.get(state.murder_weapon_id)
    murder_loc = state.locations.get(state.murder_location_id)

    _print_box(
        f"{'CASE CLOSED' if summary['accusation_correct'] else 'CASE FAILED'}\n"
        f"\n"
        f"  True answer : {culprit.full_name if culprit else '?'}"
        f"  with the {weapon.name if weapon else '?'}"
        f"  in the {murder_loc.name if murder_loc else '?'}\n"
        f"\n"
        f"  Actions used : {summary['actions_taken']} / {summary['budget']}\n"
        f"  Evidence found : {len(summary['evidence_discovered'])} / {summary['total_evidence']}\n"
        f"  Interviews : {len(summary['characters_interviewed'])} character(s) questioned\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Play a mystery case interactively")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--load", metavar="FILE", help="Load a saved world JSON")
    group.add_argument(
        "--level",
        default="MEDIUM",
        choices=list(LEVEL_NAMES),
        help="Complexity level for a fresh case (default: MEDIUM)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if omitted)")
    parser.add_argument(
        "--npc-url",
        default=None,
        help="OpenAI-compatible base URL for LLM-powered NPC interviews (e.g. http://localhost:8123/v1). "
             "If omitted, NPCs use a deterministic fallback.",
    )
    parser.add_argument(
        "--npc-model",
        default="Qwen/Qwen3.5-27B",
        help="Model served at --npc-url (default: Qwen/Qwen3.5-27B)",
    )
    parser.add_argument(
        "--npc-seed",
        type=int,
        default=42,
        help="Fixed seed for NPC responses (default: 42)",
    )
    args = parser.parse_args()

    if args.load:
        world_state = WorldState.load(args.load)
        print(f"Loaded world from {args.load}  (seed={world_state.seed})")
    else:
        import random
        seed = args.seed if args.seed is not None else random.randint(0, 999999)
        level = LEVEL_NAMES[args.level.upper()]
        config = COMPLEXITY_PRESETS[level]
        print(f"Generating a {args.level} mystery (seed={seed}) ...")
        world_state = generate_mystery(config, seed)

    env = MysteryEnvironment(world_state)

    if args.npc_url:
        from mystery_world.npc_responder import NPCResponder
        responder = NPCResponder(base_url=args.npc_url, model=args.npc_model, seed=args.npc_seed)
        env.set_npc_responder(responder)
        print(f"NPC interviews: {args.npc_model} @ {args.npc_url}")
    else:
        print("NPC interviews: deterministic fallback (pass --npc-url to use an LLM)")

    play(env)


if __name__ == "__main__":
    main()
