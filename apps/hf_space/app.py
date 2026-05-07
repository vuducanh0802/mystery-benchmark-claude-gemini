"""
Gradio web UI for MysteryArena (textual world).

Designed to be deployed as a Hugging Face Space:

    apps/hf_space/
      app.py
      requirements.txt
      README.md         # contains HF Space frontmatter

NPCs use the OpenAI ChatGPT API (default: gpt-4o-mini).
The Space must be configured with an OPENAI_API_KEY secret.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Make repo importable when running from apps/hf_space.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from mystery_world import COMPLEXITY_PRESETS, ComplexityLevel
from mystery_world.generator import generate_mystery
from mystery_world.narrator import (
    render_character_summary,
    render_evidence_summary,
    render_initial_briefing,
    render_step_observation,
)
from mystery_world.npc_responder import NPCResponder
from mystery_world.world import AgentAction, MysteryEnvironment

LEVELS = ["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXPERT"]
DEFAULT_NPC_MODEL = os.environ.get("MYSTERY_NPC_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _new_session(level: str, seed: int | None) -> dict:
    cfg = COMPLEXITY_PRESETS[ComplexityLevel[level]]
    if seed is None or seed < 0:
        seed = random.randint(0, 999_999)
    state = generate_mystery(seed=seed, config=cfg)
    env = MysteryEnvironment(state)

    # Wire OpenAI NPCs if a key is available.
    if os.environ.get("OPENAI_API_KEY"):
        env.set_npc_responder(
            NPCResponder(base_url=None, model=DEFAULT_NPC_MODEL,
                         seed=42, api_key_env="OPENAI_API_KEY")
        )

    briefing = render_initial_briefing(env)
    return {
        "env": env,
        "level": level,
        "seed": seed,
        "transcript": [("system", briefing)],
        "ended": False,
    }


def _sidebar_text(env: MysteryEnvironment) -> str:
    state = env.state
    cur = env.get_current_location()
    cur_name = cur.name if cur else "(unknown)"
    suspects = [c.full_name for c in state.characters.values()
                if any(r.name == "SUSPECT" for r in c.roles) and c.is_alive]
    weapons = [o.name for o in state.objects.values() if o.is_weapon]
    locations = [l.name for l in state.locations.values()]
    inv = [state.evidence[e].description for e in env.agent_inventory if e in state.evidence] or ["(empty)"]

    lines = [
        f"**Location:** {cur_name}",
        f"**Actions left:** {env.budget_remaining}",
        "",
        "**Suspects**",
        *[f"- {s}" for s in suspects],
        "",
        "**Weapons (candidates)**",
        *[f"- {w}" for w in weapons],
        "",
        "**Locations**",
        *[f"- {l}" for l in locations],
        "",
        "**Inventory**",
        *[f"- {i}" for i in inv],
    ]
    return "\n".join(lines)


def _render_transcript(transcript: list[tuple[str, str]]) -> str:
    out = []
    for role, text in transcript:
        if role == "system":
            out.append(f"_{text.strip()}_\n")
        elif role == "you":
            out.append(f"**> {text}**")
        else:
            out.append(text)
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Command handling (mirrors scripts/play.py but stateless)
# ---------------------------------------------------------------------------

def _parse(raw: str) -> tuple[AgentAction, dict] | None:
    tokens = raw.strip().split()
    if not tokens:
        return None
    verb = tokens[0].lower()
    rest = " ".join(tokens[1:])
    if verb in ("look", "l") and (not rest or rest in ("room", "around")):
        return AgentAction.EXAMINE_LOCATION, {}
    if verb in ("go", "move", "walk"):
        return AgentAction.MOVE, {"target_location": rest}
    if verb in ("examine", "inspect", "x") and rest:
        return AgentAction.EXAMINE_OBJECT, {"object_name": rest}
    if verb in ("wait", "w"):
        return AgentAction.WAIT, {}
    if verb in ("inventory", "inv", "i"):
        return AgentAction.CHECK_INVENTORY, {}
    if verb in ("take", "grab"):
        return AgentAction.TAKE_OBJECT, {"object_name": rest}
    if verb in ("talk", "ask"):
        # Format: "talk <name> :: <question>"
        if "::" in rest:
            name, q = rest.split("::", 1)
            return AgentAction.TALK_TO, {"character_name": name.strip(), "question": q.strip()}
        return AgentAction.TALK_TO, {"character_name": rest, "question": "What can you tell me?"}
    return None


def _help_text() -> str:
    return (
        "**Commands**\n"
        "- `look` — describe current room\n"
        "- `go <room>` — move to an adjacent room\n"
        "- `examine <object>` — inspect an object\n"
        "- `take <object>` — pick up\n"
        "- `talk <name> :: <question>` — interview an NPC (use `::` to add a question)\n"
        "- `inventory` — list collected evidence\n"
        "- `wait` — pass a turn\n"
        "- `accuse` — bring up the accusation form\n"
    )


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def cb_new_game(level: str, seed_text: str, session: dict | None):
    try:
        seed = int(seed_text) if seed_text.strip() else None
    except ValueError:
        seed = None
    s = _new_session(level, seed)
    return (
        s,
        _render_transcript(s["transcript"]),
        _sidebar_text(s["env"]),
        f"Seed: {s['seed']}  |  Level: {s['level']}",
        gr.update(value=""),  # clear command box
    )


def cb_command(cmd: str, session: dict | None):
    if not session:
        return session, "Click **New Game** to start.", "", "", ""
    if session.get("ended"):
        return session, _render_transcript(session["transcript"]) + "\n\n_Game over. Start a new game._", \
               _sidebar_text(session["env"]), "", ""

    raw = (cmd or "").strip()
    if not raw:
        return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), "", ""

    if raw.lower() in ("help", "?"):
        session["transcript"].append(("you", raw))
        session["transcript"].append(("game", _help_text()))
        return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), "", ""

    if raw.lower() == "accuse":
        # Surfaced via the accusation panel. Do not consume an action.
        session["transcript"].append(("you", raw))
        session["transcript"].append(("game", "Use the **Accusation** panel below to file your accusation."))
        return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), "", ""

    parsed = _parse(raw)
    if parsed is None:
        session["transcript"].append(("you", raw))
        session["transcript"].append(("game", "Unknown command. Type `help` for the list."))
        return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), "", ""

    action, kwargs = parsed
    env: MysteryEnvironment = session["env"]
    result = env.step(action, **kwargs)
    obs = render_step_observation(env, result.observation)
    session["transcript"].append(("you", raw))
    session["transcript"].append(("game", obs))

    # Periodic summaries (every 3 actions) — same cadence as runner.py.
    if env.actions_taken % 3 == 0:
        session["transcript"].append(("game", render_evidence_summary(env)))
        session["transcript"].append(("game", render_character_summary(env)))

    if env.budget_remaining <= 0:
        session["transcript"].append(("game", "_Action budget exhausted. File your accusation now._"))

    return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), "", ""


def cb_accuse(suspect: str, weapon: str, location: str, session: dict | None):
    if not session:
        return session, "Start a new game first.", "", ""
    if session.get("ended"):
        return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), \
               "Game already ended."

    env: MysteryEnvironment = session["env"]
    result = env.step(
        AgentAction.ACCUSE,
        suspect_name=suspect, weapon_name=weapon, location_name=location,
    )
    obs = render_step_observation(env, result.observation)
    session["transcript"].append(("you", f"ACCUSE — {suspect} | {weapon} | {location}"))
    session["transcript"].append(("game", obs))

    summary = env.get_episode_summary()
    correct = summary.get("accusation_correct")
    score = summary.get("scoring", {}).get("composite_score")
    session["transcript"].append((
        "game",
        f"_Result: accusation_correct={correct}  composite_score={score}_",
    ))
    session["ended"] = True
    return session, _render_transcript(session["transcript"]), _sidebar_text(session["env"]), \
           ("Correct!" if correct else "Incorrect.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MysteryArena — Textual World") as demo:
        gr.Markdown("# MysteryArena — Textual World\n"
                    "_A procedurally generated murder-mystery. Interview NPCs, examine objects, then accuse._")
        session = gr.State(value=None)
        info_bar = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=3):
                transcript_md = gr.Markdown("Click **New Game** to start.")
                command_box = gr.Textbox(label="Command", placeholder="e.g. look | go Library | talk Petra :: Where were you at 9pm?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    help_btn = gr.Button("Help")

                with gr.Accordion("Accusation", open=False):
                    accuse_suspect = gr.Textbox(label="Suspect (full name)")
                    accuse_weapon = gr.Textbox(label="Weapon")
                    accuse_location = gr.Textbox(label="Location")
                    accuse_btn = gr.Button("File Accusation", variant="stop")
                    accuse_result = gr.Markdown("")

            with gr.Column(scale=2):
                with gr.Row():
                    level_dd = gr.Dropdown(choices=LEVELS, value="MEDIUM", label="Level")
                    seed_box = gr.Textbox(label="Seed (blank = random)", value="")
                new_btn = gr.Button("New Game", variant="primary")
                sidebar_md = gr.Markdown("")

        new_btn.click(cb_new_game, [level_dd, seed_box, session],
                      [session, transcript_md, sidebar_md, info_bar, command_box])
        submit_btn.click(cb_command, [command_box, session],
                         [session, transcript_md, sidebar_md, command_box, accuse_result])
        command_box.submit(cb_command, [command_box, session],
                           [session, transcript_md, sidebar_md, command_box, accuse_result])
        help_btn.click(lambda s: cb_command("help", s), [session],
                       [session, transcript_md, sidebar_md, command_box, accuse_result])
        accuse_btn.click(cb_accuse,
                         [accuse_suspect, accuse_weapon, accuse_location, session],
                         [session, transcript_md, sidebar_md, accuse_result])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
