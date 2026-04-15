"""
List all available benchmark examples with their key metadata.

Usage:
    python scripts/list_examples.py
    python scripts/list_examples.py --level EASY
    python scripts/list_examples.py --show trivial_seed_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

DIVIDER = "─" * 70


def _render_action(step: dict) -> str:
    """Return a human-readable one-liner for a single oracle action step."""
    action = step["action"]
    k = step["kwargs"]

    if action == "MOVE":
        return f"Go to the {k['target_location']}"
    if action == "EXAMINE_OBJECT":
        return f"Examine the {k['object_name']}"
    if action == "EXAMINE_LOCATION":
        return "Look around the room"
    if action == "TALK_TO":
        q = k.get("question", "")
        q_part = f' — "{q}"' if q else ""
        return f"Talk to {k['character_name']}{q_part}"
    if action == "WAIT":
        return "Wait"
    if action == "CHECK_INVENTORY":
        return "Check inventory"
    if action == "TAKE_OBJECT":
        return f"Take the {k.get('object_name', '?')}"
    if action == "ACCUSE":
        lines = [
            f"Accuse {k['suspect_name']} — with the {k['weapon_name']} in the {k['location_name']}",
        ]
        sw = k.get("suspect_weapon_evidence") or []
        wv = k.get("weapon_victim_evidence") or []
        sr = k.get("suspect_room_evidence") or []
        if sw:
            lines.append(f"    Suspect ↔ Weapon  : {', '.join(sw)}")
        if wv:
            lines.append(f"    Weapon  ↔ Victim  : {', '.join(wv)}")
        if sr:
            lines.append(f"    Suspect ↔ Room    : {', '.join(sr)}")
        contra = k.get("alibi_contradiction")
        if contra:
            ev = ", ".join(contra.get("contradiction_evidence") or [])
            lines.append(
                f"    Alibi disproved   : claimed {contra.get('claimed_location', '?')}"
                f" at {contra.get('claimed_time', '?')}"
                + (f" → contradicted by {ev}" if ev else "")
            )
        elims = k.get("eliminations") or {}
        for name, info in elims.items():
            corr = info.get("corroborator", "")
            eid  = info.get("evidence_id", "")
            lines.append(
                f"    Clear {name:<20}: {eid}"
                + (f" (witnessed by {corr})" if corr else "")
            )
        return "\n".join(lines)
    return f"{action} {k}"


def _show_example(ex: dict) -> None:
    gt = ex["ground_truth"]
    sc = ex["oracle_scores"]
    multi = "yes" if ex.get("multi_evidence") else "no"

    print(DIVIDER)
    print(f"  Example  : {ex['id']}")
    print(f"  Level    : {ex['complexity']}   seed={ex['seed']}   multi-evidence={multi}")
    print(DIVIDER)

    print("\nGROUND TRUTH")
    print(f"  Culprit  : {gt['culprit']}")
    print(f"  Weapon   : {gt['weapon']}")
    print(f"  Location : {gt['location']}")

    alibi_claims = ex.get("alibi_claims") or []
    if alibi_claims:
        print("\nCULPRIT'S ALIBI CLAIMS")
        for c in alibi_claims:
            print(f"  \"{c['location']}\" at {c['time']}")

    elims = ex.get("eliminations") or []
    if elims:
        print("\nINNOCENT SUSPECTS (SUSPECT_ELSEWHERE evidence)")
        for e in elims:
            print(f"  {e['suspect']:<25} [{e['evidence_id']}]  corroborated by {e['corroborator']}")

    print("\nORACLE SCORES")
    print(f"  Composite  {sc['composite']:.4f}  |  Triangle {sc['triangle']:.1f}/3.0  "
          f"|  Alibi {sc['alibi']:.4f}  |  Elimination {sc['elimination']:.4f}")
    print(f"  SW {sc['sw']:.4f}  WV {sc['wv']:.4f}  SR {sc['sr']:.4f}")

    print(f"\nORACLE WALKTHROUGH  ({len(ex['oracle_action_sequence'])} steps)")
    print(DIVIDER)
    for step in ex["oracle_action_sequence"]:
        rendered = _render_action(step)
        # Indent continuation lines of multi-line renders
        lines = rendered.split("\n")
        print(f"  {step['step']:>2}.  {lines[0]}")
        for extra in lines[1:]:
            print(f"        {extra}")
    print(DIVIDER)
    print(f"\nTo play this case:  python scripts/play.py --example {ex['id']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="List benchmark examples")
    parser.add_argument("--level", default=None, help="Filter by complexity level")
    parser.add_argument("--show", metavar="ID", default=None,
                        help="Show full details and oracle walkthrough for one example")
    args = parser.parse_args()

    if args.show:
        path = EXAMPLES_DIR / f"{args.show}.json"
        if not path.exists():
            available = sorted(p.stem for p in EXAMPLES_DIR.glob("*.json"))
            print(f"Example '{args.show}' not found. Available:")
            for eid in available:
                print(f"  {eid}")
            sys.exit(1)
        _show_example(json.loads(path.read_text()))
        return

    paths = sorted(EXAMPLES_DIR.glob("*.json"))
    if not paths:
        print("No examples found. Run: python scripts/generate_examples.py")
        sys.exit(1)

    if args.level:
        paths = [p for p in paths if p.stem.startswith(args.level.lower())]

    header = f"{'ID':<25}  {'LEVEL':<8}  {'SEED':>5}  {'MULTI':>5}  {'CULPRIT':<30}  {'COMPOSITE':>9}"
    print(header)
    print("-" * len(header))
    for p in paths:
        ex = json.loads(p.read_text())
        multi = "yes" if ex.get("multi_evidence") else "no"
        culprit = ex["ground_truth"]["culprit"]
        composite = ex["oracle_scores"]["composite"]
        print(
            f"{ex['id']:<25}  {ex['complexity']:<8}  {ex['seed']:>5}  {multi:>5}"
            f"  {culprit:<30}  {composite:>9.4f}"
        )

    print(f"\nTotal: {len(paths)} example(s)")
    print("\nTo play one:   python scripts/play.py --example <ID>")
    print("To view one:   python scripts/list_examples.py --show <ID>")


if __name__ == "__main__":
    main()
