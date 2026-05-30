"""
Verification utilities for benchmark instances.

Provides:
    1. Structural consistency checks (graph connectivity, entity references)
    2. Solvability verification (sufficient evidence, breakable alibis)
    3. Human annotation export (readable case summaries for annotators)
    4. Cross-instance diversity metrics
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mystery_world.entities import CharacterRole, EdgeType, EvidenceState, EvidenceType
from mystery_world.world import WorldState


# ---------------------------------------------------------------------------
# Structural consistency
# ---------------------------------------------------------------------------

def check_structural_consistency(state: WorldState) -> dict[str, Any]:
    """Verify internal consistency of a generated world state."""
    issues: list[str] = []

    # 1. Location graph is connected
    if state.locations:
        visited: set[str] = set()
        start = next(iter(state.locations))
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            loc = state.locations.get(node)
            if loc:
                for adj in loc.adjacent_ids:
                    if adj not in visited:
                        stack.append(adj)
        if visited != set(state.locations.keys()):
            unreachable = set(state.locations.keys()) - visited
            issues.append(f"Disconnected locations: {unreachable}")
    
    # 2. All character location_ids reference valid locations
    for cid, char in state.characters.items():
        if char.location_id and char.location_id not in state.locations:
            issues.append(f"Character {char.full_name} references invalid location {char.location_id}")
    
    # 3. All evidence location_ids reference valid locations
    for eid, ev in state.evidence.items():
        if ev.location_id and ev.location_id not in state.locations:
            issues.append(f"Evidence {ev.name} references invalid location {ev.location_id}")

    # 4. Culprit and victim exist
    if state.culprit_id not in state.characters:
        issues.append(f"Culprit ID {state.culprit_id} not in characters")
    if state.victim_id not in state.characters:
        issues.append(f"Victim ID {state.victim_id} not in characters")
    
    # 5. Murder weapon exists
    if state.murder_weapon_id not in state.objects:
        issues.append(f"Murder weapon ID {state.murder_weapon_id} not in objects")
    
    # 6. Murder location exists
    if state.murder_location_id not in state.locations:
        issues.append(f"Murder location ID {state.murder_location_id} not in locations")
    
    # 7. Exactly one culprit
    culprits = [c for c in state.characters.values() if c.is_culprit]
    if len(culprits) != 1:
        issues.append(f"Expected exactly 1 culprit, found {len(culprits)}")
    
    # 8. Exactly one victim
    victims = [c for c in state.characters.values() if CharacterRole.VICTIM in c.roles]
    if len(victims) != 1:
        issues.append(f"Expected exactly 1 victim, found {len(victims)}")
    
    # 9. Adjacency is symmetric
    for lid, loc in state.locations.items():
        for adj_id in loc.adjacent_ids:
            adj = state.locations.get(adj_id)
            if adj and lid not in adj.adjacent_ids:
                issues.append(f"Asymmetric adjacency: {lid} -> {adj_id} but not reverse")
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "num_locations": len(state.locations),
        "num_characters": len(state.characters),
        "num_evidence": len(state.evidence),
        "num_objects": len(state.objects),
    }


# ---------------------------------------------------------------------------
# Solvability
# ---------------------------------------------------------------------------

def check_solvability(state: WorldState) -> dict[str, Any]:
    """
    Comprehensive solvability check.

    A mystery is solvable if:
        - There exists a logical chain from discoverable evidence to the culprit
        - The culprit's alibi is breakable (unverified or no corroborator)
        - At least 2 pieces of usable evidence point to the culprit
        - The murder weapon is discoverable
        - Only one fact
    """
    culprit = state.get_culprit()
    if culprit is None:
        return {"solvable": False, "reason": "No culprit defined"}
    
    # Evidence pointing to culprit
    culprit_evidence = [
        e for e in state.evidence.values()
        if e.linked_character_id == state.culprit_id
        and not e.is_red_herring
    ]
    usable_culprit_evidence = [e for e in culprit_evidence if e.is_usable()]

    # Evidence types available
    evidence_types = Counter(e.evidence_type for e in usable_culprit_evidence)

    # Alibi analysis
    alibi_breakable = (        
        not culprit.has_alibi
        or culprit.alibi_corroborator_id is None                                  
        or not culprit.alibi_corroboration_is_genuine                           
        or culprit.alibi_has_gap                                                  
    )  

    # Murder weapon discoverable
    weapon_obj = state.objects.get(state.murder_weapon_id)
    weapon_discoverable = weapon_obj is not None and weapon_obj.location_id in state.locations

    # Can innocents be eliminated?
    non_culprit_suspects = [
        c for c in state.characters.values()
        if CharacterRole.SUSPECT in c.roles and not c.is_culprit
    ]
    suspects_with_corroborated_alibis = sum(                                          
        1 for s in non_culprit_suspects                                         
        if s.has_alibi and s.alibi_corroborator_id is not None
    )
    suspects_with_unverified_alibis = sum(
        1 for s in non_culprit_suspects                                           
        if s.has_alibi and s.alibi_corroborator_id is None                      
    )

    # Solution uniqueness: is there enough to distinguish culprit from others?    
    distinguishing_evidence = len(usable_culprit_evidence)                      

    solvable = (               
        distinguishing_evidence >= 2
        and alibi_breakable    
        and weapon_discoverable                                                 
        and EvidenceType.PHYSICAL in evidence_types
    )
    return {
        "solvable": solvable,
        "culprit_evidence_total": len(culprit_evidence),
        "culprit_evidence_usable": len(usable_culprit_evidence),
        "evidence_type_breakdown": {k.name: v for k, v in evidence_types.items()},
        "alibi_breakable": alibi_breakable,                                       
        "weapon_discoverable": weapon_discoverable,                               
        "suspects_with_corroborated_alibis": suspects_with_corroborated_alibis,   
        "suspects_with_unverified_alibis": suspects_with_unverified_alibis,    
        "total_non_culprit_suspects": len(non_culprit_suspects),                  
        "red_herrings": sum(1 for e in state.evidence.values() if e.is_red_herring),                                 
    }


def check_locard_solvability(state: WorldState) -> dict[str, Any]:
    """Verify the Locard triangle is closable from ANCHORED evidence alone:
    each edge has at least one discoverable, fresh, non-portable trace pointing
    to the correct entities — proof the culprit cannot carry off."""
    issues: list[str] = []
    murder_ts = state.murder_timestamp
    threshold = state.freshness_threshold

    for edge in EdgeType:
        found = False
        for ev in state.evidence.values():
            if ev.is_red_herring or ev.state == EvidenceState.DESTROYED or ev.discovery_difficulty >= 1.0:
                continue
            if not ev.anchored:
                continue
            if ev.relevance is None or ev.relevance.edge_type != edge:
                continue
            if abs(ev.relevance.contact_timestamp - murder_ts) >= threshold:
                continue
            rel = ev.relevance
            if edge == EdgeType.SUSPECT_WEAPON and state.culprit_id in rel.subject_ids and state.murder_weapon_id in rel.subject_ids:
                found = True
            elif edge == EdgeType.WEAPON_VICTIM and state.murder_weapon_id in rel.subject_ids and state.victim_id in rel.subject_ids:
                found = True
            elif edge == EdgeType.SUSPECT_ROOM and state.culprit_id in rel.subject_ids and state.murder_location_id in rel.subject_ids:
                found = True
            if found:
                break
        if not found:
            issues.append(f"No discoverable fresh evidence for edge {edge.name}")

    return {
        "solvable": len(issues) == 0,
        "issues": issues,
        "triangle_edges_covered": 3 - len(issues),
    }
    
# ---------------------------------------------------------------------------
# Human annotation export
# ---------------------------------------------------------------------------

def export_annotation_sheet(state: WorldState, output_path: str | Path) -> None:
    """
    Export a human-readable case summary for annotation / quality review.

    Annotators verify:
        - The case is logically solvable
        - The narrative is coherent
        - Difficulty feels appropriate for the labelled level
    """
    culprit = state.get_culprit()
    victim = state.get_victim()
    weapon = state.objects.get(state.murder_weapon_id)
    murder_loc = state.locations.get(state.murder_location_id)

    lines = [
        "=" * 70,
        f"CASE #{state.seed} — ANNOTATION SHEET",
        "=" * 70,
        "",
        "--- GROUND TRUTH (for annotator reference only) ---",
        f"  Culprit: {culprit.full_name if culprit else 'N/A'}",
        f"  Victim:  {victim.full_name if victim else 'N/A'}",
        f"  Weapon:  {weapon.name if weapon else 'N/A'}",
        f"  Location: {murder_loc.name if murder_loc else 'N/A'}",
        f"  Motive:  {state.motive}",
        f"  Time of murder: step {state.murder_step}",
        "",
        "--- CHARACTERS ---",
    ]

    for cid, char in state.characters.items():
        roles = ", ".join(r.name for r in char.roles)
        alibi = f"Alibi: {char.alibi_details}" if char.has_alibi else "No alibi"
        motive = f"Motive: {char.motive}" if char.motive else ""
        lines.append(f"  {char.full_name} [{roles}] ({char.personality})")
        lines.append(f"  {alibi}")
        if motive:
            lines.append(f"    {motive}")
        if char.is_culprit:
            lines.append(f"    ** THIS IS THE CULPRIT **")
        lines.append("")

    lines.append("--- LOCATIONS ---")
    for lid, loc in state.locations.items():
        adj = [state.locations[a].name for a in loc.adjacent_ids if a in state.locations]
        lines.append(f"  {loc.name} ({loc.tag.name}) → {', '.join(adj)}")
    lines.append("")


    lines.append("--- EVIDENCE ---")
    for eid, ev in state.evidence.items():
        linked = state.characters.get(ev.linked_character_id)
        linked_name = linked.full_name if linked else "N/A"
        herring = " [RED HERRING]" if ev.is_red_herring else ""
        lines.append(f"    Points to: {linked_name}{herring}")
        lines.append(f"    Location: {state.locations.get(ev.location_id, type('', (), {'name': 'unknown'})()).name}")
        lines.append(f"    Difficulty: {ev.discovery_difficulty:.2f}")
        lines.append("")
    
    lines.extend([
        "--- ANNOTATION QUESTIONS ---",
        "1. Is this case logically solvable given the evidence? [Yes / No / Partially]",
        "2. Is the narrative coherent (no contradictions)? [Yes / No]",
        "3. Rate difficulty (1=trivial, 5=expert): [ ]",
        "4. Are the red herrings plausible? [Yes / No / N/A]",
        "5. Any issues or notes:",
        "",
        "Annotator: _______________  Date: _______________",
    ])

    Path(output_path).write_text("\n".join(lines))

# ---------------------------------------------------------------------------
# Diversity metrics across a benchmark suite
# ---------------------------------------------------------------------------

def compute_diversity_metrics(instances_dir: str | Path) -> dict[str, Any]:
    """
    Compute diversity statistics across a benchmark suite to verify
    that procedural generation produces varied instances.
    """
    instances_dir = Path(instances_dir)
    manifest = json.loads((instances_dir / "manifest.json").read_text())

    culprits: list[str] = []
    weapons: list[str] = []
    locations: list[str] = []
    motives: list[str] = []

    for entry in manifest:
        culprits.append(entry.get("culprit", ""))
        weapons.append(entry.get("weapon", ""))
        locations.append(entry.get("location", ""))
        motives.append(entry.get("motive", ""))

    def _entropy(items: list[str]) -> float:
        counts = Counter(items)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    return {
        "n_instances": len(manifest),
        "unique_culprits": len(set(culprits)),
        "unique_weapons": len(set(weapons)),
        "unique_locations": len(set(locations)),
        "unique_motives": len(set(motives)),
        "culprit_entropy": _entropy(culprits),
        "weapon_entropy": _entropy(weapons),
        "location_entropy": _entropy(locations),
        "motive_entropy": _entropy(motives),
    }
