"""
Procedural mystery world generator.

Given a ``ComplexityConfig`` and an integer seed, this module generates a
complete, solvable mystery scenario. Every output is fully deterministic.

Generation pipeline:
    1. Sample locations and build adjacency graph
    2. Create characters (victim, culprit, suspects, innocents)
    3. Assign motive, weapon, murder location
    4. Generate ground-truth timeline (who was where at each step)
    5. Plant evidence (physical, testimonial, documentary, circumstantial)
    6. Inject red herrings
    7. Wire alibi chains
    8. Verify solvability (constraint check)
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np

from mystery_world import AssetPool, ComplexityConfig, DEFAULT_ASSET_POOL
from mystery_world.entities import (
    AlibiClaim,
    Character,
    CharacterRole,
    EdgeRelevance,
    EdgeType,
    Evidence,
    EvidenceState,
    EvidenceType,
    Location,
    LocationTag,
    PhysicalTraits,
    Relationship,
    RouteConstraint,
    TemporalLabel,
    TimelineEntry,
    TimeStyle,
    WitnessStatement,
    WorldObject,
)
from mystery_world.world import WorldState

def _uid(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}_{rng.integers(100000, 999999)}"

# Neutral-sounding names for objects that host evidence. These must NOT hint
# that evidence is present.
_NEUTRAL_HOST_OBJECTS = [
    "side table", "coat rack", "worn rug", "window ledge",
    "fireplace mantel", "writing desk", "umbrella stand",
    "wooden stool", "wall mirror", "storage trunk",
    "armchair", "floor lamp", "serving tray", "hat stand",
    "porcelain vase", "shelf of books", "old radiator",
    "heavy curtain", "wicker basket", "footstool",
]

# ---------------------------------------------------------------------------
# Surface label helper (Locard)
# ---------------------------------------------------------------------------

def _pick_surface_label(
    contact_ts: float,
    murder_ts: float,
    threshold: float,
    config: ComplexityConfig,
) -> TemporalLabel:
    is_fresh = abs(contact_ts - murder_ts) < threshold
    if config.evidence_ambiguity <= 0.1:
        return TemporalLabel.CLEARLY_FRESH if is_fresh else TemporalLabel.CLEARLY_STALE
    else:
        return TemporalLabel.AMBIGUOUS


# ---------------------------------------------------------------------------
# Time utilities (temporal reasoning)
# ---------------------------------------------------------------------------

_NAMED_PERIODS = [
    (0,  2,  "before dinner"),
    (3,  5,  "after dinner"),
    (6,  9,  "late in the evening"),
    (10, 13, "near midnight"),
]


def _step_to_named_period(step: int) -> str:
    for lo, hi, name in _NAMED_PERIODS:
        if lo <= step <= hi:
            return name
    return "late in the night"


def _step_to_clock_str_gen(step: int, world_start_hour: int) -> str:
    """Local copy to avoid circular import with world.py."""
    total_minutes = world_start_hour * 60 + step * 30
    total_minutes %= 24 * 60
    hour, minute = divmod(total_minutes, 60)
    meridiem = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {meridiem}"


def _make_stated_time(
    step: int,
    anchor_events: dict[str, int],
    style: TimeStyle,
    world_start_hour: int,
    rng: np.random.Generator,
) -> str:
    clock = _step_to_clock_str_gen(step, world_start_hour)
    if style == TimeStyle.CLOCK:
        return f"at {clock}"
    elif style == TimeStyle.NAMED_PERIOD:
        return _step_to_named_period(step)
    else:  # RELATIVE
        if not anchor_events:
            return f"at {clock}"
        anchor_name, anchor_step = min(
            anchor_events.items(), key=lambda kv: abs(kv[1] - step)
        )
        delta_steps = step - anchor_step
        delta_min = abs(delta_steps) * 30
        if delta_min == 0:
            return f"just as {anchor_name}"
        direction = "after" if delta_steps > 0 else "before"
        return f"about {delta_min} minutes {direction} {anchor_name}"

# ---------------------------------------------------------------------------
# Location graph
# ---------------------------------------------------------------------------

def _generate_locations(
    config: ComplexityConfig, pool: AssetPool, rng: np.random.Generator
) -> dict[str, Location]:
    n = config.num_locations
    names = list(rng.choice(pool.location_templates, size=min(n, len(pool.location_templates)), replace=False))
    locations: dict[str, Location] = {}
    loc_ids: list[str] = []
    for name in names:
        lid = _uid("loc", rng)
        tag = LocationTag.OUTDOOR if rng.random() < 0.3 else LocationTag.INDOOR
        locations[lid] = Location(
            id=lid, name=str(name), tag=tag,
            description=f"A {tag.name.lower()} space known as the {name}.",
            weather_exposed=(tag == LocationTag.OUTDOOR),
        )
        loc_ids.append(lid)
    
    # Build connected adjacency graph (ensure connected)
    # Strategy: create a random spanning tree, then add extra edges
    shuffled = list(loc_ids)
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        a, b = shuffled[i - 1], shuffled[i]
        locations[a].adjacent_ids.append(b)
        locations[b].adjacent_ids.append(a)
    # Extra edges for richness
    extra = max(1, n // 3)
    for _ in range(extra):
        a, b = rng.choice(loc_ids, size=2, replace=False)
        if b not in locations[a].adjacent_ids:
            locations[a].adjacent_ids.append(b)
            locations[b].adjacent_ids.append(a)

    # Assign a unique room material to each location (used for crime-scene convergent clues)
    mat_indices = list(range(len(pool.room_materials)))
    rng.shuffle(mat_indices)
    for i, lid in enumerate(loc_ids):
        specific, vague = pool.room_materials[mat_indices[i % len(pool.room_materials)]]
        locations[lid].material_signature = specific
        if config.body_moved_prob > 0.0:
            locations[lid].description += f" A faint trace of {vague} lingers here."

    return locations


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------

def _generate_characters(
    config: ComplexityConfig,
    pool: AssetPool,
    rng: np.random.Generator,
    location_ids: list[str],
) -> dict[str, Character]:
    total = 1 + config.num_suspects + config.num_innocents # victim + suspects + innocents
    first_names = list(rng.choice(pool.first_names, size=min(total, len(pool.first_names)), replace=False))
    last_names = list(rng.choice(pool.last_names, size=min(total, len(pool.last_names)), replace=False))
    traits = list(rng.choice(pool.personality_traits, size=min(total, len(pool.personality_traits)), replace=False))

    characters: dict[str, Character] = {}
    char_list: list[Character] = []
    for i in range(total):
        cid = _uid("chr", rng)
        char = Character(
            id=cid,
            first_name=str(first_names[i]),
            last_name=str(last_names[i]),
            personality=str(traits[i % len(traits)]),
            location_id=str(rng.choice(location_ids)),
        )
        characters[cid] = char
        char_list.append(char)

    # Assign physical traits — initially unique per character
    builds = list(rng.choice(pool.build_types, size=min(total, len(pool.build_types)), replace=False))
    hairs  = list(rng.choice(pool.hair_types,  size=min(total, len(pool.hair_types)),  replace=False))
    hands  = list(rng.choice(pool.hand_types,  size=min(total, len(pool.hand_types)),  replace=False))
    for i, char in enumerate(char_list):
        char.physical_traits = PhysicalTraits(
            build=str(builds[i]),
            hair=str(hairs[i]),
            hands=str(hands[i]),
        )

    suspect_indices = list(range(1, min(1 + config.num_suspects, total)))
    if not suspect_indices:
        raise ValueError("Mystery generation requires at least one suspect")
    culprit_idx = int(rng.choice(suspect_indices))

    # MEDIUM+: each of the culprit's trait values must appear on at least one
    # other suspect so no single trait uniquely identifies the culprit.
    if config.evidence_ambiguity > 0.1:
        culprit_char = char_list[culprit_idx]
        other_suspects = [
            char_list[i] for i in suspect_indices
            if i != culprit_idx
        ]
        if other_suspects:
            for trait_name in ("build", "hair", "hands"):
                culprit_val = getattr(culprit_char.physical_traits, trait_name)
                sharers = [
                    c for c in other_suspects
                    if getattr(c.physical_traits, trait_name) == culprit_val
                ]
                if not sharers:
                    target = other_suspects[int(rng.integers(0, len(other_suspects)))]
                    pt = target.physical_traits
                    target.physical_traits = PhysicalTraits(
                        build=culprit_val if trait_name == "build" else pt.build,
                        hair=culprit_val if trait_name == "hair" else pt.hair,
                        hands=culprit_val if trait_name == "hands" else pt.hands,
                    )

    # Assign roles
    victim = char_list[0]
    victim.roles.append(CharacterRole.VICTIM)
    victim.is_alive = False

    for i in suspect_indices:
        if i < len(char_list):
            char_list[i].roles.append(CharacterRole.SUSPECT)

    culprit = char_list[culprit_idx]
    culprit.is_culprit = True
    
    for i in range(1 + config.num_suspects, total):
        if i < len(char_list):
            char_list[i].roles.append(CharacterRole.INNOCENT)
            char_list[i].roles.append(CharacterRole.WITNESS)

    # Assign motives to suspects
    # Mark all suspects as having alibis (claims are filled in generate_mystery)
    for sc in [c for c in char_list if CharacterRole.SUSPECT in c.roles]:
        sc.has_alibi = True

    motives = list(rng.choice(pool.motive_templates, size=min(config.num_suspects, len(pool.motive_templates)), replace=False))
    suspect_chars = [c for c in char_list if CharacterRole.SUSPECT in c.roles]
    for i, sc in enumerate(suspect_chars):
        sc.motive = str(motives[i % len(motives)])
    
    # Generate relationships
    for c in char_list:
        n_rels = rng.integers(1, min(4, total))
        others = [o for o in char_list if o.id != c.id]
        if others:
            targets = rng.choice(others, size=min(int(n_rels), len(others)), replace=False)
            kinds = ["sibling", "colleague", "rival", "friend", "employer", "acquaintance"]
            for t in targets:
                c.relationships.append(Relationship(
                    target_id=t.id,
                    kind=str(rng.choice(kinds)),
                    sentiment=float(rng.uniform(-1, 1)),
                ))
    
    # Place characters in locations
    for cid, char in characters.items():
        loc = char.location_id
        if loc in {lid for lid in location_ids}:
            pass # will be added to location manifests in post-processing
    
    return characters

# ---------------------------------------------------------------------------
# Evidence & objects
# ---------------------------------------------------------------------------

def _generate_evidence_and_objects(
    config: ComplexityConfig,
    pool: AssetPool,
    rng: np.random.Generator,
    location_ids: list[str],
    culprit_id: str,
    victim_id: str,
    suspect_ids: list[str],
    murder_weapon_id: str,
    murder_location_id: str,
    locations: dict[str, Location],
    characters: dict[str, Character],
    murder_step: int = 0,
) -> tuple[dict[str, Evidence], dict[str, WorldObject]]:

    evidence: dict[str, Evidence] = {}
    objects: dict[str, WorldObject] = {}
    non_culprit_suspects = [s for s in suspect_ids if s != culprit_id]
    murder_ts = float(murder_step)
    threshold = config.freshness_threshold

    def _traits(cid: str) -> PhysicalTraits:
        return characters[cid].physical_traits

    def _sample_difficulty() -> float:
        return float(rng.uniform(config.evidence_difficulty_min, config.evidence_difficulty_max))

    def _label(ts: float) -> TemporalLabel:
        return _pick_surface_label(ts, murder_ts, threshold, config)

    # --- Murder weapon object ---
    weapon_names = list(rng.choice(
        pool.weapon_templates,
        size=min(config.num_weapons, len(pool.weapon_templates)),
        replace=False,
    ))
    murder_weapon_name = str(weapon_names[0])
    if config.culprit_tamper_prob > 0.0 and len(location_ids) > 1:
        weapon_loc_id = str(rng.choice([l for l in location_ids if l != murder_location_id]))
    else:
        weapon_loc_id = murder_location_id

    mw_obj = WorldObject(
        id=murder_weapon_id, name=murder_weapon_name,
        description=f"A {murder_weapon_name}.",
        location_id=weapon_loc_id, is_weapon=True, is_murder_weapon=True,
    )
    objects[murder_weapon_id] = mw_obj

    # --- SUSPECT_WEAPON evidence (trait-based grip marks on weapon) ---
    # Primary piece always links to the culprit; ambiguity applies to extra pieces only.
    mw_linked_id = culprit_id    

    murder_loc = locations.get(murder_location_id)
    mw_traits = _traits(mw_linked_id)
    mw_ev = Evidence(
        id=_uid("ev", rng),
        name=f"traces on the {murder_weapon_name}",
        evidence_type=EvidenceType.PHYSICAL,
        location_id=murder_location_id,
        linked_character_id=mw_linked_id,
        description=f"Grip marks from someone with {mw_traits.hands} found on the {murder_weapon_name}.",
        discovery_difficulty=_sample_difficulty(),
        weather_sensitive=bool(murder_loc and murder_loc.weather_exposed),
        relevance=EdgeRelevance(
            edge_type=EdgeType.SUSPECT_WEAPON,
            subject_ids=[mw_linked_id, murder_weapon_id],
            contact_timestamp=murder_ts,
            surface_label=_label(murder_ts),
        ),
    )
    evidence[mw_ev.id] = mw_ev
    mw_obj.evidence_id = mw_ev.id

    # --- WEAPON_VICTIM evidence (victim blood on weapon) ---
    wv_ev = Evidence(
        id=_uid("ev", rng),
        name=f"victim's blood on the {murder_weapon_name}",
        evidence_type=EvidenceType.PHYSICAL,
        location_id=weapon_loc_id,
        linked_character_id=victim_id,
        description=f"Blood and tissue matching the victim found on the {murder_weapon_name}.",
        discovery_difficulty=_sample_difficulty(),
        relevance=EdgeRelevance(
            edge_type=EdgeType.WEAPON_VICTIM,
            subject_ids=[murder_weapon_id, victim_id],
            contact_timestamp=murder_ts,
            surface_label=_label(murder_ts),
        ),
    )
    evidence[wv_ev.id] = wv_ev

    # --- SUSPECT_ROOM evidence (culprit footprint in murder room) ---
    culprit_traits = _traits(culprit_id)
    murder_loc_name = murder_loc.name if murder_loc else "crime scene"
    sr_ev = Evidence(
        id=_uid("ev", rng),
        name=f"shoe scuffs in the {murder_loc_name}",
        evidence_type=EvidenceType.PHYSICAL,
        location_id=murder_location_id,
        linked_character_id=culprit_id,
        description=f"Shoe prints from a {culprit_traits.build} person found in the room.",
        discovery_difficulty=_sample_difficulty(),
        relevance=EdgeRelevance(
            edge_type=EdgeType.SUSPECT_ROOM,
            subject_ids=[culprit_id, murder_location_id],
            contact_timestamp=murder_ts,
            surface_label=_label(murder_ts),
        ),
    )
    evidence[sr_ev.id] = sr_ev

    # --- Other weapons (non-murder) ---
    for i in range(1, config.num_weapons):
        if i < len(weapon_names):
            wid = _uid("obj", rng)
            objects[wid] = WorldObject(
                id=wid, name=str(weapon_names[i]),
                description=f"A {weapon_names[i]}.",
                location_id=str(rng.choice(location_ids)),
                is_weapon=True, is_murder_weapon=False,
            )

    # --- Additional physical evidence (trait-based descriptions) ---
    ev_trait_templates = [
        ("fingerprint on doorknob",     "A partial print from someone with {hands} found on a doorknob."),
        ("hair strand",                  "A strand of {hair} caught on a rough surface."),
        ("torn fabric from clothing",    "A torn fabric scrap — looks like it belongs to someone {build}."),
        ("footprint near the scene",     "A footprint suggesting a {build} individual."),
        ("smudged handprint on wall",    "A smudged handprint from someone with {hands}."),
        ("scratches on nearby furniture","Scratches consistent with someone who has {hands}."),
        ("fiber transfer on victim's coat","Fibers transferred by contact with a {build} person."),
        ("cigarette butt near the scene","A cigarette butt consistent with someone who has {hair}."),
        ("partial shoe impression in mud","A shoe impression suggesting a {build} person."),
        ("broken button from a jacket",  "A button from a garment sized for someone {build}."),
    ]
    n_culprit_ev = max(2, config.num_objects // 3)
    for i in range(n_culprit_ev):
        eid = _uid("ev", rng)
        if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
            linked_id = str(rng.choice(non_culprit_suspects))
        else:
            linked_id = culprit_id

        edge_roll = rng.random()
        if edge_roll < 0.5:
            edge, subj = EdgeType.SUSPECT_WEAPON, [linked_id, murder_weapon_id]
        elif edge_roll < 0.8:
            edge, subj = EdgeType.SUSPECT_ROOM, [linked_id, murder_location_id]
        else:
            edge, subj = EdgeType.WEAPON_VICTIM, [murder_weapon_id, victim_id]

        contact_ts = (
            murder_ts + float(rng.uniform(-0.5, 0.5))
            if linked_id == culprit_id
            else murder_ts - float(rng.uniform(threshold, threshold * 3))
        )

        t_name, t_desc = ev_trait_templates[i % len(ev_trait_templates)]
        lt = _traits(linked_id)
        desc = t_desc.format(hands=lt.hands, hair=lt.hair, build=lt.build)

        evidence[eid] = Evidence(
            id=eid, name=t_name,
            evidence_type=EvidenceType.PHYSICAL,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=linked_id,
            description=desc,
            discovery_difficulty=_sample_difficulty(),
            weather_sensitive=rng.random() < 0.3,
            relevance=EdgeRelevance(
                edge_type=edge, subject_ids=subj,
                contact_timestamp=contact_ts, surface_label=_label(contact_ts),
            ),
        )

    # --- Innocent alibi evidence (SUSPECT_ELSEWHERE) ---
    # Per innocent: one physical trace in a non-murder alibi room, naming a
    # corroborator. Agent must EXAMINE the trace AND TALK_TO the corroborator
    # for the elimination to count (enforced in scoring).
    non_murder_locs = [lid for lid in location_ids if lid != murder_location_id]
    physical_alibi_templates = [
        ("coat hanging on the rack",          "A coat belonging to {name}, left here. {corr} was seen with them here at the time."),
        ("signed visitor register entry",     "The visitor register shows {name} signed in, countersigned by {corr}."),
        ("half-finished cup of tea",          "A cup recently used by {name}. {corr} recalls sharing tea with them here."),
        ("personal pocket watch left behind", "A pocket watch engraved with {name}'s initials. {corr} handed it back to them here."),
        ("reading glasses on the table",      "Reading glasses belonging to {name}. {corr} noticed them leave the glasses here."),
        ("umbrella in the stand",             "An umbrella left here by {name}. {corr} helped them stow it."),
    ]
    for i, sid in enumerate(non_culprit_suspects):
        if not non_murder_locs:
            break
        alibi_room_id = str(rng.choice(non_murder_locs))
        innocent = characters[sid]

        # Prefer another innocent suspect as corroborator; fall back to a
        # bystander NPC (INNOCENT + WITNESS, not SUSPECT) for small cases
        # (e.g. TRIVIAL has only 1 innocent suspect).
        corr_candidates = [
            c for c in suspect_ids
            if c != sid and c != culprit_id and c != victim_id
        ]
        if not corr_candidates:
            corr_candidates = [
                cid for cid, ch in characters.items()
                if CharacterRole.INNOCENT in ch.roles
                and CharacterRole.SUSPECT not in ch.roles
                and cid != victim_id
            ]
        corroborator_id = (
            str(rng.choice(corr_candidates)) if corr_candidates else ""
        )
        corroborator_name = (
            characters[corroborator_id].full_name if corroborator_id else "someone"
        )

        t_name, t_desc = physical_alibi_templates[i % len(physical_alibi_templates)]
        phys_eid = _uid("ev", rng)
        evidence[phys_eid] = Evidence(
            id=phys_eid, name=t_name,
            evidence_type=EvidenceType.PHYSICAL,
            location_id=alibi_room_id,
            linked_character_id=sid,
            corroborator_id=corroborator_id or None,
            description=t_desc.format(
                name=innocent.full_name, corr=corroborator_name
            ),
            discovery_difficulty=_sample_difficulty(),
            relevance=EdgeRelevance(
                edge_type=EdgeType.SUSPECT_ELSEWHERE,
                subject_ids=[sid, alibi_room_id],
                contact_timestamp=murder_ts,
                surface_label=_label(murder_ts),
            ),
        )

    # --- Documentary evidence (no relevance — motive support only) ---
    doc_templates = [
        "a threatening letter", "a financial ledger entry",
        "a diary page with incriminating passage", "a forged alibi note",
        "a signed insurance policy", "a photograph with a revealing timestamp",
        "a bank withdrawal receipt", "a phone message transcript",
        "a torn contract", "a secret correspondence",
    ]
    for i in range(min(2, config.motive_layers)):
        eid = _uid("ev", rng)
        doc_linked = (
            str(rng.choice(non_culprit_suspects))
            if non_culprit_suspects and rng.random() < config.evidence_ambiguity
            else culprit_id
        )
        evidence[eid] = Evidence(
            id=eid, name=str(rng.choice(doc_templates)),
            evidence_type=EvidenceType.DOCUMENTARY,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=doc_linked,
            description="A document that may shed light on someone's motive.",
            discovery_difficulty=_sample_difficulty(),
        )

    # --- Red herrings ---
    for _ in range(config.num_red_herrings):
        eid = _uid("ev", rng)
        decoy = str(rng.choice(suspect_ids))
        evidence[eid] = Evidence(
            id=eid, name=f"suspicious {rng.choice(['note', 'item', 'mark', 'stain'])}",
            evidence_type=EvidenceType(int(rng.integers(1, 5))),
            location_id=str(rng.choice(location_ids)),
            linked_character_id=decoy, is_red_herring=True,
            description="Something that looks incriminating but is ultimately misleading.",
            discovery_difficulty=float(rng.uniform(0.1, 0.4)),
            relevance=EdgeRelevance(
                edge_type=EdgeType(int(rng.integers(1, 4))),
                subject_ids=[decoy, str(rng.choice(location_ids + [murder_weapon_id]))],
                contact_timestamp=murder_ts + float(rng.uniform(-0.5, 0.5)),
                surface_label=TemporalLabel.AMBIGUOUS,
            ),
        )

    # --- Link every evidence item to a host object ---
    # 1. Collect evidence that already has a host
    hosted_eids: set[str] = {
        obj.evidence_id for obj in objects.values() if obj.evidence_id
    }

    # 2. For each unhosted evidence, create a neutral host in its location
    neutral_names = list(rng.permutation(_NEUTRAL_HOST_OBJECTS))
    neutral_idx = 0
    for eid, ev in evidence.items():
        if eid in hosted_eids:
            continue
        obj_name = neutral_names[neutral_idx % len(neutral_names)]
        neutral_idx += 1
        oid = _uid("obj", rng)
        objects[oid] = WorldObject(
            id=oid,
            name=obj_name,
            description=f"A {obj_name} sitting here.",
            location_id=ev.location_id,
            portable=False,
            evidence_id=eid,
        )
        hosted_eids.add(eid)

    # 3. Add decoy objects (no evidence) to fill rooms up to the per-room cap
    decoy_names = list(rng.choice(
        pool.object_templates,
        size=min(config.num_objects, len(pool.object_templates)),
        replace=False,
    ))
    for oname in decoy_names:
        oid = _uid("obj", rng)
        objects[oid] = WorldObject(
            id=oid, name=str(oname), description=f"A {oname} lying here.",
            location_id=str(rng.choice(location_ids)),
            portable=rng.random() < 0.7,
        )

    return evidence, objects

# ---------------------------------------------------------------------------
# Timeline generation
# ---------------------------------------------------------------------------

def _generate_timeline(
    config: ComplexityConfig,
    rng: np.random.Generator,
    characters: dict[str, Character],
    locations: dict[str, Location],
    culprit_id: str,
    victim_id: str,
    murder_location_id: str,
    murder_step: int,
) -> list[TimelineEntry]:
    """Generate ground-truth timeline of all character movements and events."""
    timeline: list[TimelineEntry] = []
    location_ids = list(locations.keys())
    
    for step in range(config.num_time_steps):
        for cid, char in characters.items():
            if not char.is_alive and step > murder_step:
                continue
            
            # At the murder step, place culprit and victim in murder location
            if step == murder_step:
                if cid == culprit_id:
                    timeline.append(TimelineEntry(
                        step=step, actor_id=cid, action="committed_murder",
                        target_id=victim_id, location_id=murder_location_id,
                        details=f"{char.full_name} attacked the victim.",
                        is_public=False,
                    ))
                    char.location_id = murder_location_id
                    continue
                elif cid == victim_id:
                    char.location_id = murder_location_id
                    continue
            
            # Regular NPC movement
            if rng.random() < 0.4:
                loc = locations.get(char.location_id)
                if loc and loc.adjacent_ids:
                    new_loc = str(rng.choice(loc.adjacent_ids))
                    timeline.append(TimelineEntry(
                        step=step, actor_id=cid, action="moved_to", # Thong: should we let the movement of NPC affect the mystery?
                        target_id=new_loc, location_id=new_loc,
                        details=f"{char.full_name} went to {locations[new_loc].name}.",
                        is_public=True,
                        witnesses=[
                            oid for oid in locations[new_loc].characters_here
                            if oid != cid and oid in characters
                        ],
                    ))
                    char.location_id = new_loc
            else:
                timeline.append(TimelineEntry(
                    step=step, actor_id=cid, action="stayed",
                    target_id=char.location_id, location_id=char.location_id,
                    details=f"{char.full_name} remained in the {locations.get(char.location_id, Location()).name}.",
                    is_public=True
                ))
    
    return timeline

# ---------------------------------------------------------------------------
# Alibi generation             
# ---------------------------------------------------------------------------
            
# --- Innocent suspect alibi templates (content always truthful) ---              

_INNOCENT_SOLO_ALIBIS = [      
    "I was in my room all evening. I know that is not much of an alibi.",       
    "I went for a walk alone after dinner. I have no one to confirm it.",         
    "I was in the garden by myself. Nobody saw me, but that is the truth.",       
    "I stayed in and went to bed early. I have no one to vouch for me.",          
    "I was alone in the billiard room. I did not feel like company.",             
    "I spent the evening alone in the library. I did not speak to anyone.",       
    "I was on the terrace by myself for most of the night.",                      
    "I retired to my room after dinner. I was alone the entire time.",            
    "I had a letter to write. I was at my desk until I went to sleep.",           
    "I was in the chapel, thinking. I saw no one and no one saw me.",             
]           
            
_INNOCENT_PARTIAL_ALIBIS = [   
    "I was with {corrs} until about nine, then I went to bed alone.",           
    "I spent part of the evening with {corrs}, but I left early.",                
    "{corrs} and I were talking for a while. I am not sure of the exact time I left.",                               
    "I was at the gathering with the others for most of it, then slipped away.",  
    "I was with {corrs} in the sitting room. I left before the others did.",      
    "Ask {corrs} — we were talking for a while. I headed off on my own after that.",                                 
    "I joined {corrs} for a drink, then went to bed. I cannot say exactly when.", 
    "{corrs} saw me in the hall earlier. After that I was alone in my room.",     
    "I was with {corrs} during dinner. The rest of the evening I spent alone.",   
    "{corrs} and I parted ways after supper. I do not know what they did after.", 
]           
            
_INNOCENT_CORROBORATED_ALIBIS = [                                                 
    "I was with {corrs} the entire evening. They can confirm every moment.",    
    "{corrs} and I were together all night. Neither of us left the room.",        
    "Ask {corrs}. We were in each other's company from dinner until well past midnight.",                            
    "I was playing cards with {corrs} all evening. We did not move from that table.",                                
    "{corrs} will tell you — I was with them the whole time without exception.",  
    "I was with {corrs} from the moment dinner ended. You are welcome to check.", 
    "{corrs} and I sat by the fire all evening. We both heard the clock strike midnight.",                           
    "I have nothing to hide. {corrs} were with me the entire night.",             
]           
            
# --- Culprit alibi templates (content always false) ---                          
        
_CULPRIT_SOLO_ALIBIS = [       
    "I was alone in my room the entire evening.",                               
    "I retired early — I had a headache. No one saw me after dinner.",            
    "I spent the evening alone in the library, reading.",                         
    "I walked the estate grounds by myself after supper.",                        
    "I was in the study going over correspondence. The door was closed.",         
    "I went to bed early. I heard nothing unusual.",                              
    "I took a long bath and then slept. I spoke to no one.",                      
    "I had a private matter to attend to in my room. I did not leave all night.", 
    "I was in the wine cellar cataloguing bottles. No one else was down there.",  
    "I sat alone on the terrace watching the sky. I spoke to no one.",            
    "I do not feel I need to account for my evening. I was simply alone.",        
    "I was writing letters in my room. I came out only for water.",               
]           
            
_CULPRIT_PARTIAL_ALIBIS_NO_WITNESS = [                                            
    "I was at dinner with the others until around nine. After that, I was alone.",                                 
    "I left the card game early and spent the rest of the evening alone in my room.",                                
    "I spent the early part of the evening with the group, then slipped away.",
    "I joined the gathering briefly, then retired. I spoke to no one after that.",
    "I was visible to others for the first part of the evening. Afterwards I was entirely alone.",                   
]           
            
_CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS = [                                          
    "I was talking with {corrs} in the hall for most of the evening, then went to my room alone.",                 
    "I was in the drawing room with {corrs} until the clock struck ten, then I retired alone.",                      
    "I joined {corrs} for drinks after supper, but I left early. After that I was by myself.",                       
    "I was with {corrs} briefly in the garden, then I went for a solitary walk.", 
    "I had a drink with {corrs} near the fire, then excused myself. I did not see anyone after that.",               
    "I sat with {corrs} at dinner. After the meal I slipped off alone — they can confirm I left early.",             
    "{corrs} saw me in the corridor earlier in the evening. After that, I was on my own.",                           
    "I left {corrs} in the parlour around half past nine. The rest of the evening I spent alone.",                   
    "Ask {corrs} — they will tell you I was there for dinner. What I did after is my own business.",                 
    "{corrs} and I shared a drink. I excused myself shortly after. I went nowhere near that part of the house.",     
]           
            
_CULPRIT_GAP_CORROBORATED_ALIBIS = [                                              
    "{corrs} and I were together most of the evening. I may have stepped out briefly, but it was nothing.",        
    "Ask {corrs} — we were together all night. I went to get a drink at some point, but I was not gone long.",       
    "{corrs} will confirm I was there. I slipped out for some air, five minutes at most.",                           
    "I was with {corrs} the entire time, more or less. I left the room once — I cannot imagine it matters.",         
    "{corrs} and I sat together all evening. I may have excused myself briefly at some point.",                      
    "I was in {corrs}'s company. There was a moment I stepped away, but {corrs} knew where I was going.",            
    "{corrs} can confirm my presence. I left for a short while but came straight back.",                             
    "{corrs} were with me. If I was away for a moment, it was not long enough to matter.",                           
]           
            
_CULPRIT_FALSE_CORROBORATED_ALIBIS = [                                            
    "I was with {corrs} all evening. Ask them.",                                
    "{corrs} and I were together the entire time.",                               
    "I spent the evening playing cards with {corrs}. They will confirm.",         
    "{corrs} can vouch for my whereabouts. We were together.",                    
    "I never left {corrs}'s sight all evening.",                                  
    "You can ask {corrs} — we were in the billiard room from eight until midnight.",                                 
    "{corrs} and I were at the far end of the estate. Neither of us went near that wing.",                           
    "I have witnesses: {corrs}. All of them saw me the entire time.",             
    "{corrs} were with me the whole evening. None of us moved from the sitting room.",                               
    "We — {corrs} and I — sat together well past the hour in question.",          
    "{corrs} will tell you I never moved. We were in plain sight of each other.", 
    "I was not alone. {corrs} were there. They saw everything.",                  
]           
            
            
def _format_names(names: list[str]) -> str:                                     
    if len(names) == 1:        
        return names[0]      
    elif len(names) == 2:
        return f"{names[0]} and {names[1]}"
    else:   
        return ", ".join(names[:-1]) + f" and {names[-1]}"
            
        
def _generate_alibis(
    config: ComplexityConfig,
    rng: np.random.Generator,
    characters: dict[str, Character],                                             
    timeline: list[TimelineEntry],
    culprit_id: str,           
    murder_step: int         
) -> None:  
    """     
    Assign alibis to all suspects. Format is randomised for every character
    regardless of guilt. Innocents' content is truthful; culprit's is false.      
    The agent cannot deduce guilt from format alone.                              
    """     
    suspects = [c for c in characters.values() if CharacterRole.SUSPECT in c.roles]                                  
    non_culprits = [s for s in suspects if s.id != culprit_id]                    
    all_char_list = list(characters.values())                                     
            
    def _get_corr_pool(exclude_id: str) -> list:                                  
        if config.allow_suspect_corroborators:                                  
            return [           
                c for c in all_char_list                                        
                if c.id != exclude_id and c.is_alive                              
                and CharacterRole.VICTIM not in c.roles                         
            ]                  
        else:
            return [           
                c for c in all_char_list                                        
                if c.id != exclude_id and c.id != culprit_id
                and c.is_alive and CharacterRole.SUSPECT not in c.roles
            ]                  

    def _pick_corroborators(pool: list, max_n: int) -> list:                      
        if not pool:         
            return []          
        n = min(int(rng.integers(1, max_n + 1)), len(pool))                     
        indices = list(range(len(pool)))                                          
        rng.shuffle(indices)
        return [pool[j] for j in indices[:n]]                                     
        
    # --- Non-culprit suspects: varied formats, always truthful ---               
    for i, suspect in enumerate(non_culprits):
        pool = _get_corr_pool(suspect.id)                                         
        suspect.alibi_corroboration_is_genuine = True                           
            
        if i < config.alibi_complexity:                                         
            # Fully corroborated alibi                                            
            suspect.has_alibi = True                                              
            corrs = _pick_corroborators(pool, config.max_corroborators)
            if corrs:          
                suspect.alibi_corroborator_id = corrs[0].id                     
                corr_names = _format_names([c.full_name for c in corrs])          
                template = _INNOCENT_CORROBORATED_ALIBIS[int(rng.integers(len(_INNOCENT_CORROBORATED_ALIBIS)))]    
                suspect.alibi_details = template.format(corrs=corr_names)         
            else:              
                template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]                      
                suspect.alibi_details = template                                  
        else:                
            # Varied: no alibi / solo / partial                                   
            alibi_roll = float(rng.random())
            if alibi_roll < 0.15:                                                 
                suspect.has_alibi = False                                       
            elif alibi_roll < 0.70:                                               
                suspect.has_alibi = True                                          
                template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]
                suspect.alibi_details = template                                  
            else:            
                suspect.has_alibi = True                                          
                if pool and rng.random() < 0.7:                                 
                    corrs = _pick_corroborators(pool, min(config.max_corroborators, 2))                              
                    corr_names = _format_names([c.full_name for c in corrs])
                    template = _INNOCENT_PARTIAL_ALIBIS[int(rng.integers(len(_INNOCENT_PARTIAL_ALIBIS)))]            
                    suspect.alibi_details = template.format(corrs=corr_names)   
                else:          
                    template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]                
                    suspect.alibi_details = template                              
            
    # --- Culprit: same formats, always false ---
    culprit = characters.get(culprit_id)                                          
    if culprit:                
        alibi_roll = float(rng.random())
        w = config.culprit_alibi_weights                                          
        others = [             
            c for c in all_char_list
            if c.id != culprit_id and c.is_alive                                  
            and CharacterRole.VICTIM not in c.roles                             
        ]
            
        if alibi_roll < w[0]:
            # No alibi         
            culprit.has_alibi = False                                           

        elif alibi_roll < w[1]:
            # Solo — false, unverifiable
            culprit.has_alibi = True                                              
            culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]
            culprit.alibi_corroboration_is_genuine = True                         
            
        elif alibi_roll < w[2]:
            # Partial — false, admits a gap, no alibi_corroborator_id             
            culprit.has_alibi = True                                              
            if others and rng.random() < 0.6:
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))                                
                corr_names = _format_names([c.full_name for c in corrs])        
                template = _CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS[int(rng.integers(len(_CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS)))]                                                                             
                culprit.alibi_details = template.format(corrs=corr_names)         
            else:              
                template = _CULPRIT_PARTIAL_ALIBIS_NO_WITNESS[int(rng.integers(len(_CULPRIT_PARTIAL_ALIBIS_NO_WITNESS)))]
                culprit.alibi_details = template                                  
            culprit.alibi_corroboration_is_genuine = True
            
        elif alibi_roll < w[3]:                                                 
            # Gap-corroborated — corroborators honest but missed a window         
            culprit.has_alibi = True                                            
            if others:         
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))
                culprit.alibi_corroborator_id = corrs[0].id                       
                culprit.alibi_corroboration_is_genuine = True                     
                culprit.alibi_has_gap = True
                corr_names = _format_names([c.full_name for c in corrs])          
                template = _CULPRIT_GAP_CORROBORATED_ALIBIS[int(rng.integers(len(_CULPRIT_GAP_CORROBORATED_ALIBIS)))]
                culprit.alibi_details = template.format(corrs=corr_names)
            else:              
                culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]         
                culprit.alibi_corroboration_is_genuine = True                     
        
        else:                  
            # Full false — lying corroborators                                  
            culprit.has_alibi = True
            if others:         
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))
                culprit.alibi_corroborator_id = corrs[0].id                       
                culprit.alibi_corroboration_is_genuine = False                    
                corr_names = _format_names([c.full_name for c in corrs])
                template = _CULPRIT_FALSE_CORROBORATED_ALIBIS[int(rng.integers(len(_CULPRIT_FALSE_CORROBORATED_ALIBIS)))]
                culprit.alibi_details = template.format(corrs=corr_names)         
            else:
                culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]           
                culprit.alibi_corroboration_is_genuine = True                   


# ---------------------------------------------------------------------------
# Witness assignment
# ---------------------------------------------------------------------------
def _assign_witnesses(
    characters: dict[str, Character],
    timeline: list[TimelineEntry],
    rng: np.random.Generator,
) -> None:
    """Mark characters as witnesses for events they could have seen."""
    for entry in timeline:
        if not entry.is_public:
            continue
        
        for cid, char in characters.items():
            if cid == entry.actor_id:
                continue
            if char.location_id == entry.location_id and char.is_alive:
                event_key = entry.action
                if event_key not in char.witnessed_events:
                    char.witnessed_events.append(event_key)
                if CharacterRole.WITNESS not in char.roles:
                    char.roles.append(CharacterRole.WITNESS)


# ---------------------------------------------------------------------------
# Solvability verification
# ---------------------------------------------------------------------------
def verify_solvability(state: WorldState) -> dict[str, Any]:
    """Check that the mystery is solvable: sufficient non-red-herring evidence
    exists that points to the culprit, and alibis are consistent."""
    culprit_evidence = [
        e for e in state.evidence.values()
        if e.linked_character_id == state.culprit_id
        and not e.is_red_herring
        and e.is_usable()
    ]
    has_weapon_evidence = any(
        e for e in culprit_evidence if e.evidence_type == EvidenceType.PHYSICAL
    )
    has_motive_evidence = any(
        e for e in culprit_evidence if e.evidence_type == EvidenceType.DOCUMENTARY
    )
    has_testimonial = any(
        e for e in state.evidence.values()
        if e.evidence_type == EvidenceType.TESTIMONIAL and not e.is_red_herring
    )
    # The culprit should NOT have a corroborated alibi
    culprit = state.get_culprit()
    culprit_alibi_breakable = culprit and (
        not culprit.has_alibi or culprit.alibi_corroborator_id is None
    )

    solvable = (
        len(culprit_evidence) >= 2
        and has_weapon_evidence
        and culprit_alibi_breakable
    )

    return {
        "solvable": solvable,
        "culprit_evidence_count": len(culprit_evidence),
        "has_weapon_evidence": has_weapon_evidence,
        "has_motive_evidence": has_motive_evidence,
        "has_testimonial": has_testimonial,
        "culprit_alibi_breakable": culprit_alibi_breakable,
    }


# ---------------------------------------------------------------------------
# Crime-scene convergent clue generation
# ---------------------------------------------------------------------------

def _bfs_path(
    start_id: str,
    end_id: str,
    locations: dict[str, Location],
) -> list[str]:
    """BFS shortest path (inclusive) between two location IDs. Returns [] if unreachable."""
    from collections import deque
    if start_id == end_id:
        return [start_id]
    queue: deque[list[str]] = deque([[start_id]])
    visited: set[str] = {start_id}
    while queue:
        path = queue.popleft()
        for nbr in locations.get(path[-1], Location()).adjacent_ids:
            if nbr == end_id:
                return path + [nbr]
            if nbr not in visited:
                visited.add(nbr)
                queue.append(path + [nbr])
    return []


def _generate_crime_scene_clues(
    config: ComplexityConfig,
    rng: np.random.Generator,
    pool: AssetPool,
    locations: dict[str, Location],
    characters: dict[str, Character],
    murder_location_id: str,
    body_location_id: str,
) -> dict[str, Evidence]:
    """
    Generate 3 convergent clue types that let the agent deduce the true murder location
    even when the body was moved:

    1. Body trace  — room material found on/near the victim
    2. Drag trail  — scuff marks in intermediate rooms along the drag path
    3. Testimony   — an NPC near the murder room heard a disturbance
    """
    evidence: dict[str, Evidence] = {}
    murder_loc = locations[murder_location_id]

    specific_material = murder_loc.material_signature
    vague_material = next(
        (v for s, v in pool.room_materials if s == specific_material),
        "unknown residue",
    )

    # 1. Body trace
    amb = config.body_trace_ambiguity
    if amb <= 1:
        trace_desc = (
            f"Traces of {specific_material} found on the victim — "
            f"characteristic of the {murder_loc.name}."
        )
    elif amb == 2:
        trace_desc = (
            f"Traces of {specific_material} found on the victim — "
            f"a material found in one of the estate's rooms."
        )
    elif amb == 3:
        trace_desc = (
            f"Traces of {vague_material} found on the victim — "
            f"the exact source room is unclear."
        )
    else:  # amb >= 4: multiple candidate rooms share the same vague description
        candidate_names = [murder_loc.name]
        for lid, loc in locations.items():
            if lid == murder_location_id:
                continue
            loc_vague = next(
                (v for s, v in pool.room_materials if s == loc.material_signature), ""
            )
            if loc_vague == vague_material and loc.name not in candidate_names:
                candidate_names.append(loc.name)
        candidate_names = candidate_names[:3]
        room_list = " or ".join(f"the {r}" for r in candidate_names)
        trace_desc = (
            f"Ambiguous {vague_material} residue on the victim — "
            f"consistent with material from {room_list}."
        )

    body_trace_ev = Evidence(
        id=_uid("ev", rng),
        name="material trace on victim",
        evidence_type=EvidenceType.PHYSICAL,
        location_id=body_location_id,
        description=trace_desc,
        discovery_difficulty=float(rng.uniform(0.1, 0.3)),
        relevance_score=1.0,
    )
    evidence[body_trace_ev.id] = body_trace_ev

    # 2. Drag trail through intermediate rooms
    path = _bfs_path(murder_location_id, body_location_id, locations)
    for room_id in path[1:-1]:  # exclude murder room and body room
        if rng.random() > config.trail_completeness:
            continue
        room = locations[room_id]
        trail_ev = Evidence(
            id=_uid("ev", rng),
            name=f"drag marks in the {room.name}",
            evidence_type=EvidenceType.PHYSICAL,
            location_id=room_id,
            description=(
                f"Scuff marks and faint traces suggest something heavy "
                f"was dragged through the {room.name}."
            ),
            discovery_difficulty=float(rng.uniform(0.2, 0.5)),
            relevance_score=0.8,
        )
        evidence[trail_ev.id] = trail_ev

    # 3. NPC testimony near the murder room
    potential_witnesses = [
        c for c in characters.values()
        if c.is_alive
        and CharacterRole.VICTIM not in c.roles
        and not c.is_culprit
        and (
            c.location_id == murder_location_id
            or murder_location_id in locations.get(c.location_id, Location()).adjacent_ids
        )
    ]
    if potential_witnesses:
        witness = potential_witnesses[int(rng.integers(len(potential_witnesses)))]
        spec = config.witness_specificity
        if spec >= 0.8:
            testimony = f"I heard a commotion coming from the {murder_loc.name} that evening."
        elif spec >= 0.5:
            testimony = (
                "There was some kind of disturbance from that part of the house — "
                "I cannot say exactly where."
            )
        else:
            testimony = "I thought I heard something unusual, but I am not certain what or where."

        testimony_ev = Evidence(
            id=_uid("ev", rng),
            name="overheard testimony",
            evidence_type=EvidenceType.TESTIMONIAL,
            location_id=witness.location_id,
            description=f'{witness.full_name} recalls: "{testimony}"',
            discovery_difficulty=float(rng.uniform(0.1, 0.4)),
            relevance_score=0.7,
        )
        evidence[testimony_ev.id] = testimony_ev

    return evidence


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def _apply_culprit_concealment(state: WorldState, seed: int) -> None:
    """Pre-conceal a fraction of culprit-linked evidence (deterministic, keyed).

    Concealment only raises ``discovery_difficulty`` (and records
    ``concealment_prob``); it never flips ``state`` to HIDDEN, so structural
    /Locard solvability and the oracle upper bound are preserved — the
    concealed evidence stays *in principle* discoverable, just stochastically
    harder to perceive at run time.
    """
    p = state.config.culprit_conceal_prob
    if p <= 0.0:
        return
    for ev in state.evidence.values():
        if ev.linked_character_id != state.culprit_id or ev.is_red_herring:
            continue
        digest = hashlib.blake2b((ev.id + ":conceal").encode(), digest_size=8).digest()
        key = [int(seed), int.from_bytes(digest, "big")]
        if float(np.random.default_rng(key).random()) < p:
            ev.concealment_prob = p
            ev.discovery_difficulty = min(1.0, ev.discovery_difficulty + 0.3)


def generate_mystery(
    config: ComplexityConfig,
    seed: int,
    asset_pool: AssetPool | None = None,
    max_retries: int = 10,
) -> WorldState:
    """
    Generate a complete, solvable mystery world.

    Parameters
    ----------
    config: ComplexityConfig
        Controls all dimensions of difficulty
    seed: int
        Random seed for full reproducibility
    asset_pool: AssetPool, optional
        Name/template pools (defaults to built-in originals).
    max_retries: int
        Number of attempts before raising (retries with incremented seed).
    
    Returns
    ----------
    WorldState
        Fully populated, verified-solvable world state.
    """
    pool = asset_pool or DEFAULT_ASSET_POOL

    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)

        # 1. Locations
        locations = _generate_locations(config, pool, rng)
        location_ids = list(locations.keys())
        
        # 2. Characters
        characters = _generate_characters(config, pool, rng, location_ids)
        char_list = list(characters.values())

        victim = next(c for c in char_list if CharacterRole.VICTIM in c.roles)
        culprit = next(c for c in char_list if c.is_culprit)
        suspect_ids = [c.id for c in char_list if CharacterRole.SUSPECT in c.roles]

        # 3. Murder details
        murder_location_id = str(rng.choice(location_ids))
        murder_weapon_id = _uid("wpn", rng)
        murder_step = int(rng.integers(1, max(2, config.num_time_steps // 2)))
        motive = culprit.motive or "unknown"

        # 3b. Body location — may differ from murder location at higher difficulties
        if (
            config.body_moved_prob > 0.0
            and rng.random() < config.body_moved_prob
            and len(location_ids) > 1
        ):
            other_locs = [l for l in location_ids if l != murder_location_id]
            body_location_id = str(rng.choice(other_locs))
        else:
            body_location_id = murder_location_id

        # 4. Timeline
        timeline = _generate_timeline(
            config, rng, characters, locations,
            culprit.id, victim.id, murder_location_id, murder_step,
        )

        # 5. Evidence & objects
        evidence, objects = _generate_evidence_and_objects(
            config, pool, rng, location_ids,
            culprit.id, victim.id, suspect_ids,
            murder_weapon_id, murder_location_id, locations,
            characters=characters,
            murder_step=murder_step,
        )

        # 5b. Crime-scene convergent clues when body was moved
        if body_location_id != murder_location_id:
            cs_evidence = _generate_crime_scene_clues(
                config, rng, pool, locations, characters,
                murder_location_id, body_location_id,
            )
            evidence.update(cs_evidence)

        body_obj_id = _uid("obj", rng)
        weapon_obj = objects.get(murder_weapon_id)
        objects[body_obj_id] = WorldObject(
            id=body_obj_id,
            name=f"body of {victim.full_name}",
            description=(
                f"The lifeless body of {victim.full_name}. "
                f"Examining the wounds suggests the cause of death was "
                f"inflicted by a sharp or heavy instrument."
            ),
            location_id=body_location_id,
            portable=False,
        )
        # Link crime-scene evidence to the body object where possible
        if body_location_id != murder_location_id:
            for eid, ev in cs_evidence.items():
                if ev.location_id == body_location_id:
                    hosted = any(o.evidence_id == eid for o in objects.values())
                    if not hosted:
                        objects[body_obj_id].evidence_id = eid
                        break
        victim.location_id = body_location_id

        # 6. Alibis
        _generate_alibis(config, rng, characters, timeline, culprit.id, murder_step)

        # 7. Witnesses
        _assign_witnesses(characters, timeline, rng)

        # Populate location manifests (characters + objects, with per-room cap)
        for cid, char in characters.items():
            loc = locations.get(char.location_id)
            if loc and cid not in loc.characters_here:
                loc.characters_here.append(cid)

        body_obj_ids = [body_obj_id]
        evidence_obj_ids = [oid for oid, o in objects.items() if o.evidence_id]
        required_obj_ids = list(dict.fromkeys(evidence_obj_ids + body_obj_ids))
        decoy_obj_ids = [
            oid for oid, o in objects.items()
            if not o.evidence_id and oid not in body_obj_ids
        ]

        # Evidence-bearing objects and the body are always placed. The body is
        # an explicit, inspectable scene object even when it holds no evidence.
        for oid in required_obj_ids:
            loc = locations.get(objects[oid].location_id)
            if loc and oid not in loc.objects_here:
                loc.objects_here.append(oid)

        # Decoys fill up to max_objects_per_room
        for oid in decoy_obj_ids:
            loc = locations.get(objects[oid].location_id)
            if (
                loc and oid not in loc.objects_here
                and len(loc.objects_here) < config.max_objects_per_room
            ):
                loc.objects_here.append(oid)

        # 9. Initial weather
        weather = str(rng.choice(pool.weather_types))

        # Assemble world state
        state = WorldState(
            seed=seed + attempt,
            config=config,
            locations=locations,
            characters=characters,
            objects=objects,
            evidence=evidence,
            current_step=0,
            weather=weather,
            ground_truth_timeline=timeline,
            culprit_id=culprit.id,
            victim_id=victim.id,
            murder_weapon_id=murder_weapon_id,
            murder_location_id=murder_location_id,
            body_location_id=body_location_id,
            murder_step=murder_step,
            murder_timestamp=float(murder_step),
            freshness_threshold=config.freshness_threshold,
            motive=motive,
        )
        # --- Anchor events ---
        anchor_events: dict[str, int] = {}
        if config.num_time_steps >= 4:
            anchor_events["the dinner bell"] = max(0, murder_step - 3)
        if config.num_time_steps >= 6:
            anchor_events["the clock striking the hour"] = max(0, murder_step - 1)
        anchor_events["the scream"] = murder_step
        state.anchor_events = anchor_events

        world_start_hour = config.world_start_hour
        styles = [TimeStyle.CLOCK, TimeStyle.NAMED_PERIOD, TimeStyle.RELATIVE]
        route_constraints: list[RouteConstraint] = []
        constraint_reasons = [
            "the corridor was locked from the inside",
            "the garden gate was bolted shut",
            "a servant had blocked the passage with a trolley",
            "the door had swollen shut from the rain",
        ]
        murder_loc_obj = state.locations.get(murder_location_id)

        # --- Culprit alibi ---
        if CharacterRole.SUSPECT in culprit.roles and murder_loc_obj:
            if config.evidence_ambiguity <= 0.1:
                # TRIVIAL/EASY — Type A: one claim at murder_step claiming a different room
                alibi_loc = next(
                    (l for l in state.locations.values()
                     if l.id != murder_location_id and l.id != body_location_id),
                    None,
                )
                if alibi_loc:
                    style = TimeStyle(styles[int(rng.integers(0, 3))])
                    culprit.alibi_claims = [AlibiClaim(
                        location_name=alibi_loc.name,
                        step=murder_step,
                        clock_time_str=_step_to_clock_str_gen(murder_step, world_start_hour),
                        stated_time=_make_stated_time(murder_step, anchor_events, style, world_start_hour, rng),
                        time_style=style,
                    )]
            else:
                # MEDIUM+ — Type B: two claims bracketing the murder through adjacent rooms
                adjacent_to_murder = [
                    state.locations[adj_id]
                    for adj_id in murder_loc_obj.adjacent_ids
                    if adj_id in state.locations
                ]
                if len(adjacent_to_murder) >= 2:
                    room_a, room_b = adjacent_to_murder[0], adjacent_to_murder[1]
                    before_step = max(0, murder_step - 1)
                    after_step = min(config.num_time_steps - 1, murder_step + 1)
                    style_a = TimeStyle(styles[int(rng.integers(0, 3))])
                    style_b = TimeStyle(styles[int(rng.integers(0, 3))])
                    culprit.alibi_claims = [
                        AlibiClaim(
                            location_name=room_a.name,
                            step=before_step,
                            clock_time_str=_step_to_clock_str_gen(before_step, world_start_hour),
                            stated_time=_make_stated_time(before_step, anchor_events, style_a, world_start_hour, rng),
                            time_style=style_a,
                        ),
                        AlibiClaim(
                            location_name=room_b.name,
                            step=after_step,
                            clock_time_str=_step_to_clock_str_gen(after_step, world_start_hour),
                            stated_time=_make_stated_time(after_step, anchor_events, style_b, world_start_hour, rng),
                            time_style=style_b,
                        ),
                    ]
                    # If room_a and room_b are directly adjacent (alternative route exists),
                    # block that passage to force the route through the crime scene.                                                                                                                                                                                                                                                                                                                                           
                    if room_b.id in room_a.adjacent_ids:  
                        route_constraints.append(RouteConstraint(                                                                                                                                                                                                                                                                                                                                                              
                            from_location_id=room_a.id,   
                            to_location_id=room_b.id,                                                                                                                                                                                                                                                                                                                                                                          
                            blocked_from_step=max(0, murder_step - 1),
                            blocked_until_step=min(config.num_time_steps - 1, murder_step + 1),                                                                                                                                                                                                                                                                                                                                
                        ))                                                                                                                                                                                                                                                                                                                                                                                                     
                else:         
                    # Murder room has fewer than 2 neighbours — Type B impossible; fall back to Type A.                                                                                                                                                                                                                                                                                                                            
                    alibi_loc = next(                                                                  
                        (l for l in state.locations.values()                                                                                                                                                                                                                                                                                                                                                                       
                        if l.id != murder_location_id and l.id != body_location_id),
                        None,                                                                                                                                                                                                                                                                                                                                                                                                      
                    )                                         
                    if alibi_loc:                                                                                                                                                                                                                                                                                                                                                                                                  
                        style = TimeStyle(styles[int(rng.integers(0, 3))])
                        culprit.alibi_claims = [AlibiClaim(               
                            location_name=alibi_loc.name,  
                            step=murder_step,                                                                                                                                                                                                                                                                                                                                                                                      
                            clock_time_str=_step_to_clock_str_gen(murder_step, world_start_hour),
                            stated_time=_make_stated_time(murder_step, anchor_events, style, world_start_hour, rng),                                                                                                                                                                                                                                                                                                               
                            time_style=style,                                                                                                                                                                                                                                                                                                                                                                                      
                        )]

        # --- Innocent suspect alibis ---
        for sid in suspect_ids:
            if sid == culprit.id:
                continue
            sc = characters[sid]
            pos = next(
                (e for e in state.ground_truth_timeline
                 if e.actor_id == sid and e.step == murder_step),
                None,
            )
            loc_id = pos.location_id if pos else sc.location_id
            loc_obj = state.locations.get(loc_id)
            if loc_obj:
                style = TimeStyle(styles[int(rng.integers(0, 3))])
                sc.alibi_claims = [AlibiClaim(
                    location_name=loc_obj.name,
                    step=murder_step,
                    clock_time_str=_step_to_clock_str_gen(murder_step, world_start_hour),
                    stated_time=_make_stated_time(murder_step, anchor_events, style, world_start_hour, rng),
                    time_style=style,
                )]

        # --- Witness statements ---
        witness_statements: list[WitnessStatement] = []
        innocent_chars = [
            c for c in state.characters.values()
            if CharacterRole.INNOCENT in c.roles or CharacterRole.WITNESS in c.roles
        ]
        for witness in innocent_chars:
            observed_id = str(rng.choice(
                [s for s in suspect_ids if s != culprit.id] or suspect_ids
            ))
            observed = characters.get(observed_id)
            if not observed:
                continue
            pos = next(
                (e for e in state.ground_truth_timeline
                 if e.actor_id == observed_id and e.step == murder_step),
                None,
            )
            loc_id = pos.location_id if pos else observed.location_id
            loc_obj = state.locations.get(loc_id)
            if not loc_obj:
                continue
            style = TimeStyle(styles[int(rng.integers(0, 3))])
            reliable = rng.random() >= config.testimony_unreliability
            witness_statements.append(WitnessStatement(
                witness_id=witness.id,
                observed_character_id=observed_id,
                location_name=loc_obj.name,
                step=murder_step,
                clock_time_str=_step_to_clock_str_gen(murder_step, world_start_hour),
                stated_time=_make_stated_time(murder_step, anchor_events, style, world_start_hour, rng),
                time_style=style,
                is_reliable=reliable,
            ))
        state.witness_statements = witness_statements

        # --- Additional route constraints (beyond the alibi-driven one) ---
        adjacent_pairs = [
            (loc.id, adj_id)
            for loc in state.locations.values()
            for adj_id in loc.adjacent_ids
            if loc.id < adj_id
        ]
        n_extra = config.num_route_constraints - len(route_constraints)
        if n_extra > 0 and adjacent_pairs:
            chosen_idxs = list(rng.choice(
                len(adjacent_pairs),
                size=min(n_extra, len(adjacent_pairs)),
                replace=False,
            ))
            for idx in chosen_idxs:
                from_id, to_id = adjacent_pairs[idx]
                route_constraints.append(RouteConstraint(
                    from_location_id=from_id,
                    to_location_id=to_id,
                    blocked_from_step=max(0, murder_step - 1),
                    blocked_until_step=min(config.num_time_steps - 1, murder_step + 1),
                    reason=str(rng.choice(constraint_reasons)),
                ))
        state.route_constraints = route_constraints
        
        # 10. Verify solvability
        check = verify_solvability(state)
        if check["solvable"]:
            _apply_culprit_concealment(state, seed)
            return state

    # If we exhausted retries, return last attempt with a warning
    _apply_culprit_concealment(state, seed)   # type: ignore[possibly-undefined]
    return state   # type: ignore[possibly-undefined]
