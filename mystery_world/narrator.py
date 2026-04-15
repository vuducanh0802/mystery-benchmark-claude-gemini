"""
Narrator: renders world state observations as natural-language text.

This module converts structured WorldState snapshots into prose that an
LLM agent receives as its observation at each step. The narrator ensures
the agent only sees information available from its current vantage point 
(partial observability).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery_world.entities import CharacterRole, EvidenceState

if TYPE_CHECKING:
    from mystery_world.world import MysteryEnvironment
    

# ---------------------------------------------------------------------------
# Compositional briefing templates — seed selects one combination per slot.
# With 6 slots × 10 variants each = 1,000,000 unique briefings.
# ---------------------------------------------------------------------------

_TITLES = [
    "=== CASE BRIEFING ===",
    "=== INCIDENT REPORT ===",
    "=== CASE FILE ===",
    "=== HOMICIDE INVESTIGATION BRIEF ===",
    "=== DETECTIVE'S JOURNAL ===",
    "=== CRIMINAL INVESTIGATION DOSSIER ===",
    "=== PRIORITY DISPATCH ===",
    "=== FIELD INVESTIGATION LOG ===",
    "=== CLASSIFIED CASE SUMMARY ===",
    "=== UNSOLVED CASE INTAKE ===",
]

_CRIME_DESCRIPTIONS = [
    "A terrible crime has occurred. {victim} has been found dead in the {location}.",
    "They found {victim} face-down in the {location}, not a breath left.",
    "DISPATCH: Homicide confirmed. Victim identified as {victim}. Body discovered in the {location}.",
    "Another grim evening. {victim} was found lifeless in the {location}.",
    "Decedent: {victim}. Recovery site: the {location}.",
    "The body of {victim} has been recovered from the {location}. Foul play is certain.",
    "{victim} is dead. The body was discovered in the {location} under suspicious circumstances.",
    "A murder has been committed. {victim} lies cold in the {location}, and someone here is responsible.",
    "Late last night, {victim} was found slain in the {location}. No witnesses have come forward.",
    "The estate is in shock. {victim} has been killed — the body was found in the {location}.",
]

_TIME_DESCRIPTIONS = [                                                                                                                      
    "Time of death is estimated around {time_of_death}.",                                                                                   
    "The coroner puts the time of death near {time_of_death}.",                                                                             
    "Estimated time of death: {time_of_death}.",                                                                                            
    "By all accounts, death occurred around {time_of_death}.",
    "Preliminary time-of-death estimate: {time_of_death}.",                                                                                 
    "Witnesses place the last signs of life at roughly {time_of_death}.",
    "The medical examiner believes death occurred at approximately {time_of_death}.",                                                       
    "All evidence points to the killing happening around {time_of_death}.",                                                                 
    "According to forensic analysis, the victim died around {time_of_death}.",
    "{time_of_death} — that is when it happened, give or take.",                                                                            
]  

_SUSPECT_INTROS = [
    "Suspects: {suspects}.",
    "Persons of interest: {suspects}.",
    "Individuals flagged for questioning: {suspects}.",
    "I have my eye on several suspects: {suspects}.",
    "Suspect pool: {suspects}.",
    "The following individuals are under suspicion: {suspects}.",
    "These people had means, motive, or opportunity: {suspects}.",
    "Initial suspect list: {suspects}.",
    "Several names keep coming up: {suspects}.",
    "Nobody has been cleared yet. Primary suspects: {suspects}.",
]

_ROLE_AND_TASK = [
    [
        "You are a detective. Your task is to determine:",
        "  1. WHO committed the murder",
        "  2. WHAT weapon was used",
        "  3. WHERE the murder took place",
    ],
    [
        "You are the lead investigator. Determine:",
        "  1. WHO is responsible for this killing",
        "  2. WHAT weapon was used to carry it out",
        "  3. WHERE the crime actually took place",
    ],
    [
        "Objectives for responding detective:",
        "  1. Identify the PERPETRATOR",
        "  2. Identify the MURDER WEAPON",
        "  3. Confirm the CRIME SCENE location",
    ],
    [
        "I need to piece together three things:",
        "  1. WHO did this",
        "  2. WHAT weapon ended the victim's life",
        "  3. WHERE the act was committed",
    ],
    [
        "Required determinations:",
        "  1. IDENTITY of the perpetrator",
        "  2. WEAPON employed",
        "  3. LOCATION where the homicide occurred",
    ],
    [
        "Your job is to answer three questions:",
        "  1. WHO is the killer",
        "  2. WHAT was the murder weapon",
        "  3. WHERE did the murder happen",
    ],
    [
        "You have been called in to solve this case. Establish:",
        "  1. The GUILTY PARTY",
        "  2. The WEAPON used in the crime",
        "  3. The SCENE of the murder",
    ],
    [
        "Three unknowns remain before this case can be closed:",
        "  1. WHO — the identity of the murderer",
        "  2. WHAT — the weapon that was used",
        "  3. WHERE — the location of the killing",
    ],
    [
        "The chief wants answers to three questions:",
        "  1. WHO killed the victim",
        "  2. WHAT weapon was involved",
        "  3. WHERE the crime was committed",
    ],
    [
        "Before you can make an arrest, you need to know:",
        "  1. WHO is guilty",
        "  2. WHAT weapon they used",
        "  3. WHERE they did it",
    ],
]

_BUDGET_DESCRIPTIONS = [
    "You have a budget of {budget} actions.",
    "You have {budget} actions before the case goes cold.",
    "Action budget allocated: {budget}.",
    "I can afford {budget} investigative actions before I must draw my conclusion.",
    "Investigation budget: {budget} actions.",
    "You must work within a limit of {budget} actions.",
    "Make them count — you only get {budget} actions.",
    "Resources are limited. You have {budget} actions at your disposal.",
    "The department has authorized {budget} actions for this investigation.",
    "You are allowed exactly {budget} actions before you must make your accusation.",
]
_TIMING_NOTES = [
    "The night's events began at {start_clock}. Bear in mind: any two neighbouring rooms in this estate are some {step_min} minutes apart on foot.",                 
    "The household had been assembled since {start_clock}. Adjacent rooms throughout the estate are connected by passages of roughly {step_min} minutes' walk — a detail the attentive mind will not overlook.",                                                           
    "By the hall clock, the evening commenced at {start_clock}. You would do well to note, mon ami, that rooms adjoining one another are no more than {step_min} minutes on foot.",
    "It was {start_clock} when the company gathered. Each room's nearest neighbour lies some {step_min} minutes' walk away — in a house like this, that gap may prove everything.",
    "The affair began, as best we can determine, at {start_clock}. The estate's rooms are linked by corridors and passages, each roughly {step_min} minutes apart.", 
    "The guests had been confined to the estate since {start_clock}. Note well: no two adjacent rooms in this house are more than {step_min} minutes apart on foot.",
    "From {start_clock} onward, the household was sealed. A useful observation: each room is some {step_min} minutes' walk from those directly adjoining it.",       
    "The clock in the drawing room showed {start_clock} when the trouble began. The rooms here are close — neighbouring ones lie roughly {step_min} minutes apart on foot.",       
    "By all accounts the evening was under way by {start_clock}. In a house of this size, adjacent rooms are perhaps {step_min} minutes apart — a trifle, unless one is in a hurry.",
    "The sequence of events stretches back to {start_clock}. The house is not labyrinthine, but each step between neighbouring rooms demands some {step_min} minutes of travel.",  
]    
_ALL_SLOTS = [_TITLES, _CRIME_DESCRIPTIONS, _TIME_DESCRIPTIONS, _SUSPECT_INTROS, _ROLE_AND_TASK, _BUDGET_DESCRIPTIONS, _TIMING_NOTES]
_PRIMES = [1, 7, 13, 31, 47, 61, 79]


def _step_to_clock(step: int, world_start_hour: int, step_duration_minutes: int) -> str:
    """Convert a step index to a clock string using world config."""
    total_minutes = world_start_hour * 60 + step * step_duration_minutes
    total_minutes %= 24 * 60
    h, m = divmod(total_minutes, 60)
    period = "AM" if h < 12 else "PM"                               
    h12 = h % 12 or 12                                            
    return f"{h12}:{m:02d} {period}"

def render_initial_briefing(env: "MysteryEnvironment") -> str:
    """Opening scene description given to the agent as episode start."""
    state = env.state
    victim = state.get_victim()
    victim_name = victim.full_name if victim else "the victim"
    body_loc_id = state.body_location_id or state.murder_location_id
    body_loc = state.locations.get(body_loc_id)
    loc_name = body_loc.name if body_loc else "an unknown location"

    suspects = [
        c for c in state.characters.values()
        if CharacterRole.SUSPECT in c.roles and c.is_alive
    ]
    suspect_names = ", ".join(s.full_name for s in suspects)

    # Deterministically select one variant per slot from the seed.
    # Each slot uses a different prime multiplier to decorrelate choices.
    _PRIMES = [1, 7, 13, 31, 47, 61, 79]
    def _pick(slot_idx):
        pool = _ALL_SLOTS[slot_idx]
        return (state.seed * _PRIMES[slot_idx]) % len(pool)

    fmt = {
        "victim": victim_name,
        "location": loc_name,
        "time_of_death": _step_to_clock(
            state.murder_step,
            state.config.world_start_hour,
            state.config.step_duration_minutes,
        ),
        "suspects": suspect_names,
        "budget": state.config.max_agent_actions,
    }

    lines = [
        _TITLES[_pick(0)],
        _CRIME_DESCRIPTIONS[_pick(1)].format(**fmt),
        _TIME_DESCRIPTIONS[_pick(2)].format(**fmt),
        "",
        _SUSPECT_INTROS[_pick(3)].format(**fmt),
        "",
    ]
    lines.extend(_ROLE_AND_TASK[_pick(4)])
    lines.extend([
        "",
        _BUDGET_DESCRIPTIONS[_pick(5)].format(**fmt),
    ])

    # Action list (fixed across all styles — agents parse this)     
    _start_clock = _step_to_clock(0, state.config.world_start_hour, state.config.step_duration_minutes)                
    timing_note = _TIMING_NOTES[_pick(6)].format(                   
        start_clock=_start_clock,                                   
        step_min=state.config.step_duration_minutes,                
    )                       
    lines.extend([                                                
        "",
        timing_note,
        "",                 
        "Available actions:",
        "  EXAMINE_LOCATION                       — look around current room",                                         
        "  EXAMINE_OBJECT <name>                  — inspect a specific object",
        "  TALK_TO <name>                         — interrogate a character",  
        "  TAKE_OBJECT <name>                     — pick up a portable object",      
        "  CHECK_INVENTORY                        — review collected evidence",                                        
        "  ANALYZE <evidence_id>                  — assess how fresh a piece of evidence is",                          
        "  TRAVEL_TIME <from> <to> [at <time>]    — minimum travel time between two rooms",  
        "  CHECK_ROUTE <from> <to> <time>         — was the direct passage open at a given time?",                     
        "  WAIT                                   — pass time",     
        "  ACCUSE <suspect> <weapon> <location>   — make final accusation",
        "",                 
    ])     

    # Current location observation
    lines.append(env.observe_location())

    # Location map
    lines.append("")
    lines.append("=== ESTATE MAP ===")
    for lid, loc in state.locations.items():
        adj_names = [state.locations[a].name for a in loc.adjacent_ids if a in state.locations]
        lines.append(f"{loc.name} → {', '.join(adj_names)}")

    return "\n".join(lines)



def render_step_observation(env: "MysteryEnvironment", action_result_text: str) -> str:
    """Combine action result with ambient information for a step observation."""
    state = env.state
    parts = [action_result_text]

    # Ambient events the agent might notice
    recent_events = [
        e for e in state.event_log
        if e.step == state.current_step - 1 and e.agent_visible   # events from the step just processed
    ]
    for ev in recent_events:
        # Only show events the agent can perceive (same location or public)
        if ev.location_id == env.agent_location_id:
            if ev.event_type.name in ("NPC_MOVE", "NPC_INTERACTION"):
                parts.append(f"[You notice: {ev.description}]")
            elif ev.event_type.name == "WEATHER_CHANGE":
                parts.append(f"[The weather shifts: {ev.description}]")
        elif ev.event_type.name == "WEATHER_CHANGE":
            # Weather is globally observable for outdoor locations
            loc = env.get_current_location()
            if loc and loc.weather_exposed:
                parts.append(f"[Weather update: {ev.description}]")
    
    # Budget reminder
    remaining = env.budget_remaining
    if remaining <= 5:
        parts.append(f"[WARNING: Only {remaining} actions remaining. Consider making your accusation.]")
    
    return "\n".join(parts)


def render_evidence_summary(env: "MysteryEnvironment") -> str:
    """Render a structured summary of all discovered evidence."""
    state = env.state
    discovered = env._discovered_evidence
    if not discovered:
        return "No evidence collected yet."
    
    lines = ["=== EVIDENCE SUMMARY ==="]
    for eid in discovered:
        ev = state.evidence.get(eid)
        if ev:
            status = ev.state.name
            herring_flag = ""
            linked = ""
            if ev.linked_character_id:
                char = state.characters.get(ev.linked_character_id)
                if char:
                    linked = f" → points to {char.full_name}"
            lines.append(f"  • {ev.name} ({ev.evidence_type.name}, {status}): {ev.description}{linked}{herring_flag}")
    return "\n".join(lines)


def render_character_summary(env: "MysteryEnvironment") -> str: # Thong; summary should not be generated in a rule-based way
    """Render what the agent knows about each character from interviews."""
    state = env.state
    lines = ["=== CHARACTER NOTES ==="]
    for cid in env._interviewed_characters:
        char = state.characters.get(cid)
        if char:
            alibi = f"Alibi: {char.alibi_details}" if char.has_alibi else "No alibi provided."
            motive = f"Possible motive: {char.motive}" if char.motive else ""
            lines.append(f"{char.full_name} — {alibi} {motive}")
    if len(lines) == 1:
        lines.append("No characters interviewed yet.")
    return "\n".join(lines)

