"""
Dynamic event system for the mystery world.

Events are deterministic functions of (world_state, rng_state) so that given
the same seed the simulation always produces the same event trace.

Event categories:
    1. Weather transitions
    2. Evidence decay / disappearance
    3. Independent NPC movement
    4. Culprit tampering with evidence
    5. Witness memory degradation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mystery_world.entities import Character
    from mystery_world.world import WorldState

# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------

class EventType(Enum):
    WEATHER_CHANGE = auto()
    EVIDENCE_DECAY = auto()
    EVIDENCE_DESTROYED = auto()
    NPC_MOVE = auto()
    CULPRIT_TAMPER = auto()
    CULPRIT_MOVE_EVIDENCE = auto()
    WITNESS_MEMORY_DECAY = auto()
    NPC_INTERACTION = auto()   # NPC-to-NPC conversation (can generate new testimony)


@dataclass
class WorldEvent:
    """An event that occurred during the simulation step."""
    event_type: EventType
    step: int
    description: str
    actor_id: str | None = None
    target_id: str | None = None
    location_id: str | None = None
    details: dict = field(default_factory=dict)
    agent_visible: bool = True

    def to_dict(self) -> dict:
        d = {
            "event_type": self.event_type.name,
            "step": self.step,
            "description": self.description,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "location_id": self.location_id,
            "details": self.details,
        }
        return d

# ---------------------------------------------------------------------------
# Weather engine
# ---------------------------------------------------------------------------

# Simple Markov chain over weather states
WEATHER_TRANSITIONS: dict[str, dict[str, float]] = {
    "clear":        {"clear": 0.5, "overcast": 0.3, "windy": 0.15, "fog": 0.05},
    "overcast":     {"overcast": 0.3, "light_rain": 0.3, "clear": 0.2, "fog": 0.1, "thunderstorm": 0.1},
    "light_rain":   {"light_rain": 0.3, "heavy_rain": 0.25, "overcast": 0.25, "thunderstorm": 0.15, "clear": 0.05},
    "heavy_rain":   {"heavy_rain": 0.3, "thunderstorm": 0.25, "light_rain": 0.25, "overcast": 0.15, "clear": 0.05},
    "fog":          {"fog": 0.4, "clear": 0.3, "overcast": 0.2, "light_rain": 0.1},
    "thunderstorm": {"thunderstorm": 0.2, "heavy_rain": 0.35, "overcast": 0.3, "clear": 0.15},
    "snow":         {"snow": 0.5, "overcast": 0.25, "clear": 0.15, "windy": 0.1},
    "windy":        {"windy": 0.3, "clear": 0.3, "overcast": 0.2, "light_rain": 0.1, "fog": 0.1},
}

def _sample_weather(current: str, rng: np.random.Generator) -> str:
    row = WEATHER_TRANSITIONS.get(current, WEATHER_TRANSITIONS["clear"])
    states = list(row.keys())
    probs = np.array([row[s] for s in states], dtype=np.float64)
    probs /= probs.sum()
    return states[int(rng.choice(len(states), p=probs))]


def weather_is_bad(weather: str) -> bool:
    return weather in {"heavy_rain", "thunderstorm", "snow"}

# ---------------------------------------------------------------------------
# Event processors — each returns a list of WorldEvents
# ---------------------------------------------------------------------------

def process_weather(state: "WorldState", rng: np.random.Generator) -> list[WorldEvent]:
    """Stochastic weather transitions via Markov chain."""
    events: list[WorldEvent] = []
    if rng.random() < state.config.weather_change_prob:
        old = state.weather
        new = _sample_weather(old, rng)
        if new != old:
            state.weather = new
            events.append(WorldEvent(
                event_type=EventType.WEATHER_CHANGE,
                step=state.current_step,
                description=f"Weather changed from {old} to {new}.",
                details={"old": old, "new": new},
            ))
    return events


def process_evidence_decay(state: "WorldState", rng: np.random.Generator) -> list[WorldEvent]:
    """Evidence degrades or is destroyed over time; bad weather accelerates."""
    from mystery_world.entities import EvidenceState
    events: list[WorldEvent] = []
    rate = state.config.evidence_decay_rate
    if weather_is_bad(state.weather):
        rate *= 1.5   # weather amplifier
    
    for eid, ev in state.evidence.items():
        if ev.state == EvidenceState.DESTROYED:
            continue
        # Weather-sensitive evidence degrades faster outdoors
        loc = state.locations.get(ev.location_id)
        effective_rate = rate
        if ev.weather_sensitive and loc and loc.weather_exposed:
            effective_rate *= 2.0
        
        if rng.random() < effective_rate:
            if ev.state == EvidenceState.PRISTINE:
                ev.state = EvidenceState.DEGRADED
                ev.relevance_score *= 0.6
                ev.degraded_at_step = state.current_step
                events.append(WorldEvent(
                    event_type=EventType.EVIDENCE_DECAY,
                    step=state.current_step,
                    description=f"Evidence '{ev.name}' has degraded.",
                    target_id=eid,
                    location_id=ev.location_id,
                    details={"new_state": "DEGRADED"},
                ))
            elif ev.state == EvidenceState.DEGRADED:
                ev.state = EvidenceState.DESTROYED
                ev.relevance_score = 0.0
                events.append(WorldEvent(
                    event_type=EventType.EVIDENCE_DESTROYED,
                    step=state.current_step,
                    description=f"Evidence '{ev.name}' has been destroyed.",
                    target_id=eid,
                    location_id=ev.location_id,
                    details={"new_state": "DESTROYED"},
                ))
    return events
    

# ---------------------------------------------------------------------------
# NPC movement helpers
# ---------------------------------------------------------------------------        

def _next_step_toward(state: "WorldState", from_id: str, goal_id: str) -> str | None:
    """BFS: return the adjacent location ID one step closer to goal_id."""
    from collections import deque                                                    
    if from_id == goal_id:
        return None                                                                  
    queue: deque[list[str]] = deque([[from_id]])
    visited: set[str] = {from_id}                                                    
    while queue:
        path = queue.popleft()                                                       
        loc = state.locations.get(path[-1])
        if not loc:                                                                  
            continue
        for adj in loc.adjacent_ids:
            if adj == goal_id:                                                       
                return path[1] if len(path) > 1 else adj
            if adj not in visited:                                                   
                visited.add(adj)
                queue.append(path + [adj])                                           
    return None 

                                                                                   
def _move_character(char: "Character", state: "WorldState", new_loc_id: str) -> None:
    """Update location manifests when a character moves."""                          
    old_loc = state.locations.get(char.location_id)                                  
    if old_loc and char.id in old_loc.characters_here:
        old_loc.characters_here.remove(char.id)                                      
    char.location_id = new_loc_id
    new_loc = state.locations.get(new_loc_id)                                        
    if new_loc and char.id not in new_loc.characters_here:                           
        new_loc.characters_here.append(char.id)
                                                                                   
                                                                                   
def _pick_social_goal(char: "Character", state: "WorldState", rng: np.random.Generator) -> str | None:
    """Return the current location of a positively-related character, or None."""    
    liked = [r for r in char.relationships if r.sentiment > 0.3]                     
    if not liked:                                                                    
        return None                                                                  
    rng.shuffle(liked)                                                               
    for rel in liked:                                                                
        target = state.characters.get(rel.target_id)
        if target and target.is_alive and target.location_id in state.locations:     
            if target.location_id != char.location_id:                               
                return target.location_id
    return None                                                                      
                
                                                                                   
def _pick_errand_goal(char: "Character", state: "WorldState", rng: np.random.Generator) -> str:
    """Return a random location the character is not already at."""                  
    candidates = [lid for lid in state.locations if lid != char.location_id]         
    if not candidates:                                                               
        return char.location_id                                                      
    return candidates[int(rng.integers(len(candidates)))]                            
                
                                                                                   
def _set_innocent_next_goal(char: "Character", state: "WorldState", rng: np.random.Generator) -> None:
    """                                                                              
    Assign dwell time and next destination for an innocent NPC.
                                                                                   
    At home  → dwell 2-4 steps, then 50% social visit / 50% errand.                  
    Away     → dwell 1-2 steps, then 70% return home / 30% new errand.               
    """                                                                              
    at_home = char.location_id == char.home_location_id
    if at_home:                                                                      
        char.movement_dwell_steps = int(rng.integers(2, 5))
        if rng.random() < 0.5:                                                       
            social = _pick_social_goal(char, state, rng)                             
            char.movement_goal_location_id = social or _pick_errand_goal(char, state, rng)
        else:                                                                        
            char.movement_goal_location_id = _pick_errand_goal(char, state, rng)
    else:                                                                            
        char.movement_dwell_steps = int(rng.integers(1, 3))
        if rng.random() < 0.7 and char.home_location_id:                             
            char.movement_goal_location_id = char.home_location_id
        else:                                                                        
            char.movement_goal_location_id = _pick_errand_goal(char, state, rng)
                                                                                   
                                                                                   
def _most_threatening_evidence_loc(culprit: "Character", state: "WorldState") -> str | None:
    """Location of culprit's highest-relevance untampered incriminating evidence.""" 
    from mystery_world.entities import EvidenceState                                 
    best_loc: str | None = None
    best_score = -1.0                                                                
    for ev in state.evidence.values():
        if ev.linked_character_id != culprit.id:                                     
            continue
        if ev.state in (EvidenceState.DESTROYED, EvidenceState.HIDDEN):              
            continue
        if ev.relevance_score > best_score:                                          
            best_score = ev.relevance_score
            best_loc = ev.location_id                                                
    return best_loc

                                                                                   
def _set_culprit_next_goal(culprit: "Character", state: "WorldState", rng: np.random.Generator) -> None:
    """                                                                              
    Assign dwell time and next destination for the culprit.

    Mirrors innocent NPC routine for camouflage. Occasionally initiates a tamper     
    run: heads directly to the most threatening evidence, acts immediately (dwell=0),
    then returns home. Tamper runs are probability-gated — not triggered every step. 
    """                                                                              
    threat_loc = _most_threatening_evidence_loc(culprit, state)                      
                                                                                   
    # Just arrived at an evidence location — process_culprit_tampering handles the   
    # actual tamper this same step. Culprit leaves promptly without lingering.
    if threat_loc is not None and threat_loc == culprit.location_id:                 
        culprit.movement_dwell_steps = 0                                             
        culprit.movement_goal_location_id = (                                        
            culprit.home_location_id                                                 
            if culprit.home_location_id and culprit.home_location_id != culprit.location_id
            else _pick_errand_goal(culprit, state, rng)                              
        )       
        return                                                                       
                
    # Initiate a tamper run? Only when evidence exists elsewhere and probability fires.                                                                                                                                                                                                                                                                       
    if (
        threat_loc is not None                                                       
        and threat_loc != culprit.location_id                                        
        and rng.random() < state.config.culprit_tamper_prob
    ):                                                                               
        culprit.movement_dwell_steps = 0  # no loitering — move immediately
        culprit.movement_goal_location_id = threat_loc                               
        return
                                                                                   
    # Normal routine — identical structure to innocent NPC (camouflage).             
    at_home = culprit.location_id == culprit.home_location_id
    if at_home:                                                                      
        culprit.movement_dwell_steps = int(rng.integers(2, 5))
        if rng.random() < 0.5:                                                       
            social = _pick_social_goal(culprit, state, rng)                          
            culprit.movement_goal_location_id = social or _pick_errand_goal(culprit, state, rng)
        else:                                                                        
            culprit.movement_goal_location_id = _pick_errand_goal(culprit, state, rng)
    else:                                                                            
        culprit.movement_dwell_steps = int(rng.integers(1, 3))
        if rng.random() < 0.7 and culprit.home_location_id:                          
            culprit.movement_goal_location_id = culprit.home_location_id
        else:                                                                        
            culprit.movement_goal_location_id = _pick_errand_goal(culprit, state, rng)
                                                                                   
                                                                                   
def process_npc_movement(state: "WorldState", rng: np.random.Generator) -> list[WorldEvent]:
    """                                                                              
    NPC movement.

    Option A (reactive_events=False): random walk at npc_move_prob per step.         
    Option B (reactive_events=True): routine-based movement for all characters.
    - Innocents have a home base, make social visits and errands, dwell naturally. 
    - Culprit follows the same visible routine but occasionally makes tamper runs. 
    - Both use the same movement system — culprit is not distinguishable by        
        movement pattern alone.                                                      
    """                                                                              
    from mystery_world.entities import CharacterRole                                 
    events: list[WorldEvent] = []

    for cid, char in state.characters.items():                                       
        if not char.is_alive:
            continue                                                                 
        if CharacterRole.VICTIM in char.roles:
            continue

        loc = state.locations.get(char.location_id)                                  
        if not loc:
            continue                                                                 
                
        # --- Option A: random walk ---
        if not state.config.reactive_events:
            if rng.random() < state.config.npc_move_prob:
                if loc.adjacent_ids:                                                 
                    new_loc_id = rng.choice(loc.adjacent_ids)
                    old_loc_id = char.location_id                                    
                    _move_character(char, state, new_loc_id)                         
                    new_loc = state.locations.get(new_loc_id)
                    events.append(WorldEvent(                                        
                        event_type=EventType.NPC_MOVE,
                        step=state.current_step,                                     
                        description=f"{char.full_name} moved from {loc.name} to {new_loc.name if new_loc else new_loc_id}.",
                        actor_id=cid,                                                
                        location_id=new_loc_id,
                        details={"from": old_loc_id, "to": new_loc_id},              
                    ))                                                               
            continue
                                                                                   
        # --- Option B: routine-based movement ---

        # Initialise home lazily on first Option B step                              
        if char.home_location_id is None:
            char.home_location_id = char.location_id                                 
                
        # Dwelling — count down and stay put                                         
        if char.movement_dwell_steps > 0:
            char.movement_dwell_steps -= 1                                           
            continue
                                                                                   
        # At goal or no goal — decide what to do next                                
        if (
            char.movement_goal_location_id is None                                   
            or char.movement_goal_location_id == char.location_id
        ):                                                                           
            if char.is_culprit:
                _set_culprit_next_goal(char, state, rng)                             
            else:                                                                    
                _set_innocent_next_goal(char, state, rng)
            continue  # goal just assigned; start moving next step                   
                                                                                   
        # Have a goal and not yet there — move one step toward it                    
        next_id = _next_step_toward(state, char.location_id, char.movement_goal_location_id)                                                                                                                                                                                                                                                                  
        if next_id:
            old_loc_id = char.location_id
            _move_character(char, state, next_id)
            new_loc = state.locations.get(next_id)
            events.append(WorldEvent(
                event_type=EventType.NPC_MOVE,
                step=state.current_step,
                description=f"{char.full_name} moved from {loc.name} to {new_loc.name if new_loc else next_id}.",
                actor_id=cid,                                                        
                location_id=next_id,
                details={"from": old_loc_id, "to": next_id},                         
            ))                                              
                                                                                   
    return events


def process_culprit_tampering(state: "WorldState", rng: np.random.Generator) -> list[WorldEvent]:
    """The culprit actively hides or relocates incriminative evidence."""
    from mystery_world.entities import EvidenceState
    events: list[WorldEvent] = []
    culprit = None
    for c in state.characters.values():
        if c.is_culprit and c.is_alive:
            culprit = c
            break

    if culprit is None:
        return events
    
    for eid, ev in state.evidence.items():
        if ev.state in (EvidenceState.DESTROYED, EvidenceState.HIDDEN):
            continue
        
        if ev.linked_character_id != culprit.id:        
            continue                            
                                                                                                                                                                                                                                                                                                                                                            
        # Option B: culprit must be physically present; higher probability since
        # they traveled there deliberately.                                                                                                                                                                                                                                                                                                                   
        if state.config.reactive_events:   
            if ev.location_id != culprit.location_id:                                                                                                                                                                                                                                                                                                         
                continue                                                                                                                                                                                                                                                                                                                                      
            tamper_prob = min(1.0, state.config.culprit_tamper_prob * 1.5)
        else:                                                                                                                                                                                                                                                                                                                                                 
            tamper_prob = state.config.culprit_tamper_prob                                                                                                                                                                                                                                                                                                    

        if rng.random() < tamper_prob:                                                                                                                                                                                                                                                                                                                        
            action = rng.choice(["hide", "move"])
            if action == "hide":
                ev.state = EvidenceState.HIDDEN
                ev.discovery_difficulty = min(1.0, ev.discovery_difficulty + 0.3)
                events.append(WorldEvent(
                    event_type=EventType.CULPRIT_TAMPER,
                    step=state.current_step,
                    description=f"The culprit hid evidence '{ev.name}'.",
                    actor_id=culprit.id,
                    target_id=eid,
                    location_id=ev.location_id,
                    details={"action": "hide"},
                    agent_visible=False
                ))
            else:
                # Move to a random location
                loc_ids = [lid for lid in state.locations if lid != ev.location_id]
                if loc_ids:
                    new_loc = rng.choice(loc_ids)
                    old_loc = ev.location_id
                    # update location manifests
                    if ev.id in state.locations[old_loc].objects_here:
                        state.locations[old_loc].objects_here.remove(ev.id)
                    state.locations[new_loc].objects_here.append(ev.id)
                    ev.location_id = new_loc
                    ev.state = EvidenceState.MOVED
                    events.append(WorldEvent(
                        event_type=EventType.CULPRIT_MOVE_EVIDENCE,
                        step=state.current_step,
                        description=f"The culprit moved evidence '{ev.name}' to {state.locations[new_loc].name}.",
                        actor_id=culprit.id,
                        target_id=eid,
                        location_id=new_loc,
                        details={"action": "move", "from": old_loc, "to": new_loc},
                        agent_visible=False
                    ))
    return events


def process_witness_memory(state: "WorldState", _rng: np.random.Generator) -> list[WorldEvent]:
    """Witnesses recall degrades exponentially with half-life from config."""
    events: list[WorldEvent] = []
    half_life = state.config.witness_memory_half_life
    if half_life <= 0:
        return events
    for cid, char in state.characters.items():
        if not char.witnessed_events:
            continue
        
        # Exponential decay: reliability = 0.5^(steps_since_event / half_life)
        # We use steps since initial observation (approximated by step count)
        steps_elapsed = state.current_step
        new_reliability = math.pow(0.5, steps_elapsed / half_life)
        old_reliability = char.memory_reliability
        char.memory_reliability = max(0.05, new_reliability)   # floor at 5 %
        if old_reliability - char.memory_reliability > 0.05:
            events.append(WorldEvent(
                event_type=EventType.WITNESS_MEMORY_DECAY,
                step=state.current_step,
                description=f"{char.full_name}'s memory reliability dropped to {char.memory_reliability:.0%}.", # Thong: this should not be exposed to the agent. The agent must deduce it by himself.
                actor_id=cid,
                details={"old": round(old_reliability, 3), "new": round(char.memory_reliability, 3)},
                agent_visible=False
            ))
    return events



# ---------------------------------------------------------------------------
# Master dispatcher
# ---------------------------------------------------------------------------

ALL_PROCESSORS = [
    process_weather,
    process_evidence_decay,
    process_npc_movement,
    process_culprit_tampering,
    process_witness_memory,
]

def process_all_events(state: "WorldState", rng: np.random.Generator) -> list[WorldEvent]:  # Thong: events should be affected by the agent's actions
    """Run every event processor for the current step. Deterministic given rng state."""
    all_events: list[WorldEvent] = []
    for proc in ALL_PROCESSORS:
        all_events.extend(proc(state, rng))
    return all_events
