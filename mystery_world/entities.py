"""
Core entity types for the mystery world.

All entities are plain dataclasses (easily serialisable) and carry unique IDs
so they can be referenced from event logs and agent observations.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# ---------------------------------------------------------------------------
# Locard triangle types
# ---------------------------------------------------------------------------

class EdgeType(Enum):
    SUSPECT_WEAPON = auto()
    WEAPON_VICTIM = auto()
    SUSPECT_ROOM = auto()
    SUSPECT_ELSEWHERE = auto()   # innocent confirmed at non-murder room at murder time


class TemporalLabel(Enum):
    """What the agent sees about evidence freshness (never the raw timestamp)."""
    CLEARLY_FRESH = auto()   # easy: "still wet", "warm to touch"
    CLEARLY_STALE = auto()   # easy: "dusty", "dried and cracked"
    AMBIGUOUS = auto()       # medium+: no obvious age indicator, must ANALYZE


@dataclass
class EdgeRelevance:
    """Links one piece of evidence to one triangle edge with temporal metadata."""
    edge_type: EdgeType = EdgeType.SUSPECT_WEAPON
    subject_ids: list[str] = field(default_factory=list)
    contact_timestamp: float = 0.0
    surface_label: TemporalLabel = TemporalLabel.AMBIGUOUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type": self.edge_type.name,
            "subject_ids": self.subject_ids,
            "contact_timestamp": self.contact_timestamp,
            "surface_label": self.surface_label.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeRelevance:
        d = dict(d)
        d["edge_type"] = EdgeType[d["edge_type"]]
        d["surface_label"] = TemporalLabel[d["surface_label"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Temporal reasoning types
# ---------------------------------------------------------------------------

class TimeStyle(Enum):
    CLOCK = auto()          # "at 9:30 PM"
    NAMED_PERIOD = auto()   # "sometime after dinner"
    RELATIVE = auto()       # "about ten minutes before the scream"


@dataclass
class AlibiClaim:
    """One position claim in a suspect's alibi. Ground truth stored internally."""
    location_name: str = ""
    step: int = 0
    clock_time_str: str = ""    # always known: "9:30 PM"
    stated_time: str = ""       # what the suspect says — may be vague
    time_style: TimeStyle = TimeStyle.CLOCK

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["time_style"] = self.time_style.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlibiClaim:
        d = dict(d)
        d["time_style"] = TimeStyle[d["time_style"]]
        return cls(**d)


@dataclass
class WitnessStatement:
    """A witness's account of where they saw someone at a given time."""
    witness_id: str = ""
    observed_character_id: str = ""
    location_name: str = ""
    step: int = 0
    clock_time_str: str = ""
    stated_time: str = ""
    time_style: TimeStyle = TimeStyle.CLOCK
    is_reliable: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["time_style"] = self.time_style.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WitnessStatement:
        d = dict(d)
        d["time_style"] = TimeStyle[d["time_style"]]
        return cls(**d)


@dataclass
class RouteConstraint:
    """A passage blocked between two locations during a step range."""
    from_location_id: str = ""
    to_location_id: str = ""
    blocked_from_step: int = 0
    blocked_until_step: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouteConstraint:
        return cls(**d)

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def _short_id() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

class LocationTag(Enum):
    INDOOR = auto()
    OUTDOOR = auto()
    UNDERGROUND = auto()


@dataclass
class Location:
    id: str = field(default_factory=_short_id)
    name: str = ""
    tag: LocationTag = LocationTag.INDOOR
    description: str = ""
    adjacent_ids: list[str] = field(default_factory=list)
    objects_here: list[str] = field(default_factory=list)  # object IDs
    characters_here: list[str] = field(default_factory=list)  # character IDs
    is_locked: bool = False
    weather_exposed: bool = False  # outdoor locations affected by weather
    material_signature: str = ""   # dominant trace material found in this room

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["tag"] = self.tag.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Location:
        d = dict(d)
        d["tag"] = LocationTag[d["tag"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Characters (suspects, innocents, victim)
# ---------------------------------------------------------------------------

class CharacterRole(Enum):
    VICTIM = auto()
    SUSPECT = auto()
    INNOCENT = auto()
    WITNESS = auto()       # can overlap with innocent / suspect


@dataclass
class Relationship:
    target_id: str = ""
    kind: str = ""         # e.g. "sibling", "rival", "employer"
    sentiment: float = 0.0  # -1.0 (hostile) … +1.0 (friendly)


@dataclass
class PhysicalTraits:
    """Observable physical characteristics — matched to evidence clues."""
    build: str = ""     # "heavy-set", "lean and tall", etc.
    hair: str = ""      # "short dark hair", "long auburn hair", etc.
    hands: str = ""     # "calloused hands", "ink-stained fingers", etc.

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhysicalTraits:
        return cls(**d)


@dataclass
class Character:
    id: str = field(default_factory=_short_id)
    first_name: str = ""
    last_name: str = ""
    roles: list[CharacterRole] = field(default_factory=list)
    personality: str = ""
    location_id: str = ""
    motive: str | None = None          # only for suspects
    has_alibi: bool = False
    alibi_details: str = ""
    alibi_corroborator_id: str | None = None
    alibi_corroboration_is_genuine: bool = True   # False = corroborator is lying 
    alibi_has_gap: bool = False                   # True = corroborators honest but missed a window     
    relationships: list[Relationship] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)   # object IDs
    physical_traits: PhysicalTraits = field(default_factory=PhysicalTraits)
    alibi_claims: list[AlibiClaim] = field(default_factory=list)
    is_alive: bool = True
    is_culprit: bool = False
    movement_goal_location_id: str | None = None  # current movement target
    movement_dwell_steps: int = 0                 # steps to remain before moving again
    home_location_id: str | None = None           # base location

    # --- Witness memory ---
    witnessed_events: list[str] = field(default_factory=list)  # event IDs
    memory_reliability: float = 1.0  # decays over time

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["roles"] = [r.name for r in self.roles]
        d["alibi_claims"] = [a.to_dict() for a in self.alibi_claims]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Character:
        d = dict(d)
        d["roles"] = [CharacterRole[r] for r in d["roles"]]
        d["relationships"] = [
            Relationship(**r) if isinstance(r, dict) else r
            for r in d.get("relationships", [])
        ]
        pt = d.get("physical_traits")
        if isinstance(pt, dict):
            d["physical_traits"] = PhysicalTraits(**pt)
        d["alibi_claims"] = [
            AlibiClaim.from_dict(a) if isinstance(a, dict) else a
            for a in d.get("alibi_claims", [])
        ]
        return cls(**d)


# ---------------------------------------------------------------------------
# Objects & Evidence
# ---------------------------------------------------------------------------

class EvidenceState(Enum):
    PRISTINE = auto()
    DEGRADED = auto()
    CONTAMINATED = auto()
    DESTROYED = auto()
    HIDDEN = auto()
    MOVED = auto()


class EvidenceType(Enum):
    PHYSICAL = auto()       # fingerprints, DNA, bloodstain
    TESTIMONIAL = auto()    # witness statement
    DOCUMENTARY = auto()    # letter, receipt, diary entry
    CIRCUMSTANTIAL = auto() # object out of place


@dataclass
class Evidence:
    """A piece of evidence that can be discovered, can degrade, and can be
    tampered with by the culprit."""
    id: str = field(default_factory=_short_id)
    name: str = ""
    evidence_type: EvidenceType = EvidenceType.PHYSICAL
    state: EvidenceState = EvidenceState.PRISTINE
    location_id: str = ""
    linked_character_id: str | None = None  # who it points to
    corroborator_id: str | None = None   # for SUSPECT_ELSEWHERE evidence only
    is_red_herring: bool = False
    relevance_score: float = 1.0  # how useful for solving the case (0-1)
    description: str = ""
    discovery_difficulty: float = 0.5  # 0=obvious, 1=very hidden
    weather_sensitive: bool = False     # degrades faster in bad weather
    is_reliable: bool = True           # for testimonial evidence; False = contains inaccuracies
    created_at_step: int = 0
    degraded_at_step: int | None = None
    # --- Locard triangle ---
    relevance: EdgeRelevance | None = None

    def is_usable(self) -> bool:
        return self.state in (EvidenceState.PRISTINE, EvidenceState.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["evidence_type"] = self.evidence_type.name
        d["state"] = self.state.name
        d["relevance"] = self.relevance.to_dict() if self.relevance else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        d = dict(d)
        d["evidence_type"] = EvidenceType[d["evidence_type"]]
        d["state"] = EvidenceState[d["state"]]
        rel = d.get("relevance")
        d["relevance"] = EdgeRelevance.from_dict(rel) if rel else None
        return cls(**d)


@dataclass
class WorldObject:
    """A generic interactive object (may or may not be evidence)."""
    id: str = field(default_factory=_short_id)
    name: str = ""
    description: str = ""
    location_id: str = ""
    is_weapon: bool = False
    is_murder_weapon: bool = False
    portable: bool = True
    evidence_id: str | None = None  # linked evidence if any

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldObject:
        return cls(**d)


# ---------------------------------------------------------------------------
# Timeline entries (for the ground-truth timeline)
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    """An immutable record of something that happened at a specific time step."""
    step: int = 0
    actor_id: str = ""
    action: str = ""        # e.g. "moved_to", "picked_up", "attacked"
    target_id: str = ""     # location / object / character ID
    location_id: str = ""
    details: str = ""
    is_public: bool = False  # observable by anyone present
    witnesses: list[str] = field(default_factory=list)  # character IDs

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimelineEntry:
        return cls(**d)



# ---------------------------------------------------------------------------
# Scoring types
# ---------------------------------------------------------------------------

@dataclass
class EdgeArgument:
    """Agent's cited evidence for one triangle edge."""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeArgument:
        return cls(**d)


@dataclass
class ScoreResult:
    """Output of the unified scoring function (accusation + triangle + alibi)."""
    # Accusation
    correct_suspect: bool = False
    correct_weapon: bool = False
    correct_room: bool = False
    accusation_score: float = 0.0

    # Locard triangle — per-edge precision, recall, F1
    suspect_weapon_precision: float = 0.0
    suspect_weapon_recall: float = 0.0
    suspect_weapon_score: float = 0.0   # F1
    weapon_victim_precision: float = 0.0
    weapon_victim_recall: float = 0.0
    weapon_victim_score: float = 0.0
    suspect_room_precision: float = 0.0
    suspect_room_recall: float = 0.0
    suspect_room_score: float = 0.0
    triangle_score: float = 0.0         # sum of the three F1s

    # Alibi
    alibi_cited: bool = False
    contradiction_found: bool = False
    contradiction_valid: bool = False
    alibi_score: float = 0.0

    # Elimination
    correct_eliminations: int = 0
    incorrect_eliminations: int = 0
    total_innocents: int = 0
    elimination_score: float = 0.0

    # Composite
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
        