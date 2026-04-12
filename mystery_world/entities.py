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
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Character:
        d = dict(d)
        d["roles"] = [CharacterRole[r] for r in d["roles"]]
        d["relationships"] = [
            Relationship(**r) if isinstance(r, dict) else r
            for r in d.get("relationships", [])
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
    is_red_herring: bool = False
    relevance_score: float = 1.0  # how useful for solving the case (0-1)
    description: str = ""
    discovery_difficulty: float = 0.5  # 0=obvious, 1=very hidden
    weather_sensitive: bool = False     # degrades faster in bad weather
    is_reliable: bool = True           # for testimonial evidence; False = contains inaccuracies
    created_at_step: int = 0
    degraded_at_step: int | None = None

    def is_usable(self) -> bool:
        return self.state in (EvidenceState.PRISTINE, EvidenceState.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["evidence_type"] = self.evidence_type.name
        d["state"] = self.state.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        d = dict(d)
        d["evidence_type"] = EvidenceType[d["evidence_type"]]
        d["state"] = EvidenceState[d["state"]]
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
