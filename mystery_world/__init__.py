"""
Configuration for mystery world generation and complexity control.

Complexity is parameterised along orthogonal axes so that benchmark instances
can be generated at any point in the difficulty space. Each ``ComplexityConfig``
is fully serialisable to / from YAML for reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any


class ComplexityLevel(IntEnum):
    """Coarse-grained difficulty buckets (1-5)."""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass(frozen=True)
class ComplexityConfig:
    """Fine-grained knobs that control how hard a mystery instance is."""

    # --- World size ---
    num_locations: int = 5
    num_suspects: int = 4
    num_innocents: int = 2   # NPCs who are not suspects
    num_weapons: int = 3
    num_objects: int = 8   # total interactive objects in the world
    num_red_herrings: int = 2   # misleading clues planted by generator

    # --- Temporal ---
    num_time_steps: int = 12   # discrete time steps in the scenario
    evidence_decay_rate: float = 0.1   # probability per step an evidence degrades
    witness_memory_half_life: int = 6   # steps until witness recall = 50%

    # --- Dynamic events ---
    weather_change_prob: float = 0.15
    npc_move_prob: float = 0.3  # independent NPC relocation probability
    culprit_tamper_prob: float = 0.2   # culprit actively hides / moves evidence
    reactive_events: bool = False      # routine-based NPC movement (HARD/EXPERT only)       

    # --- Reasoning load ---
    alibi_complexity: int = 2  # how many alibi chains to verify
    motive_layers: int = 1  # nested motive depth
    requires_deduction: bool = True   # must the agent use strict deduction?
    requires_abduction: bool = True  # must the agent reason to best explanation?

    # --- Evidence realism ---
    evidence_ambiguity: float = 0.0        # probability evidence links to non-culprit
    evidence_difficulty_min: float = 0.2   # minimum discovery difficulty
    evidence_difficulty_max: float = 0.6   # maximum discovery difficulty
    testimony_unreliability: float = 0.0   # probability a testimonial is unreliable

    # --- Alibi system ---
    allow_suspect_corroborators: bool = False   # whether suspects can corroborate alibis
    max_corroborators: int = 1                  # maximum corroborators per alibi
    # Cumulative probability thresholds for culprit alibi type:
    # [no_alibi, solo, partial, gap_corroborated, full_false]
    culprit_alibi_weights: tuple = (0.30, 0.45, 0.60, 0.80, 1.00) 

    # --- Agent budget ---
    max_agent_actions: int = 30   # action budget (affects clue efficiency)

    # --- Serialisation helpers ---
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ComplexityConfig:
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

    
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    
    @classmethod
    def from_json(cls, path: str | Path) -> ComplexityConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Pre-built complexity presets (Table 1 in the paper)
# ---------------------------------------------------------------------------

COMPLEXITY_PRESETS: dict[ComplexityLevel, ComplexityConfig] = {
    ComplexityLevel.TRIVIAL: ComplexityConfig(
        num_locations=3, num_suspects=2, num_innocents=1, num_weapons=2,
        num_objects=4, num_red_herrings=0, num_time_steps=6,
        evidence_decay_rate=0.0, witness_memory_half_life=100,
        weather_change_prob=0.0, npc_move_prob=0.0, culprit_tamper_prob=0.0,
        alibi_complexity=1, motive_layers=1,
        requires_deduction=True, requires_abduction=False,
        evidence_ambiguity=0.0, evidence_difficulty_min=0.1, evidence_difficulty_max=0.3, testimony_unreliability=0.0,
        allow_suspect_corroborators=False, max_corroborators=1, culprit_alibi_weights=(0.60, 1.00, 1.00, 1.00, 1.00),
        max_agent_actions=30,
    ),
    ComplexityLevel.EASY: ComplexityConfig(
        num_locations=4, num_suspects=3, num_innocents=1, num_weapons=2,
        num_objects=6, num_red_herrings=1, num_time_steps=8,
        evidence_decay_rate=0.05, witness_memory_half_life=10,
        weather_change_prob=0.05, npc_move_prob=0.1, culprit_tamper_prob=0.05,
        alibi_complexity=1, motive_layers=1,
        requires_deduction=True, requires_abduction=False,
        evidence_ambiguity=0.1, evidence_difficulty_min=0.15, evidence_difficulty_max=0.4, testimony_unreliability=0.1,
        allow_suspect_corroborators=False, max_corroborators=1, culprit_alibi_weights=(0.45, 0.75, 0.90, 1.00, 1.00),
        max_agent_actions=40,
    ),
    ComplexityLevel.MEDIUM: ComplexityConfig(
        num_locations=5, num_suspects=4, num_innocents=2, num_weapons=3,
        num_objects=8, num_red_herrings=2, num_time_steps=12,
        evidence_decay_rate=0.1, witness_memory_half_life=6,
        weather_change_prob=0.15, npc_move_prob=0.3, culprit_tamper_prob=0.2,
        alibi_complexity=2, motive_layers=1,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.2, evidence_difficulty_min=0.2, evidence_difficulty_max=0.6, testimony_unreliability=0.2,
        allow_suspect_corroborators=False, max_corroborators=2, culprit_alibi_weights=(0.30, 0.50, 0.70, 0.90, 1.00),
        max_agent_actions=60,
    ),
    ComplexityLevel.HARD: ComplexityConfig(
        num_locations=7, num_suspects=5, num_innocents=3, num_weapons=4,
        num_objects=12, num_red_herrings=3, num_time_steps=12,
        evidence_decay_rate=0.15, witness_memory_half_life=4,
        weather_change_prob=0.2, npc_move_prob=0.4, culprit_tamper_prob=0.3,
        alibi_complexity=3, motive_layers=2,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.35, evidence_difficulty_min=0.3, evidence_difficulty_max=0.7, testimony_unreliability=0.3,
        allow_suspect_corroborators=True, max_corroborators=2, culprit_alibi_weights=(0.20, 0.35, 0.55, 0.80, 1.00),
        max_agent_actions=80, reactive_events=True,
    ),
    ComplexityLevel.EXPERT: ComplexityConfig(
        num_locations=10, num_suspects=7, num_innocents=4, num_weapons=5,
        num_objects=18, num_red_herrings=5, num_time_steps=24,
        evidence_decay_rate=0.2, witness_memory_half_life=3,
        weather_change_prob=0.25, npc_move_prob=0.5, culprit_tamper_prob=0.4,
        alibi_complexity=4, motive_layers=3,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.5, evidence_difficulty_min=0.3, evidence_difficulty_max=0.8, testimony_unreliability=0.4,
        allow_suspect_corroborators=True, max_corroborators=3, culprit_alibi_weights=(0.10, 0.20, 0.40, 0.70, 1.00),
        max_agent_actions=120, reactive_events=True,
    ),
}

# ---------------------------------------------------------------------------
# Asset pools (names, locations, etc.) - all original, no copyrighted material
# ---------------------------------------------------------------------------

@dataclass
class AssetPool:
    """Pools of procedural-generation atoms. All names are original."""

    location_templates: list[str] = field(default_factory=lambda: [
        "Grand Foyer", "Dimly-Lit Study", "Rose Garden", "Cellar Vault",
        "Clock Tower", "Courtyard Terrace", "Servants' Quarters", "Wine Pantry",
        "Observatory Deck", "Library Annex", "Billiard Room", "Greenhouse",
        "Attic Storage", "Chapel Nook", "Boat House", "Music Salon",
    ])

    first_names: list[str] = field(default_factory=lambda: [
        "Adrian", "Beatrix", "Cedric", "Dahlia", "Emory", "Fern", "Gareth",
        "Helena", "Inigo", "Josette", "Kieran", "Linnea", "Maxfield",
        "Nadia", "Orson", "Petra", "Quinn", "Rosalind", "Silas", "Thea",
    ])

    last_names: list[str] = field(default_factory=lambda: [
        "Ashworth", "Blackwood", "Crane", "Deveraux", "Elsworth", "Finch",
        "Greystone", "Harlow", "Iverson", "Juno", "Kingsley", "Lark",
        "Montague", "Nightingale", "Oakley", "Prescott", "Quinlan",
        "Ravenswood", "Sterling", "Thornfield",
    ])

    weapon_templates: list[str] = field(default_factory=lambda: [
        "ornate letter opener", "brass candlestick", "silk scarf",
        "gardening shears", "crystal decanter", "antique revolver",
        "iron fireplace poker", "marble bookend", "coil of rope",
        "kitchen cleaver", "heavy statuette", "poison vial",
    ])

    motive_templates: list[str] = field(default_factory=lambda: [
        "inheritance dispute", "unrequited love", "business rivalry",
        "blackmail threat", "old family grudge", "political ambition",
        "jealousy over promotion", "secret identity exposure",
        "revenge for past betrayal", "debt and desperation",
        "intellectual theft accusation", "custody battle",
    ])

    object_templates: list[str] = field(default_factory=lambda: [
        "torn envelope", "muddy boots", "half-burned letter",
        "pocket watch stopped at 11:42", "lipstick-stained glass",
        "crumpled receipt", "misplaced key ring", "bloodstained glove",
        "broken spectacles", "cigar stub", "damp umbrella",
        "unfinished chess game", "open safe", "overturned chair",
        "scratched painting frame", "spilled ink bottle",
        "locked diary", "wilted bouquet", "train ticket stub",
        "smudged fingerprint on window",
    ])

    weather_types: list[str] = field(default_factory=lambda: [
        "clear", "overcast", "light_rain", "heavy_rain",
        "fog", "thunderstorm", "snow", "windy",
    ])

    personality_traits: list[str] = field(default_factory=lambda: [
        "nervous", "stoic", "gregarious", "secretive", "meticulous",
        "impulsive", "charming", "bitter", "distracted", "cunning",
    ])

DEFAULT_ASSET_POOL = AssetPool()
