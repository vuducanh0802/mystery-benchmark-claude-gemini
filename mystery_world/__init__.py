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
    max_objects_per_room: int = 5   # cap inspectable objects per room
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
    free_culprit_actions: bool = False # if True, culprit actions come from an agent, not event processors

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

    # --- Stochastic discovery (decay-retry perception model) ---
    detective_miss_base: float = 0.0       # base per-EXAMINE miss prob; effective = base × discovery_difficulty × decay**attempt
    examine_attempt_decay: float = 0.5     # geometric decay of miss prob across repeated EXAMINEs of the same object
    search_miss_multiplier: float = 0.4    # miss-prob multiplier when a thorough SEARCH is used instead of EXAMINE
    culprit_conceal_prob: float = 0.0      # generation-time: prob each culprit-linked evidence is pre-concealed

    # --- Crime scene realism ---
    body_moved_prob: float = 0.0        # probability body was moved from murder location
    body_trace_ambiguity: int = 1       # 1=room named, 2=material named, 3=vague, 4=multiple candidates
    trail_completeness: float = 1.0     # fraction of intermediate rooms that show drag traces
    witness_specificity: float = 1.0    # 1.0=names the room, 0.5=vague direction, 0.2=barely recalls

    freshness_threshold: float = 2.0   # steps: |contact_time - murder_time| < this → fresh

    # --- Temporal reasoning ---
    step_duration_minutes: int = 30    # real minutes per game step
    world_start_hour: int = 20         # 8 PM — when the evening begins
    num_route_constraints: int = 1     # locked passages forcing route rethinking

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
        detective_miss_base=0.0, culprit_conceal_prob=0.0,
        allow_suspect_corroborators=False, max_corroborators=1, culprit_alibi_weights=(0.20, 0.45, 0.65, 0.85, 1.00),
        body_moved_prob=0.0, body_trace_ambiguity=1, trail_completeness=1.0, witness_specificity=1.0,
        freshness_threshold=3.0, step_duration_minutes=30, world_start_hour=20, num_route_constraints=0, 
        max_agent_actions=40, max_objects_per_room=4,
    ),
    ComplexityLevel.EASY: ComplexityConfig(
        num_locations=4, num_suspects=3, num_innocents=1, num_weapons=2,
        num_objects=6, num_red_herrings=1, num_time_steps=8,
        evidence_decay_rate=0.05, witness_memory_half_life=10,
        weather_change_prob=0.05, npc_move_prob=0.1, culprit_tamper_prob=0.05,
        alibi_complexity=1, motive_layers=1,
        requires_deduction=True, requires_abduction=False,
        evidence_ambiguity=0.1, evidence_difficulty_min=0.15, evidence_difficulty_max=0.4, testimony_unreliability=0.1,
        detective_miss_base=0.0, culprit_conceal_prob=0.05,
        allow_suspect_corroborators=False, max_corroborators=1, culprit_alibi_weights=(0.20, 0.40, 0.60, 0.80, 1.00),
        body_moved_prob=0.1, body_trace_ambiguity=2, trail_completeness=0.8, witness_specificity=0.8,
        freshness_threshold=2.5, step_duration_minutes=30, world_start_hour=20, num_route_constraints=0,
        max_agent_actions=50, max_objects_per_room=4,
    ),
    ComplexityLevel.MEDIUM: ComplexityConfig(
        num_locations=5, num_suspects=4, num_innocents=2, num_weapons=3,
        num_objects=8, num_red_herrings=2, num_time_steps=12,
        evidence_decay_rate=0.1, witness_memory_half_life=6,
        weather_change_prob=0.15, npc_move_prob=0.3, culprit_tamper_prob=0.2,
        alibi_complexity=2, motive_layers=1,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.2, evidence_difficulty_min=0.2, evidence_difficulty_max=0.6, testimony_unreliability=0.2,
        detective_miss_base=0.10, culprit_conceal_prob=0.15,
        allow_suspect_corroborators=False, max_corroborators=2, culprit_alibi_weights=(0.30, 0.50, 0.70, 0.90, 1.00),
        body_moved_prob=0.3, body_trace_ambiguity=3, trail_completeness=0.6, witness_specificity=0.5,
        freshness_threshold=2.0, step_duration_minutes=30, world_start_hour=20, num_route_constraints=1,
        max_agent_actions=75, max_objects_per_room=5,
    ),
    ComplexityLevel.HARD: ComplexityConfig(
        num_locations=7, num_suspects=5, num_innocents=3, num_weapons=4,
        num_objects=12, num_red_herrings=3, num_time_steps=12,
        evidence_decay_rate=0.15, witness_memory_half_life=4,
        weather_change_prob=0.2, npc_move_prob=0.4, culprit_tamper_prob=0.3,
        alibi_complexity=3, motive_layers=2,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.35, evidence_difficulty_min=0.3, evidence_difficulty_max=0.7, testimony_unreliability=0.3,
        detective_miss_base=0.20, culprit_conceal_prob=0.30,
        allow_suspect_corroborators=True, max_corroborators=2, culprit_alibi_weights=(0.20, 0.35, 0.55, 0.80, 1.00),
        body_moved_prob=0.5, body_trace_ambiguity=4, trail_completeness=0.4, witness_specificity=0.3,
        freshness_threshold=1.5, step_duration_minutes=30, world_start_hour=20, num_route_constraints=2,
        max_agent_actions=100, reactive_events=True, max_objects_per_room=5
    ),
    ComplexityLevel.EXPERT: ComplexityConfig(
        num_locations=10, num_suspects=7, num_innocents=4, num_weapons=5,
        num_objects=18, num_red_herrings=5, num_time_steps=24,
        evidence_decay_rate=0.2, witness_memory_half_life=3,
        weather_change_prob=0.25, npc_move_prob=0.5, culprit_tamper_prob=0.4,
        alibi_complexity=4, motive_layers=3,
        requires_deduction=True, requires_abduction=True,
        evidence_ambiguity=0.5, evidence_difficulty_min=0.3, evidence_difficulty_max=0.8, testimony_unreliability=0.4,
        detective_miss_base=0.30, culprit_conceal_prob=0.45,
        allow_suspect_corroborators=True, max_corroborators=3, culprit_alibi_weights=(0.10, 0.20, 0.40, 0.70, 1.00),
        body_moved_prob=0.7, body_trace_ambiguity=4, trail_completeness=0.2, witness_specificity=0.2,
        freshness_threshold=1.0, step_duration_minutes=30, world_start_hour=20, num_route_constraints=3,
        max_agent_actions=150, reactive_events=True, max_objects_per_room=6,
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

    room_materials: list[tuple[str, str]] = field(default_factory=lambda: [
        # (specific_material, vague_description)
        ("fine coal dust",        "dark mineral powder"),
        ("pale sawdust",          "pale organic residue"),
        ("fireplace ash",         "grey ash residue"),
        ("iron filings",          "dark metallic particles"),
        ("chalk dust",            "white mineral powder"),
        ("dried mud",             "earthy sediment"),
        ("tobacco ash",           "grey organic residue"),
        ("plaster dust",          "white mineral dust"),
        ("chemical residue",      "acrid chemical traces"),
        ("wine stains",           "dark organic residue"),
        ("candle wax drippings",  "pale waxy residue"),
        ("fine sand",             "pale mineral granules"),
        ("wood shavings",         "pale organic residue"),
        ("damp moss",             "green organic matter"),
        ("oil stains",            "dark oily residue"),
        ("printer's ink",         "dark liquid stains"),
    ])
    
    build_types: list[str] = field(default_factory=lambda: [
        "heavy-set", "lean and tall", "stocky", "slender", "broad-shouldered",
        "wiry", "athletic", "compact", "portly", "gaunt", "barrel-chested",
        "slight of frame", "lanky", "stout", "rotund", "sinewy", "husky",
        "willowy", "rangy", "squat and broad", "angular", "paunchy", "spindly",
        "thickset", "lithe", "brawny", "frail", "petite", "towering",
        "short and stocky", "tall and slender", "powerfully built",
        "slight and wiry", "lean and angular", "narrow-shouldered",
        "wide-hipped", "long-limbed", "short-limbed", "muscular and compact",
        "soft-bodied", "trim and upright", "stooped and slight", "broad-backed",
        "narrow-waisted", "full-figured", "lean-framed", "rawboned",
        "loose-limbed", "compact and square", "well-built and upright",
        "thick-necked", "plump and short", "lean with wide shoulders",
        "barrel-bodied", "heavily muscled", "softly rounded", "sharp-shouldered",
        "sloped-shouldered", "thin through the hips", "short and round",
        "tall and angular", "big-boned", "small-boned and slight", "thick-waisted",
        "long-legged", "short-legged and sturdy", "imposingly tall",
        "flush-faced and stout", "pallid and thin", "deeply weathered and lean",
        "pear-shaped", "trim-waisted", "full-chested", "flat-chested and lean",
        "robust", "slight with a long neck", "wiry and quick-looking",
        "short with a broad chest", "tall with a slight stoop",
        "rangy and loose-jointed", "solidly built", "slender-waisted",
        "broad-girthed", "heavily set through the shoulders",
        "lean with prominent collarbones", "narrow through the chest",
        "broad through the hips and shoulders", "slightly hunched",
        "upright and square-shouldered", "stocky with short arms",
        "lean with knotted joints", "muscular arms and thin legs",
        "thin-armed and wide-hipped", "compact with a low centre of gravity",
        "long-torso'd and short-legged", "fine-boned and upright",
        "heavy through the middle", "spare and angular", "solidly fat",
        "slight with surprisingly broad hands",
    ])

    hair_types: list[str] = field(default_factory=lambda: [
        "short dark hair", "long auburn hair", "silver-streaked hair",
        "close-cropped blond hair", "curly red hair", "thinning grey hair",
        "jet-black hair tied back", "unkempt brown hair", "wavy chestnut hair",
        "cropped white hair", "thick black hair worn loose", "fine blonde hair",
        "shoulder-length brown hair", "tightly curled black hair",
        "straight copper hair", "dishevelled dark hair",
        "neat side-parted grey hair", "wild grey-streaked hair",
        "a closely shaved head", "long silver hair", "short auburn waves",
        "closely trimmed brown hair", "frizzy dark hair", "sleek black hair",
        "tousled blonde hair", "braided dark hair", "a receding dark hairline",
        "a completely bald head", "thin white hair", "coarse black hair",
        "soft brown hair worn loose", "straw-coloured hair", "glossy dark hair",
        "matted grey hair", "a neat auburn bun", "cropped salt-and-pepper hair",
        "long dark hair in a plait", "wispy blonde hair",
        "oiled black hair combed flat", "curly grey hair",
        "straight black hair to the collar", "short red hair",
        "cropped dark brown hair", "wavy silver hair", "long curly brown hair",
        "neatly parted black hair", "thinning auburn hair", "shaggy brown hair",
        "swept-back white hair", "short dark hair with a widow's peak",
        "long tangled black hair", "short wavy grey hair",
        "thin grey strands over a broad scalp", "close-cropped copper hair",
        "long fine blonde hair", "dark hair with silver temples",
        "curly blond hair", "lank brown hair", "a thick grey braid",
        "carefully styled dark hair", "dark hair streaked with white",
        "bushy auburn hair", "a shaved head with a dark shadow",
        "long grey hair pinned up", "wiry black hair",
        "fine white hair in a neat bun", "thick curly black hair",
        "straight red hair to the shoulders",
        "dark hair cropped close at the sides", "faded brown hair",
        "thick wavy black hair", "fine silver hair",
        "short grey hair neatly combed", "thinning blond hair",
        "long dark waves", "reddish-brown hair cut short", "wiry grey hair",
        "carefully groomed black hair", "long pale hair tied back",
        "curly dark hair close-cropped", "sandy-brown hair worn loose",
        "stark white hair worn long", "auburn hair cut just below the ear",
        "thin brown hair combed across a bald spot",
        "thick shoulder-length dark hair", "short grey-blond hair",
        "wavy auburn hair to the collar", "tightly pinned black hair",
        "silver hair in a loose braid", "short black hair, slightly wavy",
        "long wavy blonde hair", "dark hair with a prominent streak of white",
        "a tight auburn topknot", "coarse grey hair worn short",
        "a smoothly shaved head", "very short dark hair, almost a stubble",
        "long straight black hair parted in the middle",
        "thick wild black hair shot through with grey",
        "a neatly trimmed dark beard above a shaved scalp",
        "light brown hair with a natural wave",
    ])

    hand_types: list[str] = field(default_factory=lambda: [
        "calloused hands", "ink-stained fingers", "manicured nails",
        "grease-marked knuckles", "scarred palms", "soft unblemished hands",
        "tobacco-yellowed fingers", "paint-flecked fingers",
        "rough, cracked palms", "slender fingers with trimmed nails",
        "thick fingers with dirty nails", "delicate hands with long nails",
        "weathered, sun-darkened hands", "flour-dusted hands",
        "chemical-stained fingertips", "hands reddened from cold water",
        "soil-blackened fingernails", "bitten fingernails", "swollen knuckles",
        "heavily veined hands", "dry, papery-skinned hands",
        "an ink-stained right index finger",
        "hands with burn scars across the back", "nicotine-stained fingers",
        "chipped, unpainted nails", "heavily callused palms from rope work",
        "hands that tremble slightly", "short, stubby fingers",
        "long pianist's fingers", "a faded tattoo on the wrist",
        "copper-stained fingertips", "prominent knuckle scars",
        "soft, pale hands with neat nails", "leather-darkened hands",
        "hands rough from garden work", "deep creases in the palms",
        "small, neat hands", "large, capable hands",
        "a missing tip on the left index finger",
        "bluish veins visible through pale skin",
        "old rope burns across the palms",
        "reddened hands from hard scrubbing",
        "thin hands with protruding knuckles", "broken, uneven nails",
        "ash dust in the palm creases", "coal dust under the nails",
        "finely scarred hands from glasswork",
        "traces of silver polish on the fingers",
        "a deep cut scar across the right palm",
        "large hands with flat, wide nails", "ink-darkened cuticles",
        "a leather-hardened right thumb", "prominent liver spots",
        "smooth-backed hands with rough palms", "resin under the nails",
        "bark-roughened fingertips", "machine oil on the hands",
        "wax embedded in the palm lines", "silver nitrate stains on the fingers",
        "heavily bitten cuticles",
        "a callus on the right middle finger from writing",
        "mud dried in the hand creases",
        "powdery white residue under the nails",
        "old frostbite scarring on the fingertips",
        "a striking network of old hand scars", "very short, practical nails",
        "copper wire cuts on the fingertips",
        "ink staining on the left hand only",
        "heavily tanned hands with pale nail beds",
        "charcoal smeared across the right palm",
        "peeling skin on the knuckles", "salt-dried skin from sea work",
        "a prominent trigger callus on the right index finger",
        "silver filings in the knuckle creases", "thread cuts between the fingers",
        "beeswax on the fingertips", "dark tannin stains from bookbinding",
        "long-fingered hands with clipped, practical nails",
        "paint under the left thumbnail only",
        "a distinctive scar running across the right wrist",
        "hands worn smooth by years of polishing",
        "dark henna staining on the fingers", "fresh abrasions on the knuckles",
        "blue-black ink ground into the finger whorls",
        "a blacksmith's wide, flat thumbnails",
        "skin stretched tight over prominent hand bones",
        "green herb staining under the nails",
        "pale, soft skin from constant glove-wearing",
        "a missing little finger on the right hand",
        "split skin at the base of each thumb",
        "flour under the nails and in the knuckle creases",
        "a callus on the right ring finger from a signet ring",
        "pale hands with prominent tendons",
        "dark staining between the right thumb and forefinger",
        "hands with a bluish-grey tinge from silverworking",
        "a crudely bandaged left palm",
        "very fine scars across the fingertips from needlework",
        "deep longitudinal creases on all fingers",
        "the square, flat nails of a stonemason",
        "hands with a faint smell of sulphur from match-work",
    ])
DEFAULT_ASSET_POOL = AssetPool()
