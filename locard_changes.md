# Locard Triangle: Code Changes

Follows `claude.md`: surgical changes only, no speculative features, each change traces to a requirement.

**Assumptions stated upfront:**
- `contact_timestamp` unit = game steps (same as `murder_step`). It's a fixed historical fact, not a runtime value.
- Freshness formula: `is_fresh = abs(contact_timestamp - murder_step) < freshness_threshold`
- Each evidence has exactly one `EdgeRelevance` (one triangle edge per clue). Multi-edge evidence was overengineered — if a clue supports two edges, generate two evidence records.
- ANALYZE costs one action and returns one temporal assessment for one evidence ID.
- The agent learns object names from EXAMINE_LOCATION output, then uses them in EXAMINE_OBJECT (case-insensitive match, existing behavior).

**Plan:**
1. Add Locard types to entities.py -> verify: new classes importable
2. Add freshness_threshold to config -> verify: all 5 presets updated
3. Add body visibility + new actions to world.py -> verify: agent can see body, examine it, analyze evidence, accuse with triangle evidence
4. Add body object + edge relevances to generator.py -> verify: generated mysteries have triangle evidence
5. Add triangle solvability check to verify.py -> verify: generated mysteries pass check
6. Add Locard scores to metrics.py -> verify: new fields in output

---

## File 1: `mystery_world/entities.py`

### 1a. Add Locard types after line 14

After `from typing import Any`:

```python
# ---------------------------------------------------------------------------
# Locard triangle types
# ---------------------------------------------------------------------------

class EdgeType(Enum):
    SUSPECT_WEAPON = auto()
    WEAPON_VICTIM = auto()
    SUSPECT_ROOM = auto()


class TemporalLabel(Enum):
    """What the agent sees about evidence freshness (never the raw timestamp)."""
    CLEARLY_FRESH = auto()   # easy: "still wet", "warm to touch"
    CLEARLY_STALE = auto()   # easy: "dusty", "dried and cracked"
    AMBIGUOUS = auto()       # medium+: no obvious age indicator, must ANALYZE


@dataclass
class EdgeRelevance:
    """Links one piece of evidence to one triangle edge with temporal metadata."""
    edge_type: EdgeType = EdgeType.SUSPECT_WEAPON
    subject_ids: list[str] = field(default_factory=list)  # entity IDs involved
    contact_timestamp: float = 0.0   # game step when the contact happened (fixed historical fact)
    surface_label: TemporalLabel = TemporalLabel.AMBIGUOUS  # what the agent sees

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
```

### 1b. Add PhysicalTraits before Character (after line 57, after Location)

```python
@dataclass
class PhysicalTraits:
    """Observable physical characteristics — matched to evidence clues."""
    build: str = ""     # "heavy-set", "lean and tall", "stocky", etc.
    hair: str = ""      # "short dark hair", "long auburn hair", etc.
    hands: str = ""     # "calloused hands", "ink-stained fingers", etc.

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhysicalTraits:
        return cls(**d)
```

### 1c. Add physical_traits field to Character

After line 93 (`inventory: list[str] = field(default_factory=list)`):

```python
    physical_traits: PhysicalTraits = field(default_factory=PhysicalTraits)
```

### 1d. Update Character.from_dict (lines 113-121)

Replace with:

```python
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
        return cls(**d)
```

### 1e. Add `relevance` field to Evidence

After line 161 (`degraded_at_step: int | None = None`):

```python
    # --- Locard triangle ---
    relevance: EdgeRelevance | None = None  # which triangle edge this evidence supports
```

### 1f. Update Evidence.to_dict (lines 166-170)

Replace with:

```python
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["evidence_type"] = self.evidence_type.name
        d["state"] = self.state.name
        d["relevance"] = self.relevance.to_dict() if self.relevance else None
        return d
```

### 1g. Update Evidence.from_dict (lines 172-177)

Replace with:

```python
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        d = dict(d)
        d["evidence_type"] = EvidenceType[d["evidence_type"]]
        d["state"] = EvidenceState[d["state"]]
        rel = d.get("relevance")
        d["relevance"] = EdgeRelevance.from_dict(rel) if rel else None
        return cls(**d)
```

### 1h. Add EdgeArgument and ScoreResult after line 222

After `TimelineEntry.from_dict` (end of file):

```python


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
    """Output of the Locard triangle scoring function."""
    correct_suspect: bool = False
    correct_weapon: bool = False
    correct_room: bool = False
    accusation_score: float = 0.0

    suspect_weapon_score: float = 0.0
    weapon_victim_score: float = 0.0
    suspect_room_score: float = 0.0
    triangle_score: float = 0.0

    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
```

---

## File 2: `mystery_world/__init__.py`

### 2a. Add freshness_threshold to ComplexityConfig

After line 67 (`witness_specificity: float = 1.0`):

```python
    freshness_threshold: float = 2.0   # game-steps: |contact_time - murder_time| < this -> fresh
```

### 2b. Add to each preset after the `witness_specificity` line

| Preset | Line | Value |
|--------|------|-------|
| TRIVIAL | 112 (after `max_agent_actions=30,`) | `freshness_threshold=3.0,` |
| EASY | 124 (after `max_agent_actions=40,`) | `freshness_threshold=2.5,` |
| MEDIUM | 136 (after `max_agent_actions=60,`) | `freshness_threshold=2.0,` |
| HARD | 148 (after `max_agent_actions=80, reactive_events=True,`) | `freshness_threshold=1.5,` |
| EXPERT | 160 (after `max_agent_actions=120, reactive_events=True,`) | `freshness_threshold=1.0,` |

### 2c. Add trait pools to AssetPool

After `room_materials` (line 246, before `DEFAULT_ASSET_POOL`):

```python
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
```

---

## File 3: `mystery_world/world.py`

### 3a. Update imports (lines 21-29)

Replace:

```python
from mystery_world.entities import (
    Character,
    CharacterRole,
    Evidence,
    EvidenceState,
    Location,
    TimelineEntry,
    WorldObject,
)
```

with:

```python
from mystery_world.entities import (
    Character,
    CharacterRole,
    EdgeArgument,
    EdgeRelevance,
    EdgeType,
    Evidence,
    EvidenceState,
    Location,
    ScoreResult,
    TemporalLabel,
    TimelineEntry,
    WorldObject,
)
```

### 3b. Show the victim's body and character descriptions in observe_location

Replace lines 207-214:

```python
        # Characters present
        chars_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if cid in self._state.characters and self._state.characters[cid].is_alive
        ]
        if chars_here:
            names = ", ".join(c.full_name for c in chars_here)
            parts.append(f"Present here: {names}.")
```

with:

```python
        # Characters present (alive) — include physical description
        chars_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if cid in self._state.characters and self._state.characters[cid].is_alive
        ]
        for c in chars_here:
            pt = c.physical_traits
            desc = f"{c.full_name} — {pt.build}, {pt.hair}, {pt.hands}."
            parts.append(desc)
        # Dead bodies
        dead_here = [
            self._state.characters[cid]
            for cid in loc.characters_here
            if cid in self._state.characters and not self._state.characters[cid].is_alive
        ]
        for d in dead_here:
            parts.append(f"The body of {d.full_name} lies here.")
```

### 3c. Add ANALYZE to AgentAction

After line 43 (`ACCUSE = auto()`) add:

```python
    ANALYZE = auto()            # spend action to get temporal assessment of evidence
```

### 3d. Add murder_timestamp and freshness_threshold to WorldState

After line 86 (`murder_step: int = 0`), add:

```python
    murder_timestamp: float = 0.0   # same as murder_step, as float for scoring
    freshness_threshold: float = 2.0
```

### 3e. Update serialization

In `to_dict` (line 99), after `"murder_step": self.murder_step,` (line 116), add:

```python
            "murder_timestamp": self.murder_timestamp,
            "freshness_threshold": self.freshness_threshold,
```

In `load` (line 126), after `murder_step=d["murder_step"],` (line 139), add:

```python
            murder_timestamp=d.get("murder_timestamp", float(d["murder_step"])),
            freshness_threshold=d.get("freshness_threshold", 2.0),
```

### 3f. Register ANALYZE in dispatch

In `_dispatch_action` (line 276), add to the `handlers` dict after the ACCUSE entry (line 283):

```python
            AgentAction.ANALYZE: self._handle_analyze,
```

### 3g. Expose evidence IDs in observations

**EXAMINE_OBJECT** (line 319): Replace line 330:

```python
                        parts.append(f"This is evidence: {ev.description}")
```

with:

```python
                        parts.append(f"[Evidence {ev.id}] {ev.description}")
```

**SEARCH** (line 398): Replace line 420:

```python
                parts.append(f"  • {ev.name}: {ev.description}")
```

with:

```python
                parts.append(f"  • {ev.name} [{eid}]: {ev.description}")
```

**CHECK_INVENTORY** (line 457): Replace line 464:

```python
                parts.append(f"- {ev.name} [{ev.evidence_type.name}] ({ev.state.name}): {ev.description}")
```

with:

```python
                parts.append(f"- [{eid}] {ev.name} [{ev.evidence_type.name}] ({ev.state.name}): {ev.description}")
```

### 3h. Replace _handle_accuse (lines 426-450)

ACCUSE now accepts optional evidence IDs for each triangle edge.

Replace lines 426-450 with:

```python
    def _handle_accuse(
        self,
        suspect_name: str = "",
        weapon_name: str = "",
        location_name: str = "",
        suspect_weapon_evidence: list[str] | None = None,
        weapon_victim_evidence: list[str] | None = None,
        suspect_room_evidence: list[str] | None = None,
        **_: Any,
    ) -> ActionResult:
        """Final accusation. Ends the episode.

        Args:
            suspect_name:             who did it (by name)
            weapon_name:              with what (by name)
            location_name:            where (by name)
            suspect_weapon_evidence:  evidence IDs linking suspect to weapon
            weapon_victim_evidence:   evidence IDs linking weapon to victim
            suspect_room_evidence:    evidence IDs linking suspect to room

        """
        self.is_solved = True
        culprit = self._state.get_culprit()
        weapon = self._state.objects.get(self._state.murder_weapon_id)
        murder_loc = self._state.locations.get(self._state.murder_location_id)

        correct_suspect = culprit and culprit.full_name.lower() == suspect_name.lower()
        correct_weapon = weapon and weapon.name.lower() == weapon_name.lower()
        correct_location = murder_loc and murder_loc.name.lower() == location_name.lower()

        self.accusation_correct = correct_suspect and correct_weapon and correct_location

        details = {
            "suspect_correct": correct_suspect,
            "weapon_correct": correct_weapon,
            "location_correct": correct_location,
            "partial_score": sum([correct_suspect, correct_weapon, correct_location]) / 3.0,
        }

        # --- Locard scoring (if any triangle evidence provided) ---
        has_locard = any([suspect_weapon_evidence, weapon_victim_evidence, suspect_room_evidence])
        if has_locard:
            accused_ids = self._resolve_names_to_ids(suspect_name, weapon_name, location_name)

            triangle = {}
            if suspect_weapon_evidence:
                triangle["SUSPECT_WEAPON"] = EdgeArgument(evidence_ids=suspect_weapon_evidence)
            if weapon_victim_evidence:
                triangle["WEAPON_VICTIM"] = EdgeArgument(evidence_ids=weapon_victim_evidence)
            if suspect_room_evidence:
                triangle["SUSPECT_ROOM"] = EdgeArgument(evidence_ids=suspect_room_evidence)

            score = score_accusation(
                accused_ids=accused_ids,
                triangle=triangle,
                state=self._state,
            )
            details["score_result"] = score.to_dict()
            details["triangle_score"] = score.triangle_score
            details["composite_score"] = score.composite_score

        if self.accusation_correct:
            obs = f"CORRECT! {culprit.full_name} committed the crime with the {weapon.name} in the {murder_loc.name}."
        else:
            obs = f"INCORRECT. The true answer: {culprit.full_name if culprit else '?'} with the {weapon.name if weapon else '?'} in the {murder_loc.name if murder_loc else '?'}."

        if has_locard:
            obs += f" Triangle: {score.triangle_score:.1f}/3. Composite: {score.composite_score:.2f}."

        return ActionResult(True, obs, details=details)

    def _resolve_names_to_ids(self, suspect_name: str, weapon_name: str, location_name: str) -> dict[str, str]:
        """Resolve agent-provided names to internal IDs."""
        result = {"suspect": "", "weapon": "", "room": ""}
        for cid, c in self._state.characters.items():
            if c.full_name.lower() == suspect_name.lower():
                result["suspect"] = cid
                break
        for oid, o in self._state.objects.items():
            if o.name.lower() == weapon_name.lower():
                result["weapon"] = oid
                break
        for lid, l in self._state.locations.items():
            if l.name.lower() == location_name.lower():
                result["room"] = lid
                break
        return result
```

### 3i. Add _handle_analyze

After `_handle_accuse`. One action, one evidence ID, one assessment.

```python
    def _handle_analyze(self, evidence_id: str = "", **_: Any) -> ActionResult:
        """Spend one action to get a temporal assessment of one piece of evidence."""
        if evidence_id not in self._discovered_evidence:
            return ActionResult(False, f"Evidence '{evidence_id}' not in your collection. Discover it first.")

        ev = self._state.evidence.get(evidence_id)
        if ev is None:
            return ActionResult(False, f"Evidence '{evidence_id}' does not exist.")

        if ev.relevance is None:
            return ActionResult(True, f"You analyze {ev.name}. No forensically relevant contact traces found.")

        rel = ev.relevance
        freshness = abs(rel.contact_timestamp - self._state.murder_timestamp)
        threshold = self._state.freshness_threshold

        if rel.surface_label == TemporalLabel.CLEARLY_FRESH:
            assessment = "This trace appears very recent — consistent with the time of the murder."
        elif rel.surface_label == TemporalLabel.CLEARLY_STALE:
            assessment = "This trace is old — clearly predates the murder by a significant margin."
        else:  # AMBIGUOUS
            if freshness < threshold:
                assessment = "Analysis suggests this trace is relatively recent, possibly within the relevant timeframe."
            else:
                assessment = "Analysis suggests this trace may be older than it first appears."

        return ActionResult(True, f"You analyze {ev.name}. {assessment}")
```

### 3j. Add scoring functions at end of world.py

After the `MysteryEnvironment` class (after line 553):

```python


# ---------------------------------------------------------------------------
# Locard triangle scoring
# ---------------------------------------------------------------------------

def score_accusation(
    accused_ids: dict[str, str],
    triangle: dict[str, EdgeArgument],
    state: WorldState,
) -> ScoreResult:
    """Score an accusation: correct culprit + triangle evidence quality.

    Args:
        accused_ids: {"suspect": id, "weapon": id, "room": id}
        triangle: EdgeType.name -> EdgeArgument (evidence IDs the agent cited)
        state: ground truth
    """
    result = ScoreResult()
    murder_ts = state.murder_timestamp
    threshold = state.freshness_threshold

    # --- Score 1: Accusation (backward-compatible) ---
    result.correct_suspect = accused_ids.get("suspect") == state.culprit_id
    result.correct_weapon = accused_ids.get("weapon") == state.murder_weapon_id
    result.correct_room = accused_ids.get("room") == state.murder_location_id
    result.accusation_score = sum([result.correct_suspect, result.correct_weapon, result.correct_room]) / 3.0

    # --- Score 2: Triangle (precision-weighted per edge) ---
    def _score_edge(edge_type: EdgeType) -> float:
        arg = triangle.get(edge_type.name)
        if arg is None or not arg.evidence_ids:
            return 0.0

        total_cited = len(arg.evidence_ids)
        correct_fresh = 0
        correct_stale = 0

        for eid in arg.evidence_ids:
            ev = state.evidence.get(eid)
            if ev is None or ev.is_red_herring or ev.relevance is None:
                continue
            rel = ev.relevance
            if rel.edge_type != edge_type:
                continue
            if not _relevance_matches_truth(rel, edge_type, state):
                continue
            if abs(rel.contact_timestamp - murder_ts) < threshold:
                correct_fresh += 1
            else:
                correct_stale += 1

        if correct_fresh > 0:
            return correct_fresh / total_cited
        elif correct_stale > 0:
            return 0.5 * (correct_stale / total_cited)
        return 0.0

    result.suspect_weapon_score = _score_edge(EdgeType.SUSPECT_WEAPON)
    result.weapon_victim_score = _score_edge(EdgeType.WEAPON_VICTIM)
    result.suspect_room_score = _score_edge(EdgeType.SUSPECT_ROOM)
    result.triangle_score = result.suspect_weapon_score + result.weapon_victim_score + result.suspect_room_score

    # --- Composite ---
    result.composite_score = (
        0.50 * result.accusation_score
        + 0.50 * (result.triangle_score / 3.0)
    )
    return result


def _relevance_matches_truth(rel: EdgeRelevance, edge_type: EdgeType, state: WorldState) -> bool:
    if edge_type == EdgeType.SUSPECT_WEAPON:
        return state.culprit_id in rel.subject_ids and state.murder_weapon_id in rel.subject_ids
    elif edge_type == EdgeType.WEAPON_VICTIM:
        return state.murder_weapon_id in rel.subject_ids and state.victim_id in rel.subject_ids
    elif edge_type == EdgeType.SUSPECT_ROOM:
        return state.culprit_id in rel.subject_ids and state.murder_location_id in rel.subject_ids
    return False
```

---

## File 4: `mystery_world/generator.py`

### 4a. Update imports (lines 26-37)

Replace:

```python
from mystery_world.entities import (
    Character,
    CharacterRole,
    Evidence,
    EvidenceState,
    EvidenceType,
    Location,
    LocationTag,
    Relationship,
    TimelineEntry,
    WorldObject
)
```

with:

```python
from mystery_world.entities import (
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
    TemporalLabel,
    TimelineEntry,
    WorldObject,
)
```

### 4b. Add surface label helper after `_uid` (after line 41)

```python
def _pick_surface_label(
    contact_ts: float,
    murder_ts: float,
    threshold: float,
    config: ComplexityConfig,
) -> TemporalLabel:
    """Choose what the agent sees about freshness, based on difficulty."""
    is_fresh = abs(contact_ts - murder_ts) < threshold
    if config.evidence_ambiguity <= 0.1:       # easy
        return TemporalLabel.CLEARLY_FRESH if is_fresh else TemporalLabel.CLEARLY_STALE
    else:                                       # medium+
        return TemporalLabel.AMBIGUOUS
```

### 4c. Add trait assignment in `_generate_characters` (lines 96-166)

After the character creation loop (line 119, after `char_list.append(char)`), and before "Assign roles" (line 121), insert:

```python
    # Assign physical traits — initially all unique per character
    builds = list(rng.choice(pool.build_types, size=min(total, len(pool.build_types)), replace=False))
    hairs = list(rng.choice(pool.hair_types, size=min(total, len(pool.hair_types)), replace=False))
    hands = list(rng.choice(pool.hand_types, size=min(total, len(pool.hand_types)), replace=False))
    for i, char in enumerate(char_list):
        char.physical_traits = PhysicalTraits(
            build=str(builds[i]),
            hair=str(hairs[i]),
            hands=str(hands[i]),
        )

    # MEDIUM+: no single trait may uniquely identify the culprit.
    # Each of the culprit's trait values must appear on at least one other suspect,
    # so the agent is forced to intersect all three traits rather than solve from one clue.
    # char_list[1] = culprit, char_list[2:1+num_suspects] = other suspects (roles assigned below).
    if config.evidence_ambiguity > 0.1:
        culprit_char = char_list[1]
        other_suspects = char_list[2:1 + config.num_suspects]
        if other_suspects:
            for trait_name in ("build", "hair", "hands"):
                culprit_val = getattr(culprit_char.physical_traits, trait_name)
                sharers = [
                    c for c in other_suspects
                    if getattr(c.physical_traits, trait_name) == culprit_val
                ]
                if not sharers:
                    # Force one randomly chosen other suspect to share this trait value
                    target = other_suspects[int(rng.integers(0, len(other_suspects)))]
                    pt = target.physical_traits
                    target.physical_traits = PhysicalTraits(
                        build=culprit_val if trait_name == "build" else pt.build,
                        hair=culprit_val if trait_name == "hair" else pt.hair,
                        hands=culprit_val if trait_name == "hands" else pt.hands,
                    )
```

**Why this works**: at TRIVIAL/EASY all traits remain unique (sampled without replacement), so one clue is enough. At MEDIUM+, each individual trait matches multiple suspects, but only the culprit holds all three simultaneously — the agent must intersect all three evidence pieces to isolate one person.

Also add `PhysicalTraits` to the import from `mystery_world.entities` (change 4a).

### 4d. Replace `_generate_evidence_and_objects` entirely (lines 172-362)

Evidence descriptions now reference physical traits instead of naming characters directly. The agent must observe suspects (see 3b — physical descriptions shown in `observe_location`) and match traits to evidence clues.

```python
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
    weapon_names = list(rng.choice(pool.weapon_templates, size=min(config.num_weapons, len(pool.weapon_templates)), replace=False))
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

    # --- SUSPECT_WEAPON evidence (on the murder weapon) ---
    if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
        mw_linked_id = str(rng.choice(non_culprit_suspects))
    else:
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

    # --- WEAPON_VICTIM evidence (victim's blood on weapon) ---
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

    # --- SUSPECT_ROOM evidence (culprit in murder room) ---
    culprit_traits = _traits(culprit_id)
    sr_ev = Evidence(
        id=_uid("ev", rng),
        name=f"shoe scuffs in the {murder_loc.name if murder_loc else 'crime scene'}",
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
    # Templates: each entry is (name_template, description_template)
    # Description templates use {trait} which gets filled with the linked character's trait
    ev_trait_templates = [
        ("fingerprint on doorknob", "A partial print from someone with {hands} found on a doorknob."),
        ("hair strand", "A strand of {hair} caught on a rough surface."),
        ("torn fabric from clothing", "A torn fabric scrap — looks like it belongs to someone {build}."),
        ("footprint near the scene", "A footprint suggesting a {build} individual."),
        ("smudged handprint on wall", "A smudged handprint from someone with {hands}."),
        ("scratches on nearby furniture", "Scratches consistent with someone who has {hands}."),
        ("fiber transfer on victim's coat", "Fibers transferred by contact with a {build} person."),
        ("cigarette butt near the scene", "A cigarette butt with lip marks — consistent with someone who has {hair}."),
        ("partial shoe impression in mud", "A shoe impression suggesting a {build} person."),
        ("broken button from a jacket", "A button from a garment sized for someone {build}."),
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

        if linked_id == culprit_id:
            contact_ts = murder_ts + float(rng.uniform(-0.5, 0.5))
        else:
            contact_ts = murder_ts - float(rng.uniform(threshold, threshold * 3))

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

    # --- Testimonial evidence ---
    for sid in suspect_ids:
        if sid == culprit_id:
            continue
        eid = _uid("ev", rng)
        reliable = rng.random() >= config.testimony_unreliability
        evidence[eid] = Evidence(
            id=eid, name="testimony about suspect movements",
            evidence_type=EvidenceType.TESTIMONIAL,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=sid,
            description="A witness account regarding a suspect's whereabouts.",
            discovery_difficulty=_sample_difficulty(),
            is_reliable=reliable,
            relevance=EdgeRelevance(
                edge_type=EdgeType.SUSPECT_ROOM,
                subject_ids=[sid, str(rng.choice(location_ids))],
                contact_timestamp=murder_ts,
                surface_label=TemporalLabel.AMBIGUOUS,
            ),
        )

    # --- Documentary evidence (unchanged, no relevance) ---
    doc_templates = [
        "a threatening letter", "a financial ledger entry",
        "a diary page with incriminating passage", "a forged alibi note",
        "a signed insurance policy", "a photograph with a revealing timestamp",
        "a bank withdrawal receipt", "a phone message transcript",
        "a torn contract", "a secret correspondence",
    ]
    for i in range(min(2, config.motive_layers)):
        eid = _uid("ev", rng)
        if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
            doc_linked_id = str(rng.choice(non_culprit_suspects))
        else:
            doc_linked_id = culprit_id
        evidence[eid] = Evidence(
            id=eid, name=str(rng.choice(doc_templates)),
            evidence_type=EvidenceType.DOCUMENTARY,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=doc_linked_id,
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

    # --- Generic objects ---
    obj_names = list(rng.choice(pool.object_templates, size=min(config.num_objects, len(pool.object_templates)), replace=False))
    for oname in obj_names:
        oid = _uid("obj", rng)
        objects[oid] = WorldObject(
            id=oid, name=str(oname), description=f"A {oname} lying here.",
            location_id=str(rng.choice(location_ids)),
            portable=rng.random() < 0.7,
        )

    return evidence, objects
```

### 4e. Update call in generate_mystery (lines 978-982)

Replace:

```python
        evidence, objects = _generate_evidence_and_objects(
            config, pool, rng, location_ids,
            culprit.id, victim.id, suspect_ids,
            murder_weapon_id, murder_location_id, locations
        )
```

with:

```python
        evidence, objects = _generate_evidence_and_objects(
            config, pool, rng, location_ids,
            culprit.id, victim.id, suspect_ids,
            murder_weapon_id, murder_location_id, locations,
            characters=characters,
            murder_step=murder_step,
        )
```

### 4f. Create victim body object in generate_mystery

After `evidence.update(cs_evidence)` block (~line 990), add:

```python
        # Victim's body — wound description hints at weapon type, no evidence link
        body_obj_id = _uid("obj", rng)
        weapon_obj = objects.get(murder_weapon_id)
        weapon_name = weapon_obj.name if weapon_obj else "an unknown weapon"
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
        victim.location_id = body_location_id
```

### 4g. Set WorldState fields in generate_mystery

In the `WorldState(...)` constructor (~line 1013), after `murder_step=murder_step,` (line 1028), add:

```python
            murder_timestamp=float(murder_step),
            freshness_threshold=config.freshness_threshold,
```

---

## File 5: `benchmark/verify.py`

### 5a. Update imports (line 20)

Replace:

```python
from mystery_world.entities import CharacterRole, EvidenceState, EvidenceType
```

with:

```python
from mystery_world.entities import CharacterRole, EdgeType, EvidenceState, EvidenceType
```

### 5b. Add check_locard_solvability after check_solvability (after line 178)

```python


def check_locard_solvability(state: WorldState) -> dict[str, Any]:
    """Verify the Locard triangle is closable: each edge has at least one
    discoverable fresh evidence pointing to the correct entities."""
    issues: list[str] = []
    murder_ts = state.murder_timestamp
    threshold = state.freshness_threshold

    for edge in EdgeType:
        found = False
        for ev in state.evidence.values():
            if ev.is_red_herring or ev.state == EvidenceState.DESTROYED or ev.discovery_difficulty >= 1.0:
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
```

---

## File 6: `evaluation/metrics.py`

### 6a. Add fields to EpisodeMetrics

After line 34 (`partial_score: float = 0.0`):

```python
    # --- Locard triangle ---
    accusation_score: float = 0.0
    triangle_score: float = 0.0
    suspect_weapon_score: float = 0.0
    weapon_victim_score: float = 0.0
    suspect_room_score: float = 0.0
    composite_score: float = 0.0
```

### 6b. Add to EpisodeMetrics.to_dict

Before the closing `}` (line 81):

```python
            "accusation_score": self.accusation_score,
            "triangle_score": self.triangle_score,
            "suspect_weapon_score": self.suspect_weapon_score,
            "weapon_victim_score": self.weapon_victim_score,
            "suspect_room_score": self.suspect_room_score,
            "composite_score": self.composite_score,
```

### 6c. Add fields to AggregateMetrics

After line 175 (`std_solve_rate: float = 0.0`):

```python
    mean_triangle_score: float = 0.0
    mean_composite_score: float = 0.0
```

### 6d. Add to AggregateMetrics.to_dict

After `"std_solve_rate"` line (line 188):

```python
            "mean_triangle_score": round(self.mean_triangle_score, 4),
            "mean_composite_score": round(self.mean_composite_score, 4),
```

### 6e. Compute in aggregate_metrics

In the `AggregateMetrics(...)` constructor (line 199, before closing `)`):

```python
        mean_triangle_score=sum(e.triangle_score for e in level_eps) / n,
        mean_composite_score=sum(e.composite_score for e in level_eps) / n,
```
