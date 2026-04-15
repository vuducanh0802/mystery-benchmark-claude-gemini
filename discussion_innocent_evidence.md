# Discussion: Ground-Truth Evidence for Eliminating Innocent Suspects

I'm building a benchmark for evaluating LLM agents on murder-mystery solving. The
agent must accuse one suspect and also justify **why every other suspect is
innocent** by citing specific evidence IDs. I need help thinking through how the
world generator should create evidence that can legitimately exonerate innocents.

## The Benchmark Setup

- 4-6 suspects per case (1 culprit, rest innocent). 1 victim.
- Agent operates in a grid of interconnected rooms under a fixed action budget.
- Final accusation scoring has 4 components:
  1. **Accusation correctness (35%)** — right suspect/weapon/location
  2. **Locard triangle (35%)** — physical evidence for 3 edges (suspect-weapon,
     weapon-victim, suspect-room), scored by precision + recall (F1)
  3. **Alibi verification (15%)** — culprit's stated alibi + logical contradiction
  4. **Elimination (15%)** — for each other suspect, cite evidence IDs that
     establish their innocence
- Elimination scoring: `(correct - 2 * incorrect) / total_innocents`. The agent
  must provide `{suspect_name: [evidence_ids]}`. Incorrect eliminations (citing
  evidence for the true culprit, or citing a red herring) are penalised at 2x.

## The Current Generator State

When the world is generated, evidence items are created with these fields:

```python
class Evidence:
    id: str
    name: str
    description: str
    location_id: str
    linked_character_id: str | None      # whom the evidence "points at"
    evidence_type: PHYSICAL | TESTIMONIAL | DOCUMENTARY
    is_red_herring: bool
    discovery_difficulty: float
    relevance: EdgeRelevance | None      # Locard edge + subject_ids + timestamp
```

Here's what the generator currently produces for innocent suspects:

| Evidence type | For culprit | For innocents |
|---|---|---|
| Physical (grip marks, fibres, shoe scuffs) | Timestamps at murder time, placing them at crime | Same type, but some are **ambiguity-rerouted** to point at a random innocent with a timestamp *before* the murder window |
| Testimonial | Not created for culprit | One per innocent: "testimony about suspect movements," places them at a **random** room at murder time |
| Documentary (letters, ledgers) | Sometimes points at culprit | Sometimes ambiguity-rerouted to innocents — only motive support, no alibi |

**Critical gap:** there is no evidence designed to *prove innocence*. The
testimonial evidence places innocents at random rooms (which might accidentally
be the murder room, giving an agent reason to wrongly accuse). The physical
ambiguity evidence is meant to **mislead** by pointing at innocents with
mismatched timestamps — it's noise, not exoneration.

## The Scoring Problem

For the elimination score to be meaningful, the environment must make it
*possible* for an agent to justify innocence with concrete evidence. Right now:

- If the agent cites the innocent-directed ambiguity evidence, the scoring code
  just checks "is this suspect actually not the culprit? yes → correct." It
  doesn't verify that the cited evidence *relates to* the eliminated suspect.
- An agent could pass every innocent evidence ID that exists in the world and
  still get full elimination credit, as long as those evidence items aren't red
  herrings — even if they're completely unrelated to the suspects being cleared.

So two issues need solving together:

1. **Generator:** Create evidence items that genuinely exonerate innocents
   (e.g., alibi-confirming placement at a verifiable non-murder location at
   murder time).
2. **Scoring:** Tighten the elimination check so the cited evidence must
   plausibly support *that specific suspect's* innocence.

## What "evidence of innocence" could mean

A few candidate forms:

- **Alibi-confirming placement evidence:** a physical or testimonial trace that
  places the innocent suspect at a **non-murder** location at the **murder
  timestamp**. Example: "coat checked in at the cloakroom at 9:30 PM" with
  `subject_ids=[suspect_id, cloakroom_id]` and `contact_timestamp=murder_ts`.
- **Corroborating witness testimony:** another character independently confirms
  the innocent was elsewhere at murder time. Requires cross-referencing two
  testimonies.
- **Physical incompatibility:** evidence shows the culprit had trait X (e.g.,
  left-handed) and this suspect is right-handed. Indirect — requires Locard
  triangle evidence too.
- **Missing trace:** the murder room has only one set of shoe prints, and they
  don't match this suspect's shoes. Requires the agent to infer absence.

## Design Questions

### 1. What form should exonerating evidence take?

The benchmark already has `EdgeRelevance` for the Locard triangle (3 edge types:
SUSPECT_WEAPON, WEAPON_VICTIM, SUSPECT_ROOM). Should I:

- (a) **Add a new edge type** `SUSPECT_ELSEWHERE` with `subject_ids=[innocent_id,
  non_murder_location_id]` and `contact_timestamp=murder_ts`? This fits the
  existing relevance model and is easy to validate.
- (b) **Reuse SUSPECT_ROOM** but with a non-murder location? The scoring would
  need to distinguish "placed at crime scene" from "placed elsewhere."
- (c) **Add a separate field** `alibi_confirms_id: str` on Evidence that names
  the suspect whose alibi this corroborates? Cleaner semantics, but adds a new
  concept.
- (d) **Something else** entirely?

### 2. Where should the exonerating evidence live?

If the innocent was in the cloakroom at murder time, the exonerating trace
should logically be in the cloakroom (or on their coat, or in a register book).
Should the generator:

- Place the evidence in the room the innocent claims to have been in?
  (Requires generating alibi rooms for innocents, which doesn't currently
  happen.)
- Place it somewhere random to avoid telegraphing "this is the alibi room"?
- Always place it in a specific class of rooms (e.g., "public" rooms like
  cloakroom, dining hall) that wouldn't be the murder scene?

### 3. How many exonerating items per innocent?

- **1 item per innocent** — minimal; the agent must find every exoneration.
- **2 items per innocent** — redundancy; allows the agent to miss one and still
  clear the suspect via the other. Also enables precision/recall scoring on
  elimination just like the triangle.
- **Variable by difficulty** — TRIVIAL=1, EXPERT=3.

### 4. How strict should the elimination scoring be?

The current proposal (`correct - 2 * incorrect`) just asks "is the suspect
innocent?" It doesn't verify the cited evidence relates to that specific suspect.
Tighten options:

- (a) **Check `linked_character_id == suspect_id`** — requires the exonerating
  evidence to be tagged with the cleared suspect's ID.
- (b) **Check `relevance.subject_ids contains suspect_id`** — uses the existing
  relevance model.
- (c) **F1 on elimination too** — require the agent to cite the *full set* of
  exonerating evidence for a suspect, not just one piece.

### 5. How do I prevent accidental telegraphing?

If the generator marks certain evidence as "clears suspect X," an agent that
blindly dumps all discovered evidence into `eliminations` might get credit
without reasoning. Ways to prevent this:

- The evidence itself doesn't say "this clears X" in the description — the
  agent must infer the clearance from the location/timestamp/subject logic.
- Scoring does the check: agent provides evidence ID + suspect name, scoring
  validates that this evidence actually places that suspect elsewhere at
  murder time.
- Both — the evidence semantics are hidden in the description, but the
  underlying relevance data is checkable.

### 6. What about the culprit's alibi evidence?

The culprit **claims** an alibi that turns out to be false. If the generator
creates alibi-confirming evidence for innocents, should it also create
alibi-*refuting* evidence for the culprit? Currently, the culprit's alibi is
contradicted via "physical evidence places them at the crime scene" — which is
the existing Locard triangle. So maybe innocence and contradiction are
symmetrical: innocents have placement-elsewhere evidence, culprit has
placement-at-crime-scene evidence (already covered by the triangle).

### 7. Should innocents' "evidence" be elicited through interviews, physical
   discovery, or both?

If it's pure physical (left in rooms), the agent finds it through EXAMINE_OBJECT.
If it's testimony ("I saw Alice in the cloakroom"), the agent needs TALK_TO.
Mixing forces the agent to use both channels. Issues:

- Testimonial evidence can be unreliable (the witness might lie or mistake).
  Should alibi-confirming testimony always be reliable, or can innocents have
  false alibis pointed at them by unreliable witnesses?
- If testimony is the only source, the agent needs a way to *cite* the
  testimony as an evidence ID in the elimination dict. The current generator
  does create testimonial evidence objects with IDs, so this works.

## What I'm looking for from you

Challenge any of these framings. Suggest a minimal design that:

- Makes elimination a real reasoning challenge (not just "dump all evidence").
- Keeps the generator implementation simple (no elaborate alibi story trees).
- Fits the existing `Evidence` / `EdgeRelevance` / `Locard` model where possible.
- Makes the scoring deterministic and verifiable from ground truth (no NLP).

In particular, I'm torn between:

- **Minimal option:** Extend `EdgeRelevance` with a new edge type
  `SUSPECT_ELSEWHERE`, generate 1-2 such items per innocent, tighten scoring to
  require `relevance.edge_type == SUSPECT_ELSEWHERE` AND `suspect_id in
  subject_ids` AND `timestamp ≈ murder_ts`.
- **Richer option:** Add a full innocent-alibi generation pass (pick a room,
  generate placement evidence, generate a corroborating witness, optionally
  make one witness unreliable). More implementation work, more interesting
  gameplay.

Which do you recommend, and is there a middle ground I'm missing?
