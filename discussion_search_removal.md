# Discussion: Removing SEARCH_FOR_EVIDENCE from a Mystery-Solving Benchmark

I'm building a benchmark for evaluating LLM agents on murder-mystery solving. I want to remove a specific action and need help thinking through the implications.

## The Setup

Agents operate in a world of interconnected rooms. Each room contains **objects** (a desk, a knife, a bookshelf) and may contain hidden **evidence** (fingerprints, bloodstains, documents). Evidence is the key to solving the case — the agent needs it for scoring.

When the agent enters a room, it sees a description including the names of visible objects: "You are in the study. You notice: a desk, a letter opener, scuffed floorboards." But it does NOT see evidence directly. Evidence must be discovered through actions.

## Current evidence discovery mechanics

There are two ways to find evidence:

1. **EXAMINE_OBJECT (targeted):** The agent picks a specific object to inspect. If that object has evidence linked to it, the evidence is guaranteed to be found. Example: `EXAMINE_OBJECT("letter opener")` → discovers fingerprints on the letter opener.

2. **SEARCH_FOR_EVIDENCE (room sweep):** The agent searches the entire room. Every piece of evidence in the room has an independent probability of being found. A single SEARCH can discover **multiple items**. The probability formula is generous: `clamp(1.0 - difficulty + 0.3, 0.1, 1.0)`.

## The problem with SEARCH

SEARCH_FOR_EVIDENCE is so powerful that it makes EXAMINE_OBJECT unnecessary. Why inspect a specific object when you can sweep the whole room and potentially find everything at once? This turns mystery-solving into a room-sweeping optimization problem: visit rooms → SEARCH everywhere → accumulate evidence → accuse.

I want to **remove SEARCH entirely** so that EXAMINE_OBJECT is the only way to find physical evidence. This would force hypothesis-driven investigation: the agent must look at the room description, decide which objects seem relevant, and inspect them individually. That's more detective-like.

## The complication: most evidence has no host object

Here's where it gets tricky. In my world generator, evidence and objects are separate entities:

- **Evidence** has: `id`, `name`, `description`, `location_id`, `discovery_difficulty`, etc.
- **WorldObject** has: `id`, `name`, `description`, `location_id`, and optionally `evidence_id` (pointer to an Evidence item)

The link is one-directional: an object can point to an evidence item via `evidence_id`. EXAMINE_OBJECT checks this link — if `obj.evidence_id` is set and the evidence isn't hidden/destroyed, it's discovered.

The problem: **only 1 out of ~15+ evidence items** in a typical generated mystery has a host object. Here's what the generator currently creates:

| Evidence | Has host object? |
|---|---|
| Grip marks on murder weapon (suspect-weapon link) | **Yes** — linked to the murder weapon object |
| Victim's blood on weapon (weapon-victim link) | No |
| Shoe scuffs in murder room (suspect-room link) | No |
| Material trace on victim's body | No — body exists as an object but `evidence_id` is not set |
| Drag marks (if body was moved) | No |
| Additional physical evidence (fingerprints, hair, fabric) | No |
| Testimonial evidence (witness accounts) | No |
| Documentary evidence (letters, ledgers) | No |
| Red herrings | No |

The generator also creates **generic objects** (a candlestick, a vase, etc.) scattered in rooms, but these have `evidence_id=None` — they're just flavor.

If I remove SEARCH today, almost all evidence becomes undiscoverable through EXAMINE_OBJECT.

## The fix I'm considering

Change the generator so every evidence item gets a host object. Examples:

- Victim's blood on weapon → already has the weapon object, but `evidence_id` is taken by grip marks. Need a second linkable aspect, or a separate "bloodstained blade" object.
- Shoe scuffs → create a "scuffed floorboards" object in the murder room
- Material trace on body → link to the existing body object (currently `evidence_id=None`)
- Hair strand → create a "rough door frame" object
- Documentary evidence → create a "desk drawer" or "bookshelf" object
- Testimonial evidence → this is different since it comes from interviews, not objects

## Questions for discussion

1. **Is removing SEARCH the right call?** Or should I instead nerf it (e.g., cap at 1 item per search, remove the +0.3 bonus, require a search focus area)? Removing is cleaner but requires a generator rewrite. Nerfing is less work but still leaves a room-sweep action in the game.

2. **How should the generator create host objects?** Some options:
   - (a) Every evidence item gets a dedicated object (shoe scuffs → "scuffed floorboards"). Simple but creates lots of objects that scream "examine me, I have evidence."
   - (b) Evidence is distributed across existing objects. The generic objects (candlestick, vase) already exist in rooms — assign evidence to them. Less obvious, but the agent can't tell which objects have evidence and which don't.
   - (c) Some mix: important evidence gets dedicated objects, minor evidence is hidden on generic objects.

3. **The "too many objects" problem:** If every evidence item gets a host object, rooms might have 8-10 objects. The agent must examine each one to check for evidence. Is this just replacing one brute-force approach (SEARCH the room) with another (EXAMINE every object)? How do we make the choice of which object to examine a reasoning challenge rather than exhaustive enumeration?

4. **What about testimonial evidence?** Witness accounts don't naturally live on physical objects. Options:
   - (a) Remove testimonial evidence from the physical discovery system entirely. Witnesses reveal testimony through TALK_TO only — they don't leave physical traces.
   - (b) Create "note" or "recording" objects that contain witness accounts.
   - (c) Keep SEARCH only for testimonial evidence, remove it for physical evidence.

5. **The one-to-one constraint:** Currently, each object can link to at most one evidence item (single `evidence_id` field). The murder weapon already links to grip marks. Should I change this to a list (`evidence_ids: list[str]`) so one object can yield multiple evidence items? The weapon could then reveal both grip marks AND victim's blood when examined.

6. **What about the existing body object?** The victim's body is already generated as an object (`body of {victim}`) but has no evidence linked to it. Linking the material trace evidence to the body seems natural. But should examining the body also reveal things like cause of death, time of death estimates, or wound patterns? This could make the body a very information-rich single examine action.

Feel free to challenge the premise — maybe removing SEARCH is the wrong approach entirely, and there's a better way to make evidence discovery require reasoning.
