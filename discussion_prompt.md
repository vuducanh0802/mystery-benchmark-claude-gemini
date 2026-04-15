# Discussion: Action Space Design Issues in a Mystery-Solving Benchmark

I'm building a benchmark for evaluating LLM agents on murder-mystery solving. The agent must figure out **who** committed the murder, **with what weapon**, and **where** — then present physical evidence to support the accusation. I'd like your thoughts on two design problems I've identified.

## Benchmark Overview

The agent operates in a grid of interconnected rooms under a fixed **action budget** (e.g., 20-50 actions). Each action costs 1 step. The agent must gather evidence, interview suspects, reason about alibis, and make a final accusation before the budget runs out.

**Scoring (composite, 0-1):**
- **Accusation correctness (40%):** Did the agent name the right suspect, weapon, and location?
- **Locard triangle evidence (40%):** Did the agent provide physical evidence for three edges: suspect-weapon link, weapon-victim link, and suspect-room link?
- **Alibi verification (20%):** Did the agent identify the suspect's alibi claim and present a logical contradiction using physical evidence?

## Available Actions

| Action | What it does |
|---|---|
| **MOVE** | Move to an adjacent room |
| **EXAMINE_LOCATION** | Get a description of the current room |
| **SEARCH_FOR_EVIDENCE** | Probabilistically discover hidden evidence in the current room. Can find **multiple items** in one action. Success probability per item: `clamp(1.0 - difficulty + 0.3, 0.1, 1.0)` |
| **EXAMINE_OBJECT** | Inspect a specific object — **guarantees** discovery if that object is linked to evidence |
| **TALK_TO** | Interview a character (ask a question, get a response). Culprits are evasive; innocents share alibis. Stateful multi-turn conversation |
| **ANALYZE** | Assess temporal freshness of evidence already in inventory (fresh / stale / ambiguous) |
| **TRAVEL_TIME** | Query shortest path between two rooms (for temporal reasoning) |
| **CHECK_ROUTE** | Query whether a passage between two rooms was open at a given time |
| **TAKE_OBJECT** | Pick up a portable object |
| **CHECK_INVENTORY** | Review collected evidence |
| **ACCUSE** | Make final accusation (ends the episode) |
| **WAIT** | Do nothing for one step |

**Evidence properties:** Each piece of evidence has a `discovery_difficulty` (0.0-1.0), a state (PRISTINE, HIDDEN, DESTROYED, etc.), and metadata about which Locard triangle edge it supports. Some evidence is linked to specific world objects (enabling guaranteed discovery via EXAMINE_OBJECT).

## Problem 1: SEARCH_FOR_EVIDENCE Feels Like Cheating

The probability formula is generous:
- difficulty 0.0 to 0.3 → 100% discovery rate
- difficulty 0.7 → 60% per attempt
- difficulty 1.0 → 10% per attempt, but retryable

Critically, **one SEARCH action can discover multiple evidence items** in the same room. Since the agent only needs 3 pieces of evidence (one per triangle edge), a few room visits with SEARCH can collect everything needed. This turns the mystery into a room-sweeping optimization problem rather than a deductive reasoning task.

Compare with EXAMINE_OBJECT, which requires the agent to **hypothesize which specific object** might be relevant and then inspect it — a much more detective-like reasoning process. But SEARCH makes EXAMINE_OBJECT largely unnecessary since it's strictly broader.

## Problem 2: The Oracle Agent Never Uses TALK_TO

I built an oracle agent as a calibration upper bound. It knows the ground truth and plays optimally. Its strategy:
1. Read ground truth to identify the 3 best evidence items (one per triangle edge)
2. Compute an optimal navigation route to their locations
3. Navigate and collect evidence via EXAMINE_OBJECT (if an object link exists) or SEARCH_FOR_EVIDENCE (otherwise)
4. Build the alibi contradiction **directly from the ground truth alibi data** — without ever interviewing anyone
5. ACCUSE with all evidence and alibi contradiction attached

The oracle gets full marks on alibi verification (20% of the composite score) **without using TALK_TO at all**, because:
- It reads the suspect's alibi claims from internal state
- The scoring function only checks whether the accusation includes a well-formed `alibi_contradiction` dict (with `claimed_location`, `claimed_time`, `contradiction_evidence`) — it doesn't verify that the agent **learned** this through gameplay

This means TALK_TO is effectively dead weight in the action space. Even non-oracle LLM agents can rationally skip interviews and still reach ~80% composite by focusing purely on physical evidence gathering + correct accusation.

## Questions for Discussion

1. **How should I rebalance SEARCH_FOR_EVIDENCE?** Some ideas I'm considering:
   - Limit to 1 item per search action
   - Remove the +0.3 bonus
   - Require specifying a search focus (e.g., "search the desk" vs. "search the room")
   - Make it return vague hints rather than full evidence (requiring EXAMINE_OBJECT as follow-up)

2. **How do I make TALK_TO essential?** Ideas:
   - Require that alibi information be learned through actual interviews (track provenance)
   - Gate some evidence behind interview-derived knowledge ("a witness mentions the suspect had a briefcase" → now you can search for it)
   - Add a scoring component that requires witness corroboration
   - Make some Locard triangle edges only provable through testimony

3. **Is the oracle agent the right calibration baseline?** It bypasses the core reasoning challenge (deduction under uncertainty) by reading the answer key. Should I instead:
   - Force the oracle to play within the same information constraints (no ground truth access, just perfect reasoning)?
   - Have two baselines: an omniscient oracle (current) + a "perfect detective" that uses all actions optimally under partial observability?

4. **Broader design question:** The benchmark currently rewards evidence *collection* more than evidence *reasoning*. The hard part of detective work — forming hypotheses, eliminating suspects, catching contradictions in testimony — is underweighted. How would you restructure the scoring or action space to make reasoning the bottleneck rather than room traversal?

Feel free to challenge my assumptions or suggest alternative framings I haven't considered.
