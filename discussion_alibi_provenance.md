# Discussion: Alibi Provenance Tracking in a Mystery-Solving Benchmark

I'm building a benchmark for evaluating LLM agents on murder-mystery solving. I have a specific design problem with how alibi verification is scored, and I'd like your thoughts on a proposed solution.

## Context

Agents interact with a game environment through discrete actions (MOVE, EXAMINE_OBJECT, TALK_TO, ACCUSE, etc.) under a fixed action budget. The final accusation is scored on three components:

- **Accusation correctness (40%):** Named the right suspect, weapon, and location
- **Locard triangle evidence (40%):** Provided physical evidence linking suspect-weapon, weapon-victim, and suspect-room
- **Alibi verification (20%):** Identified the suspect's alibi claim and presented a logical contradiction

## The Problem

I have an **oracle agent** used as an upper-bound calibration baseline. It knows the full ground truth (who did it, where, with what). Its job is to demonstrate the minimum number of actions needed to achieve a perfect score.

The problem: the oracle constructs the alibi contradiction **directly from internal ground-truth data** without ever interviewing the suspect. In code, it reads `culprit.alibi_claims` (a structured list of location/time claims) from the world state and builds a contradiction dict from that.

The scoring function only checks whether the accusation includes a well-formed contradiction dict with `claimed_location`, `claimed_time`, and `contradiction_evidence`. It does **not** verify that the agent learned this information through a TALK_TO action.

This means the oracle gets full alibi score (20% of composite) while completely skipping the interview mechanic. Regular LLM agents don't have this problem in practice — they don't have access to ground truth, so they must interview suspects naturally. But the scoring system has no formal enforcement, which means:

1. The oracle's score is inflated — it appears to be a stronger baseline than it actually is
2. Any agent that somehow guesses or hardcodes alibi details would get unearned credit
3. The benchmark doesn't formally require the social reasoning (interviewing) that makes mysteries interesting

## Proposed Solution: Structured Alibi Claims in ActionResult

Instead of checking whether the agent talked to someone (too weak — they could ask about the weather) or doing NLP on conversation text (fragile and complex), I want to use **structured data provenance**.

When the TALK_TO action reveals an alibi, the environment returns structured data alongside the narrative text:

```python
# Current: only narrative text
ActionResult(True, 'John Smith: "I was in the library at 9:30 PM"')

# Proposed: narrative text + structured claim in details
ActionResult(
    True,
    'John Smith: "I was in the library at 9:30 PM"',
    details={"alibi_claim": {"location": "library", "time": "9:30 PM"}}
)
```

The environment tracks all alibi claims it has revealed through TALK_TO:

```python
self._revealed_alibi_claims: list[dict] = []
```

At scoring time, the alibi_contradiction in the accusation is validated against this list: does the `claimed_location` and `claimed_time` match something the environment actually returned through a TALK_TO action? If not, alibi score = 0.

## Questions for Discussion

1. **Is structured data provenance the right approach?** The alternative approaches I considered and rejected:
   - Binary check (did the agent talk to the culprit?) — too weak, doesn't verify what was learned
   - NLP on conversation text — fragile, complex, and the user specifically wants to avoid this
   - Are there other approaches I'm not seeing?

2. **Should the oracle be forced to comply?** The oracle knows ground truth. If we enforce provenance, the oracle must now spend an action on TALK_TO before it can score on alibi. This makes its action count slightly less optimal but makes it play the same game as other agents. Is this the right trade-off for a calibration baseline?

3. **Edge cases with the proposed approach:**
   - What if the culprit lies during the interview? The environment controls what alibi is revealed (lying is injected from ground-truth flags, not LLM-generated). So the structured claim would reflect the *stated* alibi (which may be false), and the scoring checks whether the agent correctly contradicts *that specific stated claim*. Does this make sense?
   - What if the agent interviews the culprit multiple times and gets slightly different claims? Should all revealed claims be valid for provenance, or only the most recent?
   - What if a witness reveals the culprit's alibi indirectly ("I saw John heading to the library around 9:30")? Should witness-sourced alibi information also count for provenance?

4. **Impact on benchmark design:** This change makes TALK_TO mechanically necessary for 20% of the score. Is that good (forces social reasoning) or bad (makes the benchmark more formulaic — "always interview the suspect" becomes a fixed recipe)?

Feel free to challenge the premise or suggest alternative framings.
