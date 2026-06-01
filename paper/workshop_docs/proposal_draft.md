# ICLR 2027 Workshop Proposal — DRAFT

> Working title options (pick one; #1 recommended):
> 1. **Beyond the Final Answer: Interactive, Process-Level, and Adversarial Evaluation of Agentic Reasoning**
> 2. Investigative Agents: Active Inference, Evidential Grounding, and Deception in Interactive Environments
> 3. Evaluating LLM Agents under Partial Observability, Deception, and Saturation
>
> Recommended short name: **PROBE** (Process- and Robustness-Oriented Benchmarking of Evaluation) — or keep it literal.
>
> **STATUS LEGEND:** ✅ ready · ✍️ drafted, needs your edits · ⛔ YOU must supply (cannot be invented): confirmed speakers, organizers, PC.
> **NOTE:** dates below mirror the ICLR 2026 cadence; replace with the official ICLR 2027 call when released.

---

## 1. Workshop Summary ✍️

Large language model agents are increasingly deployed in settings that are **interactive, partially observable, and adversarial** — they must decide *what information to gather next*, *reason from incomplete and sometimes deceptive evidence*, and *justify* a conclusion, not merely emit one. Yet the benchmarks we use to measure them are overwhelmingly **static, single-turn, and answer-only**: they score the final answer on a fixed question set, conflate *not looking* with *not reasoning*, give no credit for *grounding* a conclusion in evidence, and **saturate within months** of release.

This workshop convenes the community around a single question: **how should we evaluate agentic reasoning when the answer is not the whole story?** We focus on four under-served axes that static benchmarks cannot capture:

1. **Active information acquisition** — measuring *what an agent chooses to investigate* under a budget (value-of-information, optimal experimental design), separately from whether it then reasons correctly.
2. **Evidential grounding** — rewarding agents that *prove* a conclusion from a cited, verifiable evidence chain, not ones that guess it.
3. **Reasoning under deception** — robustness to lying interlocutors, red herrings, and adversarial misdirection (theory-of-mind, signaling games).
4. **Saturation-resistant evaluation** — procedural generation, difficulty curves, and *adversarial co-evaluation* (a second agent actively making the task harder) so a benchmark measures a *capability curve* tied to model strength, not a memorizable point.

These axes cut across representation learning for planning and decision-making, evaluation methodology, multi-agent systems, and interpretability — all core ICLR topics. The workshop's goal is to **crystallize a shared problem statement and a shared methodology** for process-level, robust agentic evaluation, and to build a community spanning benchmark designers, RL/planning researchers, and the evaluation/interpretability community.

*(Anchor framing for organizers: our own MysteryArena testbed — a procedurally generated, partially observable investigation game with an adversarial culprit and a decomposed perception/inference/grounding metric — is one instantiation of these axes and will seed a shared-task/demo track. It is positioned as an instrument, not the subject of the workshop.)*

## 2. Topics of Interest ✍️

Submissions are invited (but not limited to):

- **Interactive & process-level evaluation:** benchmarks that score *trajectories* (what was investigated, in what order) not just outcomes; metrics that separate perception/targeting from inference from grounding.
- **Active information-seeking & planning:** value-of-information, Bayesian/optimal experimental design, exploration under action/token budgets.
- **Evidential grounding & faithfulness:** citation/proof requirements, abductive reasoning (inference to the best explanation), distinguishing "knows" from "can justify."
- **Reasoning under deception & partial observability:** lying NPCs/agents, red-herring robustness, theory-of-mind, signaling games, POMDP formulations of reasoning.
- **Adversarial & multi-agent evaluation:** two-sided / self-play evaluation, attacker–defender or detective–culprit setups, role-specific and counterfactual (value-added) scoring.
- **Anti-saturation methodology:** procedural generation, difficulty parameterization, held-out instance distributions, capability-curve reporting, contamination resistance.
- **Theory & analysis:** formal links between agentic reasoning and POMDPs / active inference / experimental design; what these benchmarks do and do not measure.
- **Multimodal & embodied extensions** of the above (vision-language agents, 3D/embodied investigation).
- **Negative results & critiques:** where interactive benchmarks are themselves gameable, biased, or measure bookkeeping rather than reasoning.

## 3. Differences with Previous Workshops ✍️

Recent workshops — *LLM Agents* (ICLR 2024), *Reasoning and Planning for LLMs* (ICLR 2025), *System-2 Reasoning at Scale* (NeurIPS 2024), and various agent-benchmark venues — focus on **building better reasoning/planning methods** or cataloguing agent capabilities. They largely treat **evaluation as a solved means to an end**.

This workshop is distinct in three ways:
1. **Evaluation is the object, not the instrument.** We focus specifically on *how to measure* interactive reasoning, with emphasis on process-level and grounding-level signals beyond final-answer accuracy.
2. **Adversarial & partial-observability framing.** We foreground deception, hidden state, and *two-sided* (co-evolving) evaluation — under-represented in agent-capability workshops that assume a benign, fully-observable task.
3. **Saturation as a first-class concern.** We treat benchmark longevity (procedural generation, difficulty curves, contamination) as a methodological topic in its own right, responding to the field-wide problem that agent benchmarks are exhausted within months.

## 4. Format & Tentative Schedule ✅

Single-day, hybrid (in-person + livestreamed), morning/afternoon sessions with free time between for exchange. Mix of invited talks, contributed orals, two poster sessions, and a panel — to maximize discussion and community-building.

| Time | Item |
|---|---|
| 08:50–09:00 | Opening remarks |
| 09:00–09:35 | Invited Talk 1 (+10 Q&A) |
| 09:35–10:10 | Invited Talk 2 (+10 Q&A) |
| 10:10–11:00 | Poster Session 1 |
| 11:00–11:35 | Invited Talk 3 |
| 11:35–12:15 | Contributed orals (2 × 20 min) |
| 12:15–13:30 | Lunch / informal discussion |
| 13:30–14:05 | Invited Talk 4 |
| 14:05–14:40 | Invited Talk 5 |
| 14:40–15:25 | Contributed orals (2–3 × 15 min) |
| 15:25–16:25 | Poster Session 2 |
| 16:25–17:25 | **Panel:** "What should an agentic-reasoning benchmark measure in 2 years?" |
| 17:25–17:45 | Best-paper award + closing |

**Shared-task / demo (optional, distinctive):** a live MysteryArena leaderboard track where submissions enter detective and/or culprit agents — a concrete community-building hook few eval workshops offer.

## 5. Call for Papers, Tracks & Reviewing ✅

- **Main track:** 4–8 page papers (ICLR LaTeX style); references/appendix excluded from limit.
- **Tiny/Short track:** ≤4 pages — for newcomers, under-resourced and early-career researchers; counter-intuitive results, proofs-of-concept, re-analyses, negative results.
- Double-blind on OpenReview; ≥2 (target 3) reviews/paper; senior meta-reviewer decisions; oral selection from top papers (no previously-published work for orals).
- **Selection criteria:** novelty, technical quality, relevance to theme, clarity, potential impact.
- **Non-archival**, with optional arXiv posting and OpenReview hosting of accepted papers.
- LLM-usage policy: follow ICLR 2027 policy; declare any AI-as-author/reviewer participation.

## 6. Important Dates (template — update to ICLR 2027) ✍️

- Submissions open / CFP: ~Dec 2026
- Submission deadline: ~early Feb 2027
- Reviewing + discussion: Feb 2027
- Acceptance notification: by mandatory date (~1 Mar 2027)
- Camera-ready / posters public: by workshop day

## 7. Organizing Committee ⛔ (YOU supply — see guidance below)

> Target 5–9 organizers. Mix seniority (≥2 senior faculty/industry leads as anchors), gender, geography, institutions. Each bullet: name, affiliation, 3–4 lines on relevant track record (esp. prior workshop organizing).
>
> Plausible starting network (NUS/NTU + collaborators — TO CONFIRM, do not list without consent):
> - Senior anchors to invite: See Kiong Ng (NUS), Min-Yen Kan (NUS), Bryan Hooi (NUS), Anh Tuan Luu (NTU).
> - Early-career / PhD organizers: Thong T. Nguyen (NUS) + 2–3 collaborators across institutions/timezones.
> - Recruit ≥1–2 organizers from US/EU labs and ≥2 women for diversity balance.

## 8. Invited Speakers ⛔ (YOU supply — highest priority)

> Target 5–7, ideally ≥4 confirmed *before submission*. Aim for a spread across: agentic eval, POMDP/active inference & decision-making, multi-agent/game theory & deception, and evaluation/interpretability. Provide name, affiliation, gender, confirmation status, 4–5 line bio (mirror the template).
>
> Candidate profiles to approach (illustrative — NOT confirmations): leaders in LLM-agent evaluation, sequential decision-making / human-compatible AI, multi-agent RL & cooperative/adversarial agents, and benchmark/contamination methodology. Lock confirmations early — this is the dominant acceptance factor.

## 9. Program Committee & Reviewing Plan ⛔/✍️

> Recruit 25–40 reviewers (≥3 top-tier pubs each), named in a table with affiliation. Anticipate 50–80 submissions; ≤3 papers/reviewer; double-blind; meta-reviewers for decisions. Seed from organizers' networks + an open call via a Google form promoted on social media.

## 10. Diversity & Inclusion ✍️

Commit to diversity across the organizing committee (gender, seniority, geography, institutions), invited speakers, and participants. Tiny-paper track + travel/registration support (subject to sponsorship) for under-resourced and early-career researchers and low-income regions. Hybrid + captioned recordings for accessibility.

## 11. Outreach & Anticipated Size ✅

Dedicated website + OpenReview venue; promotion via X/LinkedIn/Reddit r/ML, mailing lists, and organizer/speaker networks. Anticipate ~100–150 in-person and 300–500 virtual, based on comparable agentic-reasoning workshops.

## 12. Sponsorship & COI ✅

Approach labs/industry for speaker honoraria, student/under-resourced support, and F&B. Strict COI: organizers/PC do not handle submissions from same-institution, recent co-authors, or close collaborators; declarations required before assignment.
