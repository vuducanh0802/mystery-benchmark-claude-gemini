"""
Symbolic-augmented agent: LLM + knowledge graph + constraint solver.

This agent augments the LLM with:
  1. An explicit knowledge graph (NetworkX) tracking entities and relationships
  2. A constraint solver that eliminates impossible suspect-weapon-location combos
  3. Structured action planning based on information-gain heuristics

This addresses RQ2: does symbolic state tracking improve solve rates?
"""

from __future__ import annotations

import json
import re
from typing import Any

import networkx as nx

from agents.base_agent import BaseAgent
from agents.llm_agent import LLMClient
from mystery_world.entities import CharacterRole
from mystery_world.world import AgentAction, MysteryEnvironment


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Explicit symbolic state tracker using a directed labelled graph.

    Nodes represent entities (characters, locations, objects, evidence).
    Edges represent known relationships and constraints.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._fact_counter = 0

    def add_entity(self, entity_id: str, entity_type: str, **attrs: Any) -> None:
        self.graph.add_node(entity_id, entity_type=entity_type, **attrs)

    def add_relation(self, src: str, dst: str, relation: str, **attrs: Any) -> None:
        self.graph.add_edge(src, dst, relation=relation, **attrs)

    def add_fact(self, fact: str, source: str = "observation") -> None:
        fact_id = f"fact_{self._fact_counter}"
        self._fact_counter += 1
        self.graph.add_node(fact_id, entity_type="fact", text=fact, source=source)

    def get_entity_relations(self, entity_id: str) -> list[dict[str, Any]]:
        relations = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            relations.append({"target": target, **data})
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            relations.append({"source": source, **data})
        return relations

    def get_entities_by_type(self, entity_type: str) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("entity_type") == entity_type]

    def summarize(self) -> str:
        """Produce a text summary for the LLM context."""
        lines = ["=== KNOWLEDGE GRAPH STATE ==="]
        for ntype in ["suspect", "location", "weapon", "evidence", "fact"]:
            entities = self.get_entities_by_type(ntype)
            if entities:
                lines.append(f"\n[{ntype.upper()}S]")
                for eid in entities:
                    data = self.graph.nodes[eid]
                    name = data.get("name", eid)
                    extra = {k: v for k, v in data.items() if k not in ("entity_type", "name")}
                    rels = self.get_entity_relations(eid)
                    rel_strs = [f"  → {r.get('relation', '?')}: {r.get('target', r.get('source', '?'))}" for r in rels[:5]]
                    lines.append(f"  {name} {extra}")
                    lines.extend(rel_strs)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------

class ConstraintSolver:
    """
    Eliminates impossible (suspect, weapon, location) tuples using
    hard constraints derived from evidence and alibis.

    Constraints are simple boolean predicates:
      - "suspect X has verified alibi" → eliminate X
      - "weapon Y was confirmed unused" → eliminate Y
      - "location Z was occupied by witnesses at murder time" → eliminate Z
    """

    def __init__(self) -> None:
        self.eliminated_suspects: set[str] = set()
        self.eliminated_weapons: set[str] = set()
        self.eliminated_locations: set[str] = set()
        self.hard_constraints: list[str] = []

    def eliminate_suspect(self, name: str, reason: str) -> None:
        self.eliminated_suspects.add(name)
        self.hard_constraints.append(f"ELIMINATE suspect '{name}': {reason}")

    def eliminate_weapon(self, name: str, reason: str) -> None:
        self.eliminated_weapons.add(name)
        self.hard_constraints.append(f"ELIMINATE weapon '{name}': {reason}")

    def eliminate_location(self, name: str, reason: str) -> None:
        self.eliminated_locations.add(name)
        self.hard_constraints.append(f"ELIMINATE location '{name}': {reason}")

    def get_remaining_candidates(
        self,
        all_suspects: list[str],
        all_weapons: list[str],
        all_locations: list[str],
    ) -> dict[str, list[str]]:
        return {
            "suspects": [s for s in all_suspects if s not in self.eliminated_suspects],
            "weapons": [w for w in all_weapons if w not in self.eliminated_weapons],
            "locations": [l for l in all_locations if l not in self.eliminated_locations],
        }

    def count_remaining_combos(
        self,
        all_suspects: list[str],
        all_weapons: list[str],
        all_locations: list[str],
    ) -> int:
        remaining = self.get_remaining_candidates(all_suspects, all_weapons, all_locations)
        return (
            len(remaining["suspects"])
            * len(remaining["weapons"])
            * len(remaining["locations"])
        )

    def summarize(self) -> str:
        lines = ["=== CONSTRAINT STATE ==="]
        lines.append(f"Eliminated suspects: {sorted(self.eliminated_suspects) or 'none'}")
        lines.append(f"Eliminated weapons: {sorted(self.eliminated_weapons) or 'none'}")
        lines.append(f"Eliminated locations: {sorted(self.eliminated_locations) or 'none'}")
        for c in self.hard_constraints[-10:]:
            lines.append(f"  • {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Information-gain planner
# ---------------------------------------------------------------------------

def _plan_next_action(
    kg: KnowledgeGraph,
    solver: ConstraintSolver,
    env: MysteryEnvironment,
    all_suspects: list[str],
    all_weapons: list[str],
    all_locations: list[str],
    interviewed: set[str],
    examined_locations: set[str],
) -> tuple[str, str]:
    """
    Heuristic planner that picks the action maximising expected information gain.

    Returns (action_suggestion, rationale) as strings for the LLM prompt.
    """
    remaining = solver.get_remaining_candidates(all_suspects, all_weapons, all_locations)

    # Priority 1: interview remaining suspects we haven't talked to
    uninterviewed = [s for s in remaining["suspects"] if s not in interviewed]
    if uninterviewed:
        return (
            f"TALK_TO {uninterviewed[0]}",
            f"Interviewing {uninterviewed[0]} could provide alibi info to eliminate or confirm them.",
        )

    # Priority 2: examine objects in unexplored locations
    loc = env.get_current_location()
    current_loc_name = loc.name if loc else ""
    unexamined = [l for l in remaining["locations"] if l not in examined_locations]
    if unexamined:
        target = unexamined[0]
        if target != current_loc_name:
            return (
                f"MOVE {target}",
                f"Moving to {target} to examine its objects for evidence.",
            )
        return (
            "EXAMINE_LOCATION",
            f"Scanning {target} to find objects worth examining.",
        )

    # Priority 3: if only one candidate left, accuse
    if (len(remaining["suspects"]) == 1
        and len(remaining["weapons"]) == 1
        and len(remaining["locations"]) == 1):
        return (
            f"ACCUSE {remaining['suspects'][0]} {remaining['weapons'][0]} {remaining['locations'][0]}",
            "Only one possibility remains. Making accusation.",
        )

    return ("EXAMINE_LOCATION", "Continuing investigation.")


# ---------------------------------------------------------------------------
# Symbolic Agent
# ---------------------------------------------------------------------------

SYMBOLIC_SYSTEM_PROMPT = """\
You are a detective AI augmented with a knowledge graph and constraint solver.
You will be provided with:
1. The case briefing and observation history
2. A KNOWLEDGE GRAPH summary of all known entities and relationships
3. A CONSTRAINT STATE showing eliminated suspects/weapons/locations
4. An ACTION SUGGESTION from the planner (you may override it)

Output EXACTLY this JSON:
{
  "reasoning": "<step-by-step reasoning incorporating symbolic state>",
  "kg_updates": [
    {"type": "add_fact", "fact": "<new fact>"},
    {"type": "eliminate_suspect", "name": "<n>", "reason": "<why>"},
    {"type": "eliminate_weapon", "name": "<n>", "reason": "<why>"},
    {"type": "eliminate_location", "name": "<n>", "reason": "<why>"},
    {"type": "add_relation", "src": "<entity>", "dst": "<entity>", "relation": "<rel>"}
  ],
  "beliefs": {
    "top_suspect": "<name or null>",
    "suspect_confidence": <0.0-1.0>,
    "top_weapon": "<name or null>",
    "weapon_confidence": <0.0-1.0>,
    "top_location": "<name or null>",
    "location_confidence": <0.0-1.0>
  },
  "action": "<ACTION_NAME>",
  "action_args": {"<key>": "<value>", ...}
}

Valid actions: MOVE, EXAMINE_LOCATION, EXAMINE_OBJECT, TALK_TO, TAKE_OBJECT,
CHECK_INVENTORY, WAIT, ACCUSE. There is no SEARCH_FOR_EVIDENCE — evidence is
found by EXAMINE_OBJECT.

ACCUSE action_args schema (all keys required for full credit):
{
  "suspect_name": "<name>",
  "weapon_name": "<name>",
  "location_name": "<room>",
  "suspect_weapon_evidence": ["<evidence_id>", ...],
  "weapon_victim_evidence": ["<evidence_id>", ...],
  "suspect_room_evidence": ["<evidence_id>", ...],
  "alibi_contradiction": {
    "claimed_location": "<where the culprit claimed to be>",
    "claimed_time": "<time they claimed>",
    "contradiction_evidence": ["<evidence_id>", ...]
  },
  "eliminations": {
    "<innocent_name>": {"evidence_id": "<id>", "corroborator": "<witness_name>"}
  }
}

Use deductive reasoning: if a suspect has a verified alibi, ELIMINATE them.
Use abductive reasoning: which hypothesis best explains ALL evidence?
"""

class SymbolicAgent(BaseAgent):
    """
    LLM agent augmented with explicit symbolic reasoning tools.

    The knowledge graph and constraint solver maintain structured state
    that the LLM can query and update, reducing the reasoning burden on
    the language model and providing auditable inference chains.
    """

    def __init__(
        self,
        agent_id: str = "symbolic_agent",
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
    ):
        super().__init__(agent_id)
        self.llm = LLMClient(provider=provider, model=model)
        self.kg = KnowledgeGraph()
        self.solver = ConstraintSolver()
        self.briefing: str = ""
        self._env: MysteryEnvironment | None = None
        self._all_suspects: list[str] = []
        self._all_weapons: list[str] = []
        self._all_locations: list[str] = []
        self._interviewed: set[str] = set()
        self._examined_locations: set[str] = set()

    def initialize(self, env: MysteryEnvironment, briefing: str) -> None:
        self.briefing = briefing
        self._env = env
        state = env.state

        # Populate knowledge graph with initial entities
        for cid, char in state.characters.items():
            if CharacterRole.SUSPECT in char.roles:
                name = char.full_name
                self.kg.add_entity(name, "suspect", name=name, personality=char.personality)
                self._all_suspects.append(name)
                self.belief_state.suspect_probs[name] = 1.0 / max(1, len(self._all_suspects))

        for oid, obj in state.objects.items():
            if obj.is_weapon:
                self.kg.add_entity(obj.name, "weapon", name=obj.name)
                self._all_weapons.append(obj.name)
                self.belief_state.weapon_probs[obj.name] = 1.0

        for lid, loc in state.locations.items():
            self.kg.add_entity(loc.name, "location", name=loc.name)
            self._all_locations.append(loc.name)
            self.belief_state.location_probs[loc.name] = 1.0

        self.belief_state.normalize()

    def decide_action(self, observation: str) -> tuple[AgentAction, dict[str, str]]:
        # Get planner suggestion
        suggestion, rationale = _plan_next_action(
            self.kg, self.solver, self._env,
            self._all_suspects, self._all_weapons, self._all_locations,
            self._interviewed, self._examined_locations,
        )

        # Build augmented prompt
        kg_summary = self.kg.summarize()
        constraint_summary = self.solver.summarize()

        remaining = self.solver.get_remaining_candidates(
            self._all_suspects, self._all_weapons, self._all_locations,
        )
        combos = self.solver.count_remaining_combos(
            self._all_suspects, self._all_weapons, self._all_locations,
        )

        budget = self._env.budget_remaining if self._env else 0
        user_parts = [
            self.briefing, "",
            "=== OBSERVATION HISTORY (last 8) ===",
            *self.observation_history[-8:],
            "",
            f"=== CURRENT OBSERVATION (budget: {budget}) ===",
            observation,
            "",
            kg_summary,
            "",
            constraint_summary,
            "",
            f"Remaining solution space: {combos} combinations",
            f"Remaining suspects: {remaining['suspects']}",
            f"Remaining weapons: {remaining['weapons']}",
            f"Remaining locations: {remaining['locations']}",
            "",
            f"=== PLANNER SUGGESTION ===",
            f"Action: {suggestion}",
            f"Rationale: {rationale}",
            "",
            "Respond with the JSON object. You may override the planner suggestion.",
        ]
        user_msg = "\n".join(user_parts)

        response_text, tokens = self.llm.complete(SYMBOLIC_SYSTEM_PROMPT, user_msg)
        self.total_tokens_used += tokens

        parsed = self._parse_response(response_text)

        # Apply KG updates
        self._apply_kg_updates(parsed.get("kg_updates", []))

        # Update beliefs
        beliefs = parsed.get("beliefs", {})
        self._apply_belief_update(beliefs)

        # Extract action
        action_str = parsed.get("action", "EXAMINE_LOCATION").upper()
        action_args = parsed.get("action_args", {})

        # Track agent activities for planner
        if action_str == "TALK_TO" and "character_name" in action_args:
            self._interviewed.add(action_args["character_name"])
        if action_str == "EXAMINE_LOCATION":
            loc = self._env.get_current_location() if self._env else None
            if loc:
                self._examined_locations.add(loc.name)

        try:
            action = AgentAction[action_str]
        except KeyError:
            action = AgentAction.EXAMINE_LOCATION
            action_args = {}

        return action, action_args

    def update_beliefs(self, observation: str) -> None:
        # Handled inside decide_action
        pass

    def _parse_response(self, text: str) -> dict[str, Any]:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "EXAMINE_LOCATION", "action_args": {}, "beliefs": {}, "kg_updates": []}

    def _apply_kg_updates(self, updates: list[dict[str, Any]]) -> None:
        for u in updates:
            utype = u.get("type", "")
            if utype == "add_fact":
                self.kg.add_fact(u.get("fact", ""))
            elif utype == "eliminate_suspect":
                self.solver.eliminate_suspect(u.get("name", ""), u.get("reason", ""))
            elif utype == "eliminate_weapon":
                self.solver.eliminate_weapon(u.get("name", ""), u.get("reason", ""))
            elif utype == "eliminate_location":
                self.solver.eliminate_location(u.get("name", ""), u.get("reason", ""))
            elif utype == "add_relation":
                self.kg.add_relation(u.get("src", ""), u.get("dst", ""), u.get("relation", ""))

    def _apply_belief_update(self, beliefs: dict[str, Any]) -> None:
        bs = self.belief_state
        remaining = self.solver.get_remaining_candidates(
            self._all_suspects, self._all_weapons, self._all_locations,
        )

        # Reset eliminated to 0, redistribute among remaining
        for s in self._all_suspects:
            if s in self.solver.eliminated_suspects:
                bs.suspect_probs[s] = 0.0

        if beliefs.get("top_suspect") and beliefs["top_suspect"] in bs.suspect_probs:
            conf = beliefs.get("suspect_confidence", 0.5)
            for k in bs.suspect_probs:
                if k not in self.solver.eliminated_suspects:
                    bs.suspect_probs[k] = (1 - conf) / max(1, len(remaining["suspects"]) - 1)
            bs.suspect_probs[beliefs["top_suspect"]] = conf

        if beliefs.get("top_weapon") and beliefs["top_weapon"] in bs.weapon_probs:
            conf = beliefs.get("weapon_confidence", 0.5)
            for k in bs.weapon_probs:
                if k not in self.solver.eliminated_weapons:
                    bs.weapon_probs[k] = (1 - conf) / max(1, len(remaining["weapons"]) - 1)
            bs.weapon_probs[beliefs["top_weapon"]] = conf

        if beliefs.get("top_location") and beliefs["top_location"] in bs.location_probs:
            conf = beliefs.get("location_confidence", 0.5)
            for k in bs.location_probs:
                if k not in self.solver.eliminated_locations:
                    bs.location_probs[k] = (1 - conf) / max(1, len(remaining["locations"]) - 1)
            bs.location_probs[beliefs["top_location"]] = conf

        bs.normalize()
