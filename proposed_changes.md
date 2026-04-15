# Oracle max-score + corroborator exclusion

Two files:
- `agents/oracle_agent.py`
- `mystery_world/world.py`

---

## `agents/oracle_agent.py`

### Change 1 — OraclePlan: replace single targets with lists

**Lines 57–68** — replace:

```python
@dataclass
class OraclePlan:
    sw_target: _EvidenceTarget | None = None
    wv_target: _EvidenceTarget | None = None
    sr_target: _EvidenceTarget | None = None
    visit_order: list[str] = field(default_factory=list)
    alibi_contradiction: dict[str, Any] | None = None
    talk_targets: list[str] = field(default_factory=list)           # full names
    elimination_targets: list[_EliminationTarget] = field(default_factory=list)
    culprit_name: str = ""
    weapon_name: str = ""
    location_name: str = ""
```

with:

```python
@dataclass
class OraclePlan:
    sw_targets: list[_EvidenceTarget] = field(default_factory=list)
    wv_targets: list[_EvidenceTarget] = field(default_factory=list)
    sr_targets: list[_EvidenceTarget] = field(default_factory=list)
    visit_order: list[str] = field(default_factory=list)
    alibi_contradiction: dict[str, Any] | None = None
    talk_targets: list[str] = field(default_factory=list)           # full names
    elimination_targets: list[_EliminationTarget] = field(default_factory=list)
    culprit_name: str = ""
    weapon_name: str = ""
    location_name: str = ""
```

### Change 2 — `_make_accuse_action`: cite all discovered evidence per edge

**Lines 165–189** — replace:

```python
    def _make_accuse_action(self) -> tuple[AgentAction, dict[str, Any]]:
        plan = self._plan
        assert plan is not None
        kwargs: dict[str, Any] = {
            "suspect_name": plan.culprit_name,
            "weapon_name": plan.weapon_name,
            "location_name": plan.location_name,
        }
        if plan.sw_target:
            kwargs["suspect_weapon_evidence"] = [plan.sw_target.evidence_id]
        if plan.wv_target:
            kwargs["weapon_victim_evidence"] = [plan.wv_target.evidence_id]
        if plan.sr_target:
            kwargs["suspect_room_evidence"] = [plan.sr_target.evidence_id]
        if plan.alibi_contradiction:
            kwargs["alibi_contradiction"] = plan.alibi_contradiction
        if plan.elimination_targets:
            kwargs["eliminations"] = {
                et.suspect_name: {
                    "evidence_id": et.evidence_id,
                    "corroborator": et.corroborator_name,
                }
                for et in plan.elimination_targets
            }
        return AgentAction.ACCUSE, kwargs
```

with:

```python
    def _make_accuse_action(self) -> tuple[AgentAction, dict[str, Any]]:
        plan = self._plan
        assert plan is not None
        discovered = self._env._discovered_evidence if self._env else set()

        def _cited(targets: list[_EvidenceTarget]) -> list[str]:
            return [t.evidence_id for t in targets if t.evidence_id in discovered]

        kwargs: dict[str, Any] = {
            "suspect_name": plan.culprit_name,
            "weapon_name": plan.weapon_name,
            "location_name": plan.location_name,
        }
        sw_ids = _cited(plan.sw_targets)
        wv_ids = _cited(plan.wv_targets)
        sr_ids = _cited(plan.sr_targets)
        if sw_ids:
            kwargs["suspect_weapon_evidence"] = sw_ids
        if wv_ids:
            kwargs["weapon_victim_evidence"] = wv_ids
        if sr_ids:
            kwargs["suspect_room_evidence"] = sr_ids
        if plan.alibi_contradiction:
            # Update contradiction_evidence to all discovered SR ids
            contra = dict(plan.alibi_contradiction)
            contra["contradiction_evidence"] = sr_ids
            kwargs["alibi_contradiction"] = contra
        if plan.elimination_targets:
            kwargs["eliminations"] = {
                et.suspect_name: {
                    "evidence_id": et.evidence_id,
                    "corroborator": et.corroborator_name,
                }
                for et in plan.elimination_targets
            }
        return AgentAction.ACCUSE, kwargs
```

### Change 3 — `_build_plan`: collect all evidence per edge

**Lines 209–212** — replace:

```python
        # Triangle targets
        plan.sw_target = self._best_evidence(EdgeType.SUSPECT_WEAPON)
        plan.wv_target = self._best_evidence(EdgeType.WEAPON_VICTIM)
        plan.sr_target = self._best_evidence(EdgeType.SUSPECT_ROOM)
```

with:

```python
        # Triangle targets — collect ALL valid evidence per edge
        plan.sw_targets = self._all_evidence_for_edge(EdgeType.SUSPECT_WEAPON)
        plan.wv_targets = self._all_evidence_for_edge(EdgeType.WEAPON_VICTIM)
        plan.sr_targets = self._all_evidence_for_edge(EdgeType.SUSPECT_ROOM)
```

**Lines 214–219** — update the alibi call to use the first SR target:

```python
        # Alibi (Change 6: evidence_id form)
        if culprit and culprit.alibi_claims:
            sr_eid = plan.sr_target.evidence_id if plan.sr_target else None
            plan.alibi_contradiction = self._build_alibi_contradiction(
                culprit.alibi_claims, sr_eid
            )
```

with:

```python
        # Alibi — pass the first SR evidence id for the initial contradiction dict;
        # _make_accuse_action will overwrite contradiction_evidence with all discovered SR ids.
        if culprit and culprit.alibi_claims:
            sr_eid = plan.sr_targets[0].evidence_id if plan.sr_targets else None
            plan.alibi_contradiction = self._build_alibi_contradiction(
                culprit.alibi_claims, sr_eid
            )
```

**Lines 249–275** — update the visit-order block to use the new lists:

```python
        # Visit order (TSP over evidence locations + talk targets)
        ev_targets = [t for t in (plan.sw_target, plan.wv_target, plan.sr_target) if t is not None]
        for et in plan.elimination_targets:
            se_ev = state.evidence.get(et.evidence_id)
            if se_ev:
                obj_name = next(
                    (o.name for o in state.objects.values() if o.evidence_id == et.evidence_id),
                    None,
                )
                ev_targets.append(_EvidenceTarget(
                    evidence_id=et.evidence_id,
                    location_id=se_ev.location_id,
                    object_name=obj_name,
                ))

        talk_locs = [
            state.characters[cid].location_id
            for cid in talk_ids if cid in state.characters
        ]

        unique_locs = list(dict.fromkeys(
            [t.location_id for t in ev_targets] + talk_locs
        ))
        plan.visit_order = self._greedy_visit_order(env.agent_location_id, unique_locs)

        # Store all evidence targets for decide_action
        self._all_ev_targets = ev_targets
        return plan
```

with:

```python
        # Visit order (TSP over evidence locations + talk targets)
        ev_targets: list[_EvidenceTarget] = []
        ev_targets += plan.sw_targets
        ev_targets += plan.wv_targets
        ev_targets += plan.sr_targets
        for et in plan.elimination_targets:
            se_ev = state.evidence.get(et.evidence_id)
            if se_ev:
                obj_name = next(
                    (o.name for o in state.objects.values() if o.evidence_id == et.evidence_id),
                    None,
                )
                ev_targets.append(_EvidenceTarget(
                    evidence_id=et.evidence_id,
                    location_id=se_ev.location_id,
                    object_name=obj_name,
                ))

        talk_locs = [
            state.characters[cid].location_id
            for cid in talk_ids if cid in state.characters
        ]

        unique_locs = list(dict.fromkeys(
            [t.location_id for t in ev_targets] + talk_locs
        ))
        plan.visit_order = self._greedy_visit_order(env.agent_location_id, unique_locs)

        # Store all evidence targets for decide_action
        self._all_ev_targets = ev_targets
        return plan
```

### Change 4 — rename `_best_evidence` → `_all_evidence_for_edge`, return a list

**Lines 279–324** — replace:

```python
    def _best_evidence(self, edge: EdgeType) -> _EvidenceTarget | None:
        """Return the easiest-to-discover, fresh, non-red-herring evidence for edge."""
        env = self._env
        assert env is not None
        state = env.state
        murder_ts = state.murder_timestamp
        threshold = state.freshness_threshold

        candidates = []
        for ev in state.evidence.values():
            if ev.is_red_herring:
                continue
            if ev.state in (EvidenceState.DESTROYED,):
                continue
            if ev.discovery_difficulty >= 1.0:
                continue
            if ev.relevance is None or ev.relevance.edge_type != edge:
                continue
            # Freshness check
            if abs(ev.relevance.contact_timestamp - murder_ts) >= threshold:
                continue
            # Must point to the right entities
            from mystery_world.world import _relevance_matches_truth
            if not _relevance_matches_truth(ev.relevance, edge, state):
                continue
            candidates.append(ev)

        if not candidates:
            return None

        # Sort: prefer non-HIDDEN first, then by difficulty ascending
        candidates.sort(key=lambda e: (e.state == EvidenceState.HIDDEN, e.discovery_difficulty))
        ev = candidates[0]

        # Find the linked WorldObject — match by evidence_id only;
        # the object may be in a different room than ev.location_id
        # (e.g. tampered weapon moved before discovery).
        obj_name: str | None = None
        obj_location_id: str = ev.location_id
        for obj in state.objects.values():
            if obj.evidence_id == ev.id:
                obj_name = obj.name
                obj_location_id = obj.location_id
                break

        return _EvidenceTarget(
            evidence_id=ev.id,
            location_id=obj_location_id,
            object_name=obj_name,
        )
```

with:

```python
    def _all_evidence_for_edge(self, edge: EdgeType) -> list[_EvidenceTarget]:
        """Return ALL fresh, non-red-herring evidence targets for this edge."""
        env = self._env
        assert env is not None
        state = env.state
        murder_ts = state.murder_timestamp
        threshold = state.freshness_threshold
        from mystery_world.world import _relevance_matches_truth

        candidates = []
        for ev in state.evidence.values():
            if ev.is_red_herring:
                continue
            if ev.state in (EvidenceState.DESTROYED,):
                continue
            if ev.discovery_difficulty >= 1.0:
                continue
            if ev.relevance is None or ev.relevance.edge_type != edge:
                continue
            if abs(ev.relevance.contact_timestamp - murder_ts) >= threshold:
                continue
            if not _relevance_matches_truth(ev.relevance, edge, state):
                continue
            candidates.append(ev)

        targets = []
        for ev in candidates:
            obj_name: str | None = None
            obj_location_id: str = ev.location_id
            for obj in state.objects.values():
                if obj.evidence_id == ev.id:
                    obj_name = obj.name
                    obj_location_id = obj.location_id
                    break
            targets.append(_EvidenceTarget(
                evidence_id=ev.id,
                location_id=obj_location_id,
                object_name=obj_name,
            ))
        return targets
```

Also update the `run()` plan_summary block which references the old single-target fields.

**Lines 437–448** — replace:

```python
            "plan_summary": {
                "culprit": self._plan.culprit_name if self._plan else "",
                "weapon": self._plan.weapon_name if self._plan else "",
                "location": self._plan.location_name if self._plan else "",
                "sw_evidence": self._plan.sw_target.evidence_id if self._plan and self._plan.sw_target else None,
                "wv_evidence": self._plan.wv_target.evidence_id if self._plan and self._plan.wv_target else None,
                "sr_evidence": self._plan.sr_target.evidence_id if self._plan and self._plan.sr_target else None,
                "alibi_type": (
                    "A" if self._plan and self._plan.alibi_contradiction
                    and len(env.state.get_culprit().alibi_claims) == 1
                    else ("B" if self._plan and self._plan.alibi_contradiction else "none")
                ) if self._plan else "none",
            },
```

with:

```python
            "plan_summary": {
                "culprit": self._plan.culprit_name if self._plan else "",
                "weapon": self._plan.weapon_name if self._plan else "",
                "location": self._plan.location_name if self._plan else "",
                "sw_evidence": [t.evidence_id for t in self._plan.sw_targets] if self._plan else [],
                "wv_evidence": [t.evidence_id for t in self._plan.wv_targets] if self._plan else [],
                "sr_evidence": [t.evidence_id for t in self._plan.sr_targets] if self._plan else [],
                "alibi_type": (
                    "A" if self._plan and self._plan.alibi_contradiction
                    and len(env.state.get_culprit().alibi_claims) == 1
                    else ("B" if self._plan and self._plan.alibi_contradiction else "none")
                ) if self._plan else "none",
            },
```

---

## `mystery_world/world.py`

### Change 5 — exclude corroborators from `total_innocents`

Witnesses (corroborators) confirm another suspect's alibi — their presence
elsewhere is implicit. Agents should not be required to separately eliminate
them. Only non-corroborator innocent suspects count toward the denominator.

**Lines 905–911** — replace:

```python
    # --- Score 4: Elimination (SUSPECT_ELSEWHERE + corroborator interview) ---
    if eliminations:
        total_innocents = sum(
            1 for c in state.characters.values()
            if c.is_alive and not c.is_culprit
            and CharacterRole.SUSPECT in c.roles
        )
```

with:

```python
    # --- Score 4: Elimination (SUSPECT_ELSEWHERE + corroborator interview) ---
    if eliminations:
        corroborator_ids = {
            ev.corroborator_id
            for ev in state.evidence.values()
            if ev.relevance is not None
            and ev.relevance.edge_type == EdgeType.SUSPECT_ELSEWHERE
            and ev.corroborator_id
        }
        total_innocents = sum(
            1 for c in state.characters.values()
            if c.is_alive and not c.is_culprit
            and CharacterRole.SUSPECT in c.roles
            and c.id not in corroborator_ids
        )
```
