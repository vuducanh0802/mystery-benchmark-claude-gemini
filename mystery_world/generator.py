"""
Procedural mystery world generator.

Given a ``ComplexityConfig`` and an integer seed, this module generates a
complete, solvable mystery scenario. Every output is fully deterministic.

Generation pipeline:
    1. Sample locations and build adjacency graph
    2. Create characters (victim, culprit, suspects, innocents)
    3. Assign motive, weapon, murder location
    4. Generate ground-truth timeline (who was where at each step)
    5. Plant evidence (physical, testimonial, documentary, circumstantial)
    6. Inject red herrings
    7. Wire alibi chains
    8. Verify solvability (constraint check)
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from mystery_world import AssetPool, ComplexityConfig, DEFAULT_ASSET_POOL
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
from mystery_world.world import WorldState

def _uid(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}_{rng.integers(100000, 999999)}"

# ---------------------------------------------------------------------------
# Location graph
# ---------------------------------------------------------------------------

def _generate_locations(
    config: ComplexityConfig, pool: AssetPool, rng: np.random.Generator
) -> dict[str, Location]:
    n = config.num_locations
    names = list(rng.choice(pool.location_templates, size=min(n, len(pool.location_templates)), replace=False))
    locations: dict[str, Location] = {}
    loc_ids: list[str] = []
    for name in names:
        lid = _uid("loc", rng)
        tag = LocationTag.OUTDOOR if rng.random() < 0.3 else LocationTag.INDOOR
        locations[lid] = Location(
            id=lid, name=str(name), tag=tag,
            description=f"A {tag.name.lower()} space known as the {name}.",
            weather_exposed=(tag == LocationTag.OUTDOOR),
        )
        loc_ids.append(lid)
    
    # Build connected adjacency graph (ensure connected)
    # Strategy: create a random spanning tree, then add extra edges
    shuffled = list(loc_ids)
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        a, b = shuffled[i - 1], shuffled[i]
        locations[a].adjacent_ids.append(b)
        locations[b].adjacent_ids.append(a)
    # Extra edges for richness
    extra = max(1, n // 3)
    for _ in range(extra):
        a, b = rng.choice(loc_ids, size=2, replace=False)
        if b not in locations[a].adjacent_ids:
            locations[a].adjacent_ids.append(b)
            locations[b].adjacent_ids.append(a)
    return locations


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------

def _generate_characters(
    config: ComplexityConfig,
    pool: AssetPool,
    rng: np.random.Generator,
    location_ids: list[str],
) -> dict[str, Character]:
    total = 1 + config.num_suspects + config.num_innocents # victim + suspects + innocents
    first_names = list(rng.choice(pool.first_names, size=min(total, len(pool.first_names)), replace=False))
    last_names = list(rng.choice(pool.last_names, size=min(total, len(pool.last_names)), replace=False))
    traits = list(rng.choice(pool.personality_traits, size=min(total, len(pool.personality_traits)), replace=False))

    characters: dict[str, Character] = {}
    char_list: list[Character] = []
    for i in range(total):
        cid = _uid("chr", rng)
        char = Character(
            id=cid,
            first_name=str(first_names[i]),
            last_name=str(last_names[i]),
            personality=str(traits[i % len(traits)]),
            location_id=str(rng.choice(location_ids)),
        )
        characters[cid] = char
        char_list.append(char)
    
    # Assign roles
    victim = char_list[0]
    victim.roles.append(CharacterRole.VICTIM)
    victim.is_alive = False

    culprit_idx = 1   # first suspect is the culprit
    culprit = char_list[culprit_idx]
    culprit.roles.append(CharacterRole.SUSPECT)
    culprit.is_culprit = True

    for i in range(2, 1 + config.num_suspects):
        if i < len(char_list):
            char_list[i].roles.append(CharacterRole.SUSPECT)
    
    for i in range(1 + config.num_suspects, total):
        if i < len(char_list):
            char_list[i].roles.append(CharacterRole.INNOCENT)
            char_list[i].roles.append(CharacterRole.WITNESS)

    # Assign motives to suspects
    motives = list(rng.choice(pool.motive_templates, size=min(config.num_suspects, len(pool.motive_templates)), replace=False))
    suspect_chars = [c for c in char_list if CharacterRole.SUSPECT in c.roles]
    for i, sc in enumerate(suspect_chars):
        sc.motive = str(motives[i % len(motives)])
    
    # Generate relationships
    for c in char_list:
        n_rels = rng.integers(1, min(4, total))
        others = [o for o in char_list if o.id != c.id]
        if others:
            targets = rng.choice(others, size=min(int(n_rels), len(others)), replace=False)
            kinds = ["sibling", "colleague", "rival", "friend", "employer", "acquaintance"]
            for t in targets:
                c.relationships.append(Relationship(
                    target_id=t.id,
                    kind=str(rng.choice(kinds)),
                    sentiment=float(rng.uniform(-1, 1)),
                ))
    
    # Place characters in locations
    for cid, char in characters.items():
        loc = char.location_id
        if loc in {lid for lid in location_ids}:
            pass # will be added to location manifests in post-processing
    
    return characters

# ---------------------------------------------------------------------------
# Evidence & objects
# ---------------------------------------------------------------------------

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
) -> tuple[dict[str, Evidence], dict[str, WorldObject]]:

    evidence: dict[str, Evidence] = {}
    objects: dict[str, WorldObject] = {}
    non_culprit_suspects = [s for s in suspect_ids if s != culprit_id]

    def _sample_difficulty() -> float:
        return float(rng.uniform(config.evidence_difficulty_min, config.evidence_difficulty_max))

    # --- Murder weapon ---
    weapon_names = list(rng.choice(pool.weapon_templates, size=min(config.num_weapons, len(pool.weapon_templates)), replace=False))
    murder_weapon_name = str(weapon_names[0])
    mw_obj = WorldObject(
        id=murder_weapon_id,
        name=murder_weapon_name,
        description=f"A {murder_weapon_name}.",
        location_id=murder_location_id,
        is_weapon=True,
        is_murder_weapon=True,
    )
    objects[murder_weapon_id] = mw_obj

    # Evidence on the murder weapon — deceptive traces possible
    if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
        mw_linked_id = str(rng.choice(non_culprit_suspects))
    else:
        mw_linked_id = culprit_id

    murder_loc = locations.get(murder_location_id)
    mw_ev = Evidence(
        id=_uid("ev", rng),
        name=f"traces on the {murder_weapon_name}",
        evidence_type=EvidenceType.PHYSICAL,
        location_id=murder_location_id,
        linked_character_id=mw_linked_id,
        description=f"Forensic traces recovered from the {murder_weapon_name}.",
        discovery_difficulty=_sample_difficulty(),
        weather_sensitive=bool(murder_loc and murder_loc.weather_exposed),
    )
    evidence[mw_ev.id] = mw_ev
    mw_obj.evidence_id = mw_ev.id

    # --- Other weapons (non-murder) — same generic description style ---
    for i in range(1, config.num_weapons):
        if i < len(weapon_names):
            wid = _uid("obj", rng)
            w = WorldObject(
                id=wid,
                name=str(weapon_names[i]),
                description=f"A {weapon_names[i]}.",
                location_id=str(rng.choice(location_ids)),
                is_weapon=True,
                is_murder_weapon=False,
            )
            objects[wid] = w

    # --- Physical evidence ---
    ev_templates = [
        "fingerprint on doorknob",
        "bloodstain on carpet",
        "torn fabric from clothing",
        "footprint near the scene",
        "hair strand",
        "scratches on nearby furniture",
        "skin cells under victim's nails",
        "fiber transfer on victim's coat",
        "smudged handprint on wall",
        "cigarette butt near the scene",
        "partial shoe impression in mud",
        "broken button from a jacket",
        "perfume residue on victim's clothing",
        "small paint chip from a tool",
        "glass fragment with partial prints",
    ]
    n_culprit_ev = max(2, config.num_objects // 3)
    for i in range(n_culprit_ev):
        tmpl_name = ev_templates[i % len(ev_templates)]
        eid = _uid("ev", rng)
        if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
            linked_id = str(rng.choice(non_culprit_suspects))
        else:
            linked_id = culprit_id
        ev = Evidence(
            id=eid,
            name=tmpl_name,
            evidence_type=EvidenceType.PHYSICAL,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=linked_id,
            description=f"{tmpl_name.capitalize()} found during investigation.",
            discovery_difficulty=_sample_difficulty(),
            weather_sensitive=rng.random() < 0.3,
        )
        evidence[eid] = ev

    # --- Testimonial evidence (witness accounts) ---
    for sid in suspect_ids:
        if sid == culprit_id:
            continue
        eid = _uid("ev", rng)
        reliable = rng.random() >= config.testimony_unreliability
        ev = Evidence(
            id=eid,
            name=f"testimony about suspect movements",
            evidence_type=EvidenceType.TESTIMONIAL,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=sid,
            description=f"A witness account regarding a suspect's whereabouts.",
            discovery_difficulty=_sample_difficulty(),
            is_reliable=reliable,
        )
        evidence[eid] = ev

    # --- Documentary evidence ---
    doc_templates = [
        "a threatening letter",
        "a financial ledger entry",
        "a diary page with incriminating passage",
        "a forged alibi note",
        "a signed insurance policy",
        "a photograph with a revealing timestamp",
        "a bank withdrawal receipt",
        "a phone message transcript",
        "a torn contract",
        "a secret correspondence",
    ]
    for i in range(min(2, config.motive_layers)):
        eid = _uid("ev", rng)
        if non_culprit_suspects and rng.random() < config.evidence_ambiguity:
            doc_linked_id = str(rng.choice(non_culprit_suspects))
        else:
            doc_linked_id = culprit_id
        ev = Evidence(
            id=eid,
            name=str(rng.choice(doc_templates)),
            evidence_type=EvidenceType.DOCUMENTARY,
            location_id=str(rng.choice(location_ids)),
            linked_character_id=doc_linked_id,
            description=f"A document that may shed light on someone's motive.",
            discovery_difficulty=_sample_difficulty(),
        )
        evidence[eid] = ev

    # --- Red herrings ---
    for _ in range(config.num_red_herrings):
        eid = _uid("ev", rng)
        decoy_suspect = str(rng.choice(suspect_ids))
        ev = Evidence(
            id=eid,
            name=f"suspicious {rng.choice(['note', 'item', 'mark', 'stain'])}",
            evidence_type=EvidenceType(int(rng.integers(1, 5))),
            location_id=str(rng.choice(location_ids)),
            linked_character_id=decoy_suspect,
            is_red_herring=True,
            description="Something that looks incriminating but is ultimately misleading.",
            discovery_difficulty=float(rng.uniform(0.1, 0.4)),
        )
        evidence[eid] = ev

    # --- Generic objects ---
    obj_names = list(rng.choice(pool.object_templates, size=min(config.num_objects, len(pool.object_templates)), replace=False))
    for oname in obj_names:
        oid = _uid("obj", rng)
        obj = WorldObject(
            id=oid,
            name=str(oname),
            description=f"A {oname} lying here.",
            location_id=str(rng.choice(location_ids)),
            portable=rng.random() < 0.7,
        )
        objects[oid] = obj

    return evidence, objects

# ---------------------------------------------------------------------------
# Timeline generation
# ---------------------------------------------------------------------------

def _generate_timeline(
    config: ComplexityConfig,
    rng: np.random.Generator,
    characters: dict[str, Character],
    locations: dict[str, Location],
    culprit_id: str,
    victim_id: str,
    murder_location_id: str,
    murder_step: int,
) -> list[TimelineEntry]:
    """Generate ground-truth timeline of all character movements and events."""
    timeline: list[TimelineEntry] = []
    location_ids = list(locations.keys())
    
    for step in range(config.num_time_steps):
        for cid, char in characters.items():
            if not char.is_alive and step > murder_step:
                continue
            
            # At the murder step, place culprit and victim in murder location
            if step == murder_step:
                if cid == culprit_id:
                    timeline.append(TimelineEntry(
                        step=step, actor_id=cid, action="committed_murder",
                        target_id=victim_id, location_id=murder_location_id,
                        details=f"{char.full_name} attacked the victim.",
                        is_public=False,
                    ))
                    char.location_id = murder_location_id
                    continue
                elif cid == victim_id:
                    char.location_id = murder_location_id
                    continue
            
            # Regular NPC movement
            if rng.random() < 0.4:
                loc = locations.get(char.location_id)
                if loc and loc.adjacent_ids:
                    new_loc = str(rng.choice(loc.adjacent_ids))
                    timeline.append(TimelineEntry(
                        step=step, actor_id=cid, action="moved_to", # Thong: should we let the movement of NPC affect the mystery?
                        target_id=new_loc, location_id=new_loc,
                        details=f"{char.full_name} went to {locations[new_loc].name}.",
                        is_public=True,
                        witnesses=[
                            oid for oid in locations[new_loc].characters_here
                            if oid != cid and oid in characters
                        ],
                    ))
                    char.location_id = new_loc
            else:
                timeline.append(TimelineEntry(
                    step=step, actor_id=cid, action="stayed",
                    target_id=char.location_id, location_id=char.location_id,
                    details=f"{char.full_name} remained in the {locations.get(char.location_id, Location()).name}.",
                    is_public=True
                ))
    
    return timeline

# ---------------------------------------------------------------------------
# Alibi generation             
# ---------------------------------------------------------------------------
            
# --- Innocent suspect alibi templates (content always truthful) ---              

_INNOCENT_SOLO_ALIBIS = [      
    "I was in my room all evening. I know that is not much of an alibi.",       
    "I went for a walk alone after dinner. I have no one to confirm it.",         
    "I was in the garden by myself. Nobody saw me, but that is the truth.",       
    "I stayed in and went to bed early. I have no one to vouch for me.",          
    "I was alone in the billiard room. I did not feel like company.",             
    "I spent the evening alone in the library. I did not speak to anyone.",       
    "I was on the terrace by myself for most of the night.",                      
    "I retired to my room after dinner. I was alone the entire time.",            
    "I had a letter to write. I was at my desk until I went to sleep.",           
    "I was in the chapel, thinking. I saw no one and no one saw me.",             
]           
            
_INNOCENT_PARTIAL_ALIBIS = [   
    "I was with {corrs} until about nine, then I went to bed alone.",           
    "I spent part of the evening with {corrs}, but I left early.",                
    "{corrs} and I were talking for a while. I am not sure of the exact time I left.",                               
    "I was at the gathering with the others for most of it, then slipped away.",  
    "I was with {corrs} in the sitting room. I left before the others did.",      
    "Ask {corrs} — we were talking for a while. I headed off on my own after that.",                                 
    "I joined {corrs} for a drink, then went to bed. I cannot say exactly when.", 
    "{corrs} saw me in the hall earlier. After that I was alone in my room.",     
    "I was with {corrs} during dinner. The rest of the evening I spent alone.",   
    "{corrs} and I parted ways after supper. I do not know what they did after.", 
]           
            
_INNOCENT_CORROBORATED_ALIBIS = [                                                 
    "I was with {corrs} the entire evening. They can confirm every moment.",    
    "{corrs} and I were together all night. Neither of us left the room.",        
    "Ask {corrs}. We were in each other's company from dinner until well past midnight.",                            
    "I was playing cards with {corrs} all evening. We did not move from that table.",                                
    "{corrs} will tell you — I was with them the whole time without exception.",  
    "I was with {corrs} from the moment dinner ended. You are welcome to check.", 
    "{corrs} and I sat by the fire all evening. We both heard the clock strike midnight.",                           
    "I have nothing to hide. {corrs} were with me the entire night.",             
]           
            
# --- Culprit alibi templates (content always false) ---                          
        
_CULPRIT_SOLO_ALIBIS = [       
    "I was alone in my room the entire evening.",                               
    "I retired early — I had a headache. No one saw me after dinner.",            
    "I spent the evening alone in the library, reading.",                         
    "I walked the estate grounds by myself after supper.",                        
    "I was in the study going over correspondence. The door was closed.",         
    "I went to bed early. I heard nothing unusual.",                              
    "I took a long bath and then slept. I spoke to no one.",                      
    "I had a private matter to attend to in my room. I did not leave all night.", 
    "I was in the wine cellar cataloguing bottles. No one else was down there.",  
    "I sat alone on the terrace watching the sky. I spoke to no one.",            
    "I do not feel I need to account for my evening. I was simply alone.",        
    "I was writing letters in my room. I came out only for water.",               
]           
            
_CULPRIT_PARTIAL_ALIBIS_NO_WITNESS = [                                            
    "I was at dinner with the others until around nine. After that, I was alone.",                                 
    "I left the card game early and spent the rest of the evening alone in my room.",                                
    "I spent the early part of the evening with the group, then slipped away.",
    "I joined the gathering briefly, then retired. I spoke to no one after that.",
    "I was visible to others for the first part of the evening. Afterwards I was entirely alone.",                   
]           
            
_CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS = [                                          
    "I was talking with {corrs} in the hall for most of the evening, then went to my room alone.",                 
    "I was in the drawing room with {corrs} until the clock struck ten, then I retired alone.",                      
    "I joined {corrs} for drinks after supper, but I left early. After that I was by myself.",                       
    "I was with {corrs} briefly in the garden, then I went for a solitary walk.", 
    "I had a drink with {corrs} near the fire, then excused myself. I did not see anyone after that.",               
    "I sat with {corrs} at dinner. After the meal I slipped off alone — they can confirm I left early.",             
    "{corrs} saw me in the corridor earlier in the evening. After that, I was on my own.",                           
    "I left {corrs} in the parlour around half past nine. The rest of the evening I spent alone.",                   
    "Ask {corrs} — they will tell you I was there for dinner. What I did after is my own business.",                 
    "{corrs} and I shared a drink. I excused myself shortly after. I went nowhere near that part of the house.",     
]           
            
_CULPRIT_GAP_CORROBORATED_ALIBIS = [                                              
    "{corrs} and I were together most of the evening. I may have stepped out briefly, but it was nothing.",        
    "Ask {corrs} — we were together all night. I went to get a drink at some point, but I was not gone long.",       
    "{corrs} will confirm I was there. I slipped out for some air, five minutes at most.",                           
    "I was with {corrs} the entire time, more or less. I left the room once — I cannot imagine it matters.",         
    "{corrs} and I sat together all evening. I may have excused myself briefly at some point.",                      
    "I was in {corrs}'s company. There was a moment I stepped away, but {corrs} knew where I was going.",            
    "{corrs} can confirm my presence. I left for a short while but came straight back.",                             
    "{corrs} were with me. If I was away for a moment, it was not long enough to matter.",                           
]           
            
_CULPRIT_FALSE_CORROBORATED_ALIBIS = [                                            
    "I was with {corrs} all evening. Ask them.",                                
    "{corrs} and I were together the entire time.",                               
    "I spent the evening playing cards with {corrs}. They will confirm.",         
    "{corrs} can vouch for my whereabouts. We were together.",                    
    "I never left {corrs}'s sight all evening.",                                  
    "You can ask {corrs} — we were in the billiard room from eight until midnight.",                                 
    "{corrs} and I were at the far end of the estate. Neither of us went near that wing.",                           
    "I have witnesses: {corrs}. All of them saw me the entire time.",             
    "{corrs} were with me the whole evening. None of us moved from the sitting room.",                               
    "We — {corrs} and I — sat together well past the hour in question.",          
    "{corrs} will tell you I never moved. We were in plain sight of each other.", 
    "I was not alone. {corrs} were there. They saw everything.",                  
]           
            
            
def _format_names(names: list[str]) -> str:                                     
    if len(names) == 1:        
        return names[0]      
    elif len(names) == 2:
        return f"{names[0]} and {names[1]}"
    else:   
        return ", ".join(names[:-1]) + f" and {names[-1]}"
            
        
def _generate_alibis(
    config: ComplexityConfig,
    rng: np.random.Generator,
    characters: dict[str, Character],                                             
    timeline: list[TimelineEntry],
    culprit_id: str,           
    murder_step: int         
) -> None:  
    """     
    Assign alibis to all suspects. Format is randomised for every character
    regardless of guilt. Innocents' content is truthful; culprit's is false.      
    The agent cannot deduce guilt from format alone.                              
    """     
    suspects = [c for c in characters.values() if CharacterRole.SUSPECT in c.roles]                                  
    non_culprits = [s for s in suspects if s.id != culprit_id]                    
    all_char_list = list(characters.values())                                     
            
    def _get_corr_pool(exclude_id: str) -> list:                                  
        if config.allow_suspect_corroborators:                                  
            return [           
                c for c in all_char_list                                        
                if c.id != exclude_id and c.is_alive                              
                and CharacterRole.VICTIM not in c.roles                         
            ]                  
        else:
            return [           
                c for c in all_char_list                                        
                if c.id != exclude_id and c.id != culprit_id
                and c.is_alive and CharacterRole.SUSPECT not in c.roles
            ]                  

    def _pick_corroborators(pool: list, max_n: int) -> list:                      
        if not pool:         
            return []          
        n = min(int(rng.integers(1, max_n + 1)), len(pool))                     
        indices = list(range(len(pool)))                                          
        rng.shuffle(indices)
        return [pool[j] for j in indices[:n]]                                     
        
    # --- Non-culprit suspects: varied formats, always truthful ---               
    for i, suspect in enumerate(non_culprits):
        pool = _get_corr_pool(suspect.id)                                         
        suspect.alibi_corroboration_is_genuine = True                           
            
        if i < config.alibi_complexity:                                         
            # Fully corroborated alibi                                            
            suspect.has_alibi = True                                              
            corrs = _pick_corroborators(pool, config.max_corroborators)
            if corrs:          
                suspect.alibi_corroborator_id = corrs[0].id                     
                corr_names = _format_names([c.full_name for c in corrs])          
                template = _INNOCENT_CORROBORATED_ALIBIS[int(rng.integers(len(_INNOCENT_CORROBORATED_ALIBIS)))]    
                suspect.alibi_details = template.format(corrs=corr_names)         
            else:              
                template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]                      
                suspect.alibi_details = template                                  
        else:                
            # Varied: no alibi / solo / partial                                   
            alibi_roll = float(rng.random())
            if alibi_roll < 0.25:                                                 
                suspect.has_alibi = False                                       
            elif alibi_roll < 0.60:                                               
                suspect.has_alibi = True                                          
                template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]
                suspect.alibi_details = template                                  
            else:            
                suspect.has_alibi = True                                          
                if pool and rng.random() < 0.7:                                 
                    corrs = _pick_corroborators(pool, min(config.max_corroborators, 2))                              
                    corr_names = _format_names([c.full_name for c in corrs])
                    template = _INNOCENT_PARTIAL_ALIBIS[int(rng.integers(len(_INNOCENT_PARTIAL_ALIBIS)))]            
                    suspect.alibi_details = template.format(corrs=corr_names)   
                else:          
                    template = _INNOCENT_SOLO_ALIBIS[int(rng.integers(len(_INNOCENT_SOLO_ALIBIS)))]                
                    suspect.alibi_details = template                              
            
    # --- Culprit: same formats, always false ---
    culprit = characters.get(culprit_id)                                          
    if culprit:                
        alibi_roll = float(rng.random())
        w = config.culprit_alibi_weights                                          
        others = [             
            c for c in all_char_list
            if c.id != culprit_id and c.is_alive                                  
            and CharacterRole.VICTIM not in c.roles                             
        ]
            
        if alibi_roll < w[0]:
            # No alibi         
            culprit.has_alibi = False                                           

        elif alibi_roll < w[1]:
            # Solo — false, unverifiable
            culprit.has_alibi = True                                              
            culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]
            culprit.alibi_corroboration_is_genuine = True                         
            
        elif alibi_roll < w[2]:
            # Partial — false, admits a gap, no alibi_corroborator_id             
            culprit.has_alibi = True                                              
            if others and rng.random() < 0.6:
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))                                
                corr_names = _format_names([c.full_name for c in corrs])        
                template = _CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS[int(rng.integers(len(_CULPRIT_PARTIAL_ALIBIS_WITH_WITNESS)))]                                                                             
                culprit.alibi_details = template.format(corrs=corr_names)         
            else:              
                template = _CULPRIT_PARTIAL_ALIBIS_NO_WITNESS[int(rng.integers(len(_CULPRIT_PARTIAL_ALIBIS_NO_WITNESS)))]
                culprit.alibi_details = template                                  
            culprit.alibi_corroboration_is_genuine = True
            
        elif alibi_roll < w[3]:                                                 
            # Gap-corroborated — corroborators honest but missed a window         
            culprit.has_alibi = True                                            
            if others:         
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))
                culprit.alibi_corroborator_id = corrs[0].id                       
                culprit.alibi_corroboration_is_genuine = True                     
                culprit.alibi_has_gap = True
                corr_names = _format_names([c.full_name for c in corrs])          
                template = _CULPRIT_GAP_CORROBORATED_ALIBIS[int(rng.integers(len(_CULPRIT_GAP_CORROBORATED_ALIBIS)))]
                culprit.alibi_details = template.format(corrs=corr_names)
            else:              
                culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]         
                culprit.alibi_corroboration_is_genuine = True                     
        
        else:                  
            # Full false — lying corroborators                                  
            culprit.has_alibi = True
            if others:         
                corrs = _pick_corroborators(others, min(config.max_corroborators, 3))
                culprit.alibi_corroborator_id = corrs[0].id                       
                culprit.alibi_corroboration_is_genuine = False                    
                corr_names = _format_names([c.full_name for c in corrs])
                template = _CULPRIT_FALSE_CORROBORATED_ALIBIS[int(rng.integers(len(_CULPRIT_FALSE_CORROBORATED_ALIBIS)))]
                culprit.alibi_details = template.format(corrs=corr_names)         
            else:
                culprit.alibi_details = _CULPRIT_SOLO_ALIBIS[int(rng.integers(len(_CULPRIT_SOLO_ALIBIS)))]           
                culprit.alibi_corroboration_is_genuine = True                   


# ---------------------------------------------------------------------------
# Witness assignment
# ---------------------------------------------------------------------------
def _assign_witnesses(
    characters: dict[str, Character],
    timeline: list[TimelineEntry],
    rng: np.random.Generator,
) -> None:
    """Mark characters as witnesses for events they could have seen."""
    for entry in timeline:
        if not entry.is_public:
            continue
        
        for cid, char in characters.items():
            if cid == entry.actor_id:
                continue
            if char.location_id == entry.location_id and char.is_alive:
                event_key = entry.action
                if event_key not in char.witnessed_events:
                    char.witnessed_events.append(event_key)
                if CharacterRole.WITNESS not in char.roles:
                    char.roles.append(CharacterRole.WITNESS)


# ---------------------------------------------------------------------------
# Solvability verification
# ---------------------------------------------------------------------------
def verify_solvability(state: WorldState) -> dict[str, Any]:
    """Check that the mystery is solvable: sufficient non-red-herring evidence
    exists that points to the culprit, and alibis are consistent."""
    culprit_evidence = [
        e for e in state.evidence.values()
        if e.linked_character_id == state.culprit_id
        and not e.is_red_herring
        and e.is_usable()
    ]
    has_weapon_evidence = any(
        e for e in culprit_evidence if e.evidence_type == EvidenceType.PHYSICAL
    )
    has_motive_evidence = any(
        e for e in culprit_evidence if e.evidence_type == EvidenceType.DOCUMENTARY
    )
    has_testimonial = any(
        e for e in state.evidence.values()
        if e.evidence_type == EvidenceType.TESTIMONIAL and not e.is_red_herring
    )
    # The culprit should NOT have a corroborated alibi
    culprit = state.get_culprit()
    culprit_alibi_breakable = culprit and (
        not culprit.has_alibi or culprit.alibi_corroborator_id is None
    )

    solvable = (
        len(culprit_evidence) >= 2
        and has_weapon_evidence
        and culprit_alibi_breakable
    )

    return {
        "solvable": solvable,
        "culprit_evidence_count": len(culprit_evidence),
        "has_weapon_evidence": has_weapon_evidence,
        "has_motive_evidence": has_motive_evidence,
        "has_testimonial": has_testimonial,
        "culprit_alibi_breakable": culprit_alibi_breakable,
    }


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_mystery(
    config: ComplexityConfig,
    seed: int,
    asset_pool: AssetPool | None = None,
    max_retries: int = 10,
) -> WorldState:
    """
    Generate a complete, solvable mystery world.

    Parameters
    ----------
    config: ComplexityConfig
        Controls all dimensions of difficulty
    seed: int
        Random seed for full reproducibility
    asset_pool: AssetPool, optional
        Name/template pools (defaults to built-in originals).
    max_retries: int
        Number of attempts before raising (retries with incremented seed).
    
    Returns
    ----------
    WorldState
        Fully populated, verified-solvable world state.
    """
    pool = asset_pool or DEFAULT_ASSET_POOL

    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)

        # 1. Locations
        locations = _generate_locations(config, pool, rng)
        location_ids = list(locations.keys())
        
        # 2. Characters
        characters = _generate_characters(config, pool, rng, location_ids)
        char_list = list(characters.values())

        victim = next(c for c in char_list if CharacterRole.VICTIM in c.roles)
        culprit = next(c for c in char_list if c.is_culprit)
        suspect_ids = [c.id for c in char_list if CharacterRole.SUSPECT in c.roles]

        # 3. Murder details
        murder_location_id = str(rng.choice(location_ids))
        murder_weapon_id = _uid("wpn", rng)
        murder_step = int(rng.integers(1, max(2, config.num_time_steps // 2)))
        motive = culprit.motive or "unknown"

        # 4. Timeline
        timeline = _generate_timeline(
            config, rng, characters, locations,
            culprit.id, victim.id, murder_location_id, murder_step,
        )

        # 5. Evidence & objects
        evidence, objects = _generate_evidence_and_objects(
            config, pool, rng, location_ids,
            culprit.id, victim.id, suspect_ids,
            murder_weapon_id, murder_location_id, locations
        )

        # 6. Alibis
        _generate_alibis(config, rng, characters, timeline, culprit.id, murder_step)

        # 7. Witnesses
        _assign_witnesses(characters, timeline, rng)

        # 8. Populate location manifests
        for cid, char in characters.items():
            loc = locations.get(char.location_id)
            if loc and cid not in loc.characters_here:
                loc.characters_here.append(cid)
        
        for oid, obj in objects.items():
            loc = locations.get(obj.location_id)
            if loc and oid not in loc.objects_here:
                loc.objects_here.append(oid)

        # 9. Initial weather
        weather = str(rng.choice(pool.weather_types))

        # Assemble world state
        state = WorldState(
            seed=seed + attempt,
            config=config,
            locations=locations,
            characters=characters,
            objects=objects,
            evidence=evidence,
            current_step=0,
            weather=weather,
            ground_truth_timeline=timeline,
            culprit_id=culprit.id,
            victim_id=victim.id,
            murder_weapon_id=murder_weapon_id,
            murder_location_id=murder_location_id,
            murder_step=murder_step,
            motive=motive,
        )

        # 10. Verify solvability
        check = verify_solvability(state)
        if check["solvable"]:
            return state
    
    # If we exhausted retries, return last attempt with a warning
    return state   # type: ignore[possibly-undefined] 
