#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escape the Castle — Rooms 1–2 (Prototype Stage 2)
Author: (you)
Platform: Python 3.9+ (tested on macOS), works great in VS Code terminal
Model backend: Ollama with llama3.1:8b (or compatible)

Setup:
1) Install Ollama: https://ollama.com
2) Pull the model: `ollama pull llama3.1:8b`
3) Python deps: `pip install requests`
4) Run: `python escape_castle.py`
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

# -----------------------------
# Configuration
# -----------------------------

# (valfritt) sätt via env-variabler: OLLAMA_HOST, MODEL_NAME
import os
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_API = f"{OLLAMA_HOST}/api/chat"
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")

TEMPERATURE = 0.2
TOP_P = 0.9

DEV_LOG_PATH = "game_dev.log"
NOTHING_LINE = "Nothing special happened."

# -----------------------------
# Utilities
# -----------------------------

def dev_log(line: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(DEV_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def strip_code_fences(s: str) -> str:
    s = s.strip()
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    m = fence.match(s)
    if m:
        return m.group(1).strip()
    return s


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        text2 = strip_code_fences(text)
        try:
            return json.loads(text2)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                snippet = m.group(0)
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class GameState:
    current_room: str = "cell_01"
    hp: int = 100

    # World items: single-slot inventory design; torch has "lit" state
    # location: "player" | "cell_01" | "coal_01" | "hall_01"
    items: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
    # Start: träfacklan ligger i källaren (o-tänd)
    "torch": {"location": "coal_01", "lit": False},
    # Nyckelknippan börjar i hallen (på riddaren)
    "keys":  {"location": "hall_01"},
    # Armborst med pilar: står uppe i tornet i Slottsgården
    "crossbow": {"location": "courtyard_tower_top"}
})


    inventory: List[str] = field(default_factory=list)   # derived from items; kept for UI text

    # Room 1 (cell) flags
    flags_cell: Dict[str, bool] = field(default_factory=lambda: {
        "found_loose_stone": False,
        "stone_moved": False,
        "entered_hole": False,
    })
    # Room 2 (coal) flags (kept for LLM context; derived from items when possible)
    flags_coal: Dict[str, bool] = field(default_factory=lambda: {
        "has_torch_stick": False,   # True if holding an unlit torch
        "torch_lit": False,         # True if there is torchlight with you or in your room
        "coal_intro_shown": False,  # printed once when first entering
    })

    # Hall (great hall) flags
    flags_hall: Dict[str, bool] = field(default_factory=lambda: {
        "knight_knocked_out": False,       # Etapp 2 använder denna
        "courtyard_door_unlocked": False,  # dörren låst tills den låses upp med keys
        "hall_intro_shown": False          # skrivs ut en gång när man kommer in i hallen
    })

    # Courtyard flags/state
    flags_courtyard: Dict[str, Any] = field(default_factory=lambda: {
        "at_tower_top": False,          # du står uppe i tornet (på plattformen)
        "gate_lowered": False,          # porten är nedfälld (och fungerar som bro)
        "guards_present": False,        # vakter är aktiva på gräsmattan
        "guards_remaining": 0,          # antal vakter kvar (0–3)
        "courtyard_intro_shown": False, # skrivs ut första gången man kommer ut
        "in_moat": False                # du ligger i vallgraven
    })



@dataclass
class LLMResult:
    narration: str
    noise_level: int
    hp_delta: int
    events: List[str]
    flags_set: List[str]
    progression: str
    safety_reason: str


# -----------------------------
# Scene Cards
# -----------------------------

ROOM_CELL_SCENE_CARD = {
    "room_id": "cell_01",
    "title": "Prison Cell",
    "room_description": (
        "You awaken in a cramped 5×5 meter dungeon cell. The walls, ceiling, and floor are built from large "
        "rounded cobblestones about 20–30 centimeters across. There is a straw bed on the cobblestone floor. A single torch flickers on one wall. "
        "An iron door with a barred window faces a corridor lit by torches. A guard dozes on a bench outside; "
        "he wears full armor and looks irritable—there is no realistic chance to trick him or defeat him. "
        "High up, a tiny 10×10 cm window shows a foggy night sky and the outline of the castle—a lethal 20-meter drop. "
        "On the floor lies a thin straw ‘bed’. Hidden under the straw is a loose cobblestone; lifting it reveals a crawlable hole "
        "leading down to a coal cellar."
    ),
    "facts_and_constraints": [
        "Strictly physical and realistic; no magic, no conjuration of tools, no impossible stunts.",
        "You cannot remove the wall torch or use it as an item.",
        "The high window cannot be used to escape (fatal drop).",
        "The guard is outside, drowsy but will react to disruptive noise; he wears full armor and cannot be overpowered or tricked.",
        "Quiet interactions with the guard are flavor only (no progress).",
        "There are no hidden items beyond what is stated.",
        "English only.",
    ],
    "stateful_flags": [
        "found_loose_stone", "stone_moved", "entered_hole",
        "has_torch_stick", "torch_lit"
    ],
    "noise_scale_reference": {
        "0": "Silent/minimal: careful inspection, tiptoeing, whispering, gently moving straw.",
        "1": "Quiet: normal walking, soft speaking, light rummaging.",
        "2": "Noticeable/disruptive: humming, singing, dancing with steps, knocking, moving objects loudly. Calling for help aloud or addressing the guard aloud counts as 2 unless explicitly whispered.",
        "3": "Loud/alarming: shouting, banging metal, breaking objects, screaming."
    },
    "triggers_policy": [
        "Noise reflects physical loudness, not persuasion success.",
        "If noise_level >= 2, the guard reacts immediately this turn: he unlocks the door, strikes you for -20 HP, then returns. Must be narrated.",
        "Any careful inspection/searching/looking under/moving/rummaging/removing of the straw MUST add 'straw_rummaged' and set 'found_loose_stone' if not already.",
        "If the player lifts/moves/pries/drags the loose stone (once discovered), set 'stone_moved' and reveal a crawlable hole.",
        "If the player crawls into/goes through the hole (once available), set 'entered_hole' and add 'enter_coal_cellar'.",
        "If the player holds a wooden torch (has_torch_stick) and lights it on the wall torch, add 'light_torch' and set 'torch_lit' (the wall torch itself cannot be removed).",
        "If the player drops/throws/places their torch here, add 'drop_torch'. If they pick it up, add 'pickup_torch'.",
        "If an action is impossible, playful grounded refusal and end with 'Nothing special happened.'",
        "Never invent items/exits beyond the scene; never escape via the high window.",
        "If the player interacts with the hole using non-movement verbs, you MUST refuse and end with 'Nothing special happened.' Do NOT add traversal events.",
        "If the player tries to light a torch without holding the wooden torch stick, give a brief flavorful warning about heat/singeing and end with 'Nothing special happened.'",

    ],
    "style_and_tone": [
        "Second person, immersive, concise, vivid.",
        "Maximum ~3 sentences per turn.",
        "Announce each key outcome exactly once; do not repeat the same fact.",
        "If nothing meaningful happens, end with 'Nothing special happened.'"
    ],
}

ROOM_COAL_SCENE_CARD = {
    "room_id": "coal_01",
    "title": "Coal Cellar",
    "room_description": (
        "A pitch-black coal cellar: tight stone walls, heaps of coal, dust in the air, and a cold stone floor. "
        "You emerge from a crawlspace in the wall. In the dark you bump your foot against something lying on the ground—"
        "you can’t tell what it is until you pick it up. With light, you can make out heaps of coal and, at the far end, "
        "a short stone staircase leading down to a door."
    ),
    "facts_and_constraints": [
        "English only.",
        "No guards here; noise does not summon anyone.",
        "Physical realism applies—no magic or conjured tools.",
        "The crawl opening behind you leads back up to the prison cell.",
        "The far cellar door is CLOSED but NOT locked; with light it can be opened freely.",
        "It is very dark by default. Without a lit torch, moving around is dangerous."
    ],
    "stateful_flags": [
        "has_torch_stick", "torch_lit"
    ],
    "noise_scale_reference": {
        "0": "Careful whisper/motion.",
        "1": "Quiet movement.",
        "2": "Noticeable noise.",
        "3": "Very loud."
    },
    "triggers_policy": [
        "If the player attempts to move deeper while in darkness (no 'torch_lit'), set 'dark_stumble' and hp_delta -10. Warn them and say they remain where they are.",
        "If 'torch_lit' is true, never set 'dark_stumble'; describe visible surroundings instead.",
        "If the player picks up the unknown object at their feet, reveal a wooden torch stick: set 'has_torch_stick' and add 'pickup_stick' or 'pickup_torch'.",
        "If the player intends to return through the crawl opening, add 'return_to_cell'.",
        "If the player drops/throws/places their torch here, add 'drop_torch'. If they pick it up, add 'pickup_torch'.",
        "If 'torch_lit' is true, you may describe coal heaps, cramped floor, and a short staircase leading to a door.",
        "If the player opens OR 'unlocks' the far door (with light), add 'open_hall_door' and succeed — the door is not locked.",
        "If an action is impossible, playful grounded refusal and end with 'Nothing special happened.'",
        "Never invent items or exits not stated.",
        "If the player tries to ignite/set the coal heaps on fire, refuse with a realistic reason (damp dust, no airflow) and end with 'Nothing special happened.'",

    ],
    "style_and_tone": [
        "Second person, immersive, concise, vivid.",
        "Maximum ~3 sentences per turn.",
        "Announce each key outcome exactly once."
    ],
}

ROOM_HALL_SCENE_CARD = {
    "room_id": "hall_01",
    "title": "Great Hall",
    "room_description": (
        "A grand, well-lit hall about 30×10 meters. Five stone pillars line each side; "
        "chandeliers burn high above, and paintings hang along the walls. "
        "You enter from a door at one short end; at the opposite short end stands a heavy door to the courtyard. "
        "A knight stands near the entry-side short wall with his back turned, a keyring hanging from his belt."
    ),
    "facts_and_constraints": [
        "English only.",
        "The hall is brightly lit—no darkness mechanics here.",
        "Single-slot inventory still applies.",
        "The keyring is here (initially on the knight). It can be picked up only if your hands are free.",
        "The courtyard door is locked until you unlock it with the keys.",
        "You can return to the coal cellar through the door you came from."
    ],
    "stateful_flags": [
        "knight_knocked_out", "courtyard_door_unlocked"
    ],
    "noise_scale_reference": {
        "0": "Silent/minimal: careful sneaking, soft footsteps, hugging pillars.",
        "1": "Quiet: normal walking, brief cloth/armor rustle.",
        "2": "Noticeable/disruptive: stumbling, clanking, dropping items, raised voice.",
        "3": "Loud/alarming: shouting, banging, sprinting heavily."
    },
    "triggers_policy": [
        # Navigation & keys:
        "If the player goes back to the cellar, add 'return_to_coal'.",
        "If the player tries to pick up the keys, add 'pickup_keys'. Require empty hands (single-slot).",
        "If the player drops the keys here, add 'drop_keys'.",
        "To unlock the courtyard door, add 'unlock_courtyard_door' and require keys in inventory. If locked and no keys, refuse and end with 'Nothing special happened.'",
        # Stage 2: Knight logic — the engine will enforce, but the model should mirror it:
        "If noise_level >= 2 and the knight is not already knocked out, add 'knight_notice' and 'combat_knock_guard' this turn; the player always knocks him unconscious (knight_knocked_out=True).",
        "After the knight is unconscious (knight_knocked_out=True), you MUST NOT add 'knight_notice' or new combat in later turns. If the player makes noise or addresses him, respond that he’s out cold.",

        "Stealth paths are allowed: quietly take the keys (requires empty hands) and slip behind pillars to the door.",
        "Avoid stealth phrasing if knight_knocked_out=True (he’s unconscious).",
        "Do NOT invent custom events like 'exit_hall'; use provided events only."
    ],
    "style_and_tone": [
        "Second person, immersive, concise, vivid.",
        "Maximum ~3 sentences per turn.",
        "Announce each key outcome exactly once."
    ],
}

ROOM_COURTYARD_SCENE_CARD = {
    "room_id": "courtyard_01",
    "title": "Castle Courtyard",
    "room_description": (
        "A wide grassy courtyard enclosed by high stone walls, lit by torches and starlight. "
        "About 50 meters ahead stands a massive wooden gate, currently closed. "
        "To the right of the gate, a squat tower meets the wall; a ladder leads up to a small platform. "
        "On the platform: a lever that lowers the gate into a bridge across the moat, a crossbow leaning against the wall, and a heap of bolts. "
        "Outside the walls lies a deep moat, roughly 30 meters down."
    ),
    "facts_and_constraints": [
        "English only.",
        "No darkness mechanics here; everything is readable by starlight and torches along the walls.",
        "Single-slot inventory still applies (torch OR keys OR crossbow).",
        "Guards cannot climb the ladder due to their armor; they remain on the grass.",
        "Pulling the lever always works and produces a booming noise; this noise summons three guards who charge across the lawn.",
        "The wooden gate, when lowered, forms a bridge across the moat.",
        "You cannot go back into the Great Hall; that door locks behind you."
    ],
    "stateful_flags": [
        "at_tower_top", "gate_lowered", "guards_present"
    ],
    "noise_scale_reference": {
        "0": "Quiet: cautious steps on grass, steady breathing.",
        "1": "Noticeable: jogging, speaking normally.",
        "2": "Loud: shouting across the yard, dropping metal.",
        "3": "Very loud: the gate slamming down (lever)."
    },
    "triggers_policy": [
        "Climbing the ladder sets 'at_tower_top' true; climbing down unsets it.",
        "The gate cannot be opened from the ground. Only pulling the lever at the tower top lowers it: 'pull_lever' MUST succeed there, sets 'gate_lowered' true and summons three guards ('guards_arrive').",
        "Attempts like 'open the gate' from the grass should be refused with a short flavored line pointing to the lever, then end with 'Nothing special happened.'",
        "Picking up the crossbow requires free hands and being on the tower platform.",
        "'shoot_guard' requires holding the crossbow; remove 1–3 guards depending on the player’s intent. If no guards remain, say so.",
        "Crossing the gate-bridge requires the gate to be lowered; doing so ends the game in victory.",

    ],
    "style_and_tone": [
        "Second person, immersive, concise, vivid.",
        "Maximum ~3 sentences per turn.",
        "Announce each key outcome exactly once."
    ],
}


SCENES: Dict[str, Dict[str, Any]] = {
    "cell_01": ROOM_CELL_SCENE_CARD,
    "coal_01": ROOM_COAL_SCENE_CARD,
    "hall_01": ROOM_HALL_SCENE_CARD,
    "courtyard_01": ROOM_COURTYARD_SCENE_CARD,

}

# Static room graph (engine controls legal travel)
ROOM_GRAPH: Dict[str, List[str]] = {
    "cell_01": ["coal_01"],
    "coal_01": ["cell_01", "hall_01"],
    "hall_01": ["coal_01", "courtyard_01"],  # nu leder hallen till slottsgården
    "courtyard_01": [],                       # ingen backtracking in i hallen
}



# -----------------------------
# Prompt templates
# -----------------------------

SYSTEM_PROMPT = """You are the Game Master for a grounded, text-based escape adventure.
Follow the CURRENT ROOM's scene card and the rules below. Respond in VALID JSON only.

Global rules:
- English only. Interpret player intent semantically; no verb lists.
- Evaluate noise_level (0–3) using the current room's scale; resets next turn.

- Guard punishment applies ONLY in cell_01.
  - If noise_level >= 2 THIS TURN in cell_01, then:
    * You MUST include 'guard_punishes' in events.
    * You MUST set hp_delta to -20 (exactly once).
    * You MUST NOT narrate the guard as oblivious/unresponsive/not noticing the player this turn.
    * Keep narration consistent with 'guard_punishes'. You may briefly mention the strike; the engine may append a fixed line.
  - If noise_level < 2 in cell_01, the guard does NOT punish; do not narrate a strike.

- Traversal events are mandatory and ONLY movement verbs cause traversal:
  * Down from the cell => 'enter_coal_cellar'.
  * Up from the cellar => 'return_to_cell'.
  * Interactions with the **hole** that are **not** movement (e.g., throw/poop/spit/look/talk) MUST NOT traverse. Give a playful grounded refusal and end with "Nothing special happened."

- Straw in cell_01:
  * Any interaction that explicitly mentions **straw/hay/bed** MUST add 'straw_rummaged' and set 'found_loose_stone' if not already.
  * Do NOT set 'stone_moved' from straw-only actions. Lifting/prying the **stone** is separate.
  * Do NOT invent straw interaction if the player didn’t mention straw/hay/bed.

- Multi-action: if player both clears straw AND lifts stone, include 'straw_rummaged' and 'stone_lifted', and set both flags the same turn.
- Never invent items, tools, magic, or exits beyond the active scene card.

- **Flavorful denials** for impossible/silly actions:
  * Give a short, vivid, **playful** line tailored to the user’s verbs/nouns, then end with **"Nothing special happened."**
  * Prefer color over scolding; keep it one sentence of color + the standard line.
  * Example themes to re-use when apt: “face near wall torch (too hot)”; “baseball swing with coal (sad thud)”; “imagining cupcake recipe with coal/water (bad idea)”.

- Humor: For every response (success or denial), include one short, witty quip tailored to the action; keep it PG-13 and brief (one clause).

- **Self-harm attempts:** never describe the player injuring or killing themselves. Give a grounded, in-universe refusal (survival instinct, hesitation), then end with "Nothing special happened." Do NOT apply damage.

- Narration: immersive 2nd person, ~3 sentences, announce each key outcome once. If impossible/no change, end with "Nothing special happened."
- Use only these semantic events: 'straw_rummaged','stone_lifted','enter_coal_cellar','return_to_cell','open_hall_door','return_to_coal','dark_stumble','pickup_stick','pickup_torch','drop_torch','light_torch','extinguish_torch','pickup_keys','drop_keys','unlock_courtyard_door','guard_punishes','knight_notice','combat_knock_guard','climb_ladder_up','climb_ladder_down','pull_lever','guards_arrive','pickup_crossbow','drop_crossbow','shoot_guard','cross_gate_bridge','jump_into_moat','swim_across'.
- Event parity: only narrate outcomes that correspond to 'events' and/or 'flags_set'. Do NOT invent new event names.

Coal cellar (coal_01) — darkness & safety:
- Default state is **pitch-black**. Do NOT describe it as dim/visible unless 'torch_lit' is true here.
- Visibility parity: don’t narrate using a torch unless the player carries it or a lit torch is present **in this room**.
- Moving deeper while dark: include 'dark_stumble' and hp_delta -10; warn they remain in place. Returning UP never stumbles.
- If 'torch_lit' is true, NEVER set 'dark_stumble'; instead describe heaps/staircase/far door.
- Identifying the unknown floor object happens ONLY on pickup: reveal a wooden torch stick ('pickup_stick'/'pickup_torch').
- Opening the far door ('open_hall_door') is ONLY possible if torchlight is present in coal_01 **this turn**; otherwise refuse and end with "Nothing special happened."
- Do NOT describe lighting a torch **outside** cell_01; refuse and end with "Nothing special happened."
- Trying to ignite coal heaps is **not feasible here** (no draft/prep); give a realistic, flavorful refusal + "Nothing special happened."

Great hall (hall_01):
- Well-lit; ignore darkness.
- Stealth viable at noise 0–1 (quietly steal keys; requires empty hands).
- If noise >= 2 this turn AND the knight is not already out:
  * Include 'knight_notice' and 'combat_knock_guard' in events.
  * The player always knocks the knight unconscious (set knight_knocked_out=True).
  * HP penalty depends on torch-in-hand snapshot (engine may adjust).
- **After the knight is unconscious**, you MUST NOT add 'knight_notice' or new combat; instead flavor-line that he’s out cold if relevant.

- Keys pickup requires free hands. Keys can be dropped. The courtyard door is locked until 'unlock_courtyard_door' with keys.

Courtyard (courtyard_01):
- Well lit; ignore darkness.
- The **gate cannot be opened from the ground.** Only the **tower lever** lowers it:
  * 'pull_lever' MUST succeed **only when at the tower top**, sets 'gate_lowered' true and summons three guards ('guards_arrive').
  * Attempts like “open the gate” from the ground: give a flavorful denial pointing at the lever up top + "Nothing special happened."
- Guards remain on the grass; they cannot climb the ladder. After they arrive, they are real targets—do NOT claim “no guards here.”
- The crossbow is on the tower platform; pickup requires free hands. 'shoot_guard' requires holding the crossbow.
- You may let the player shoot 1–3 guards in a single turn; if none remain, say so.
- 'cross_gate_bridge' only after the gate is lowered; doing so wins (escape).
- 'jump_into_moat' only from tower top; costs 40 HP (engine may apply). If then 'swim_across', player wins.
- If the player ends a turn on the ground while guards are present, expect heavy damage (engine may apply).

Inventory protocol:
- Single slot (torch OR keys OR crossbow). If hands are full and they try to pick something else, refuse and end with "Nothing special happened."
- Dropping/placing the torch removes it from inventory ('drop_torch'); if lit, it still provides light only in that room.

Return a single JSON object with keys:
narration, noise_level, hp_delta, events, flags_set, progression, safety_reason
"""


USER_INSTRUCTION_TEMPLATE = """CURRENT ROOM:
- Id: {room_id}
- Title: {room_title}

SCENE CARD:
{scene_card_json}

CURRENT STATE:
- HP: {hp}
- Flags (cell): {flags_cell}
- Flags (coal): {flags_coal}
- Flags (hall): {flags_hall}
- Flags (courtyard): {flags_courtyard}

- Items: {items}
- Inventory (one slot): {inventory}


REMINDERS:
- Traversal (MANDATORY): down from the cell => 'enter_coal_cellar'; up from the cellar => 'return_to_cell'.
- Straw in the Prison Cell reveals the loose stone if not already.
- To move the stone, the player must explicitly try to lift/pry/drag it ('stone_lifted').
- Inventory: SINGLE slot; to pick another item, drop the current one ('drop_torch').
- Coal Cellar: default is pitch-black. Without lit torch, moving causes 'dark_stumble' and you remain in place. With light, describe staircase and door.
- Lighting the torch works ONLY in the Prison Cell using the wall torch while holding the stick.
- IMPORTANT (cell_01 only): If noise_level >= 2 THIS TURN, include 'guard_punishes', set hp_delta to -20, and DO NOT narrate the guard as oblivious/unresponsive.
- IMPORTANT (coal_01): Do NOT narrate using a torch unless it's in inventory or a lit torch is present here. Do NOT describe a "dimly lit" cellar unless 'torch_lit' is true.
- Do NOT add 'straw_rummaged' or narrate interacting with straw unless the player explicitly wrote straw/hay/bed.
- Do NOT narrate lighting a torch outside cell_01; if attempted, refuse and end with "Nothing special happened."
- Great Hall: well-lit. You may 'return_to_coal'. Keys can be picked up only with free hands (single-slot). 
  The courtyard door is locked until 'unlock_courtyard_door' with keys.
- Non-movement interactions with the hole (throw/poop/etc.) MUST NOT cause traversal; give a playful denial + "Nothing special happened."
- Flavorful denials are preferred over plain refusals for impossible/silly actions; keep them short and end with "Nothing special happened."
- Self-harm attempts must be refused diegetically; do not inflict damage.
- In the courtyard, the gate can’t be opened from the ground; only the tower lever lowers it.
- After the knight is unconscious, do not create new combat; mention he’s out cold if addressed.


PLAYER ACTION:
{player_action}

IMPORTANT:
- Output VALID JSON ONLY. No code fences, no commentary.
- Keep narration concise (<= ~3 sentences).
"""


# -----------------------------
# LLM client (Ollama)
# -----------------------------

class OllamaChat:
    def __init__(self, model: str = MODEL_NAME, temperature: float = TEMPERATURE, top_p: float = TOP_P):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.history: List[Dict[str, str]] = []

    def reset(self):
        self.history = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        tail_history = self.history[-6:]
        messages.extend(tail_history)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p
            },
            "messages": messages
        }

        resp = requests.post(OLLAMA_CHAT_API, json=payload, timeout=120)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            raise ValueError("Model response was not valid JSON (HTTP OK but non-JSON body).")

        content = data.get("message", {}).get("content", "")
        dev_log(f"RAW_MODEL_OUTPUT: {content}")

        parsed = parse_llm_json(content)
        if parsed is None:
            dev_log("JSON parse failed; attempting one strict re-ask.")
            strict_user = user_prompt + "\n\nYour last output was invalid JSON. Respond again with VALID JSON ONLY."
            payload["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": strict_user},
            ]
            resp2 = requests.post(OLLAMA_CHAT_API, json=payload, timeout=120)
            resp2.raise_for_status()
            try:
                data2 = resp2.json()
            except ValueError:
                raise ValueError("Retry response was not valid JSON.")
            content2 = data2.get("message", {}).get("content", "")
            dev_log(f"RAW_MODEL_OUTPUT_RETRY: {content2}")
            parsed = parse_llm_json(content2)
            if parsed is None:
                raise ValueError("Model failed to return valid JSON twice.")

        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": json.dumps(parsed)})

        return parsed


# -----------------------------
# Engine: validation & progression
# -----------------------------
ALLOWED_HALL_FLAGS = {"knight_knocked_out", "courtyard_door_unlocked"}
ALLOWED_CELL_FLAGS = {"found_loose_stone", "stone_moved", "entered_hole", "has_torch_stick", "torch_lit"}
ALLOWED_COAL_FLAGS = {"has_torch_stick", "torch_lit"}
ALLOWED_COURTYARD_FLAGS = {"at_tower_top", "gate_lowered", "guards_present"}

ALLOWED_EVENTS = {
    "straw_rummaged", "stone_lifted",
    "enter_coal_cellar", "return_to_cell",
    "dark_stumble",
    "pickup_stick", "pickup_torch", "drop_torch", "light_torch", "extinguish_torch",
    "open_hall_door",  # från källaren till hallen

    # HALL / KEYS
    "return_to_coal",
    "pickup_keys", "drop_keys",
    "unlock_courtyard_door",

    # Cell & Hall
    "guard_punishes",
    "knight_notice", "combat_knock_guard",

    # COURTYARD
    "climb_ladder_up", "climb_ladder_down",
    "pull_lever", "guards_arrive",
    "pickup_crossbow", "drop_crossbow",
    "shoot_guard",
    "cross_gate_bridge",
    "jump_into_moat", "swim_across",
}


# Robust hit-verb detection to avoid duplicate guard line
HIT_VERBS_RE = re.compile(
    r"\b(strike|strikes|struck|striking|hit|hits|hitting|"
    r"smack|smacks|smacked|punch|punches|punched|"
    r"kick|kicks|kicked|club|clubs|clubbed)\b",
    re.IGNORECASE
)

# ---------- Intent inference (from player's input) ----------

_DROP_RE = re.compile(r"\b(drop|throw|toss|discard|leave|place|put\s+down|set\s+(?:it\s+)?down)\b", re.IGNORECASE)
_PICK_RE = re.compile(r"\b(pick\s*up|take|grab|collect|retrieve|get|remove|detach|unhook|unmount|pull\s+off|rip\s+off)\b", re.IGNORECASE)
_LIGHT_TORCH_RE = re.compile(
    r"(?:(?:light|ignite|set)\s+(?:the|your)?\s*(?:torch|stick)\b|"
    r"\b(?:torch|stick)\b.*\b(light|ignite|alight)|"
    r"set\s+(?:the\s+)?(?:torch|stick|it)\s+on\s+fire)",
    re.IGNORECASE
)

_STRAW_RE = re.compile(r"\b(straw|hay|straw\s*bed|bed)\b", re.IGNORECASE)
_CELL_WORD = re.compile(r"\bcell\b(?!ar)", re.IGNORECASE)  # 'cell' men inte 'cellar'
# Extra intent / narration detectors
_EXTINGUISH_RE = re.compile(r"\b(extinguish|snuff|put\s+out|douse|blow\s+out|quench)\b", re.IGNORECASE)

# --- Stone intent (lyfta/bända/ta bort) med fler synonymer ---
_STONE_ACT_VERBS = (
    r"(?:lift|raise|pry|prise|lever|heave|move|drag|shift|wiggle|jiggle|shake|work|"
    r"remove|take(?:\s+out)?|pull(?:\s+out)?|yank|rip|loosen|dislodge|free|unseat|unwedge|"
    r"bend|smash|break)"
)

_STONE_ACT_RE1 = re.compile(
    rf"\b{_STONE_ACT_VERBS}\b.*\b(loose\s+)?(stone|cobblestone|rock)\b",
    re.IGNORECASE
)
_STONE_ACT_RE2 = re.compile(
    rf"\b(loose\s+)?(stone|cobblestone|rock)\b.*\b{_STONE_ACT_VERBS}\b",
    re.IGNORECASE
)



# Courtyard intents
_CROSS_GATE_RE = re.compile(r"\b(cross|run|dash|go|head)\b.*\b(gate|bridge|drawbridge)\b", re.IGNORECASE)

_SHOOT_RE = re.compile(r"\b(shoot|fire|loose|let\s+fly|squeeze(?:\s+the)?\s+trigger|aim\s+and\s+fire)\b", re.IGNORECASE)
_ALL_THREE_RE = re.compile(r"\b(all\s+three|all\s+3|three|3|shoot\s+them\s+all|shoot\s+all)\b", re.IGNORECASE)
_TWO_RE = re.compile(r"\b(two|2|both)\b", re.IGNORECASE)
_LADDER_UP_RE = re.compile(r"\b(climb|go|head|get)\b.*\b(up|ladder\s+up|up\s+the\s+ladder)\b", re.IGNORECASE)
_LADDER_DOWN_RE = re.compile(r"\b(climb|go|head|get)\b.*\b(down|ladder\s+down|down\s+the\s+ladder)\b", re.IGNORECASE)
_LEVER_RE = re.compile(r"\b(pull|yank|throw|press)\b.*\b(lever)\b", re.IGNORECASE)
_TOWER_RE = re.compile(r"\b(go|head|move)\b.*\b(tower)\b", re.IGNORECASE)
_JUMP_MOAT_RE = re.compile(r"\b(jump)\b.*\b(moat)\b", re.IGNORECASE)
_SWIM_RE = re.compile(r"\b(swim|swim\s+across|cross\s+the\s+moat)\b", re.IGNORECASE)
_JUMP_GENERIC_RE = re.compile(r"\bjump\b.*\b(down|off|from)\b", re.IGNORECASE)

_CROSSBOW_RE = re.compile(r"\b(crossbow|bow|weapon)\b", re.IGNORECASE)

_JUMP_TO_GROUND_RE = re.compile(
    r"\b(jump|drop)\b.*\b(down|onto|to)\b.*\b(lawn|grass|ground|courtyard)\b",
    re.IGNORECASE
)


# ord/fraser som tyder på ljus i narrationen
_TORCHLIGHT_WORDS_RE = re.compile(
    r"\b(torchlight|by\s+torchlight|the\s+torch(?:'s)?\s+faint\s+light|faint\s+light|dim(?:ly)?\s+lit|lit\s+torch|with\s+(?:your|the)\s+(?:lit\s+)?torch)\b",
    re.IGNORECASE
)

_TORCH_POSSESSION_CLAIM_RE = re.compile(
    r"\b(with\s+(?:your|the)\s+torch|grab(?:s|bing)?\s+the\s+torch|pick(?:s|ing)?\s+up\s+the\s+torch|holding\s+the\s+torch)\b",
    re.IGNORECASE
)

_SEEING_DETAILS_IN_COAL_RE = re.compile(
    r"\b(see|make\s+out|glimpse|visible)\b.*\b(door|stair|staircase|steps?|coal|heaps?)\b",
    re.IGNORECASE
)
_OTHER_SIDE_SWIM_RE = re.compile(
    r"\b(go|get|climb|scramble|haul|make)\b.*\b(other\s+side|far\s+bank|opposite\s+bank|far\s+side)\b",
    re.IGNORECASE
)


def infer_move_event(current_room: str, text: str) -> Optional[str]:
    t = (text or "").lower()

    if current_room == "cell_01":
        # Down through the hole / to the cellar

            # NEW: allow bare 'crawl', 'crawl in', or 'crawl down' to mean entering the hole
        if re.search(r"^\s*crawl(?:\s+(?:in|down))?\s*$", t):
            return "enter_coal_cellar"

        # NEW: 'go back' in the cell means back down into the cellar
        if re.search(r"\bgo\s+back\b", t):
            return "enter_coal_cellar"


        if re.search(r"\b(crawl|go|climb|head|move|enter)\b.*\b(hole|opening|crawl(?:space)?)\b", t):
            return "enter_coal_cellar"
        if re.search(r"\b(?:to|towards?)\b.*\b(?:coal\s+cellar|cellar)\b", t):
            return "enter_coal_cellar"
        if "cellar" in t and any(w in t for w in ["go", "enter", "head", "toward", "to"]):
            return "enter_coal_cellar"
        
        # NEW: allow terse phrasing "crawl down" from the cell to the cellar
        if re.search(r"\bcrawl\s+down\b", t):
            return "enter_coal_cellar"

                # short intent like "crawl back" (engine will still gate on stone_moved)
        if re.search(r"\bcrawl\b.*\bback\b", t) and re.search(r"\b(hole|opening|crawl(?:space)?)\b", t):
            return "enter_coal_cellar"



    elif current_room == "coal_01":
        
                # Allow pronoun phrasing: "go to the door and open it"
        if "door" in t and re.search(r"\bopen\s+it\b", t):
            return "open_hall_door"

        
        # öppna/gå igenom dörren längst bort i källaren
        if re.search(r"\b(open|unlatch|unlock|go\s+through|enter)\b.*\bdoor\b", t):
            return "open_hall_door"

        # Up through the hole or back to (prison) cell
        up_back = any(w in t for w in ["back", "go back", "head back", "return", "up", "climb", "crawl up", "back up", "go up"])
        via_hole = re.search(r"\b(hole|opening|crawl(?:space)?)\b", t) is not None
        to_cell_phrase = re.search(r"\b(?:prison\s+cell|the\s+cell)\b", t) is not None
        to_cell_generic = _CELL_WORD.search(t) is not None  # 'cell' as a word, not 'cellar'
        if (up_back and via_hole) or (up_back and (to_cell_phrase or to_cell_generic)):
            return "return_to_cell"
        if re.search(r"\b(?:return|go|head|crawl|climb)\b.*\bto\b.*\b(?:prison\s+cell|the\s+cell|cell\b(?!ar))", t):
            return "return_to_cell"

    elif current_room == "hall_01":

                # Pronoun / bare-verb support for the obvious courtyard door
        if re.search(r"\b(unlock|open)\s+it\b", t) or re.search(r"^\s*(unlock|open)\s*$", t):
            return "unlock_courtyard_door"

        # öppna/låsa upp gårdsdörren – behandlas lika (motorn kräver keys i inventory)
        if re.search(r"\b(open|unlock|use\s+keys?|go\s+through|enter)\b.*\b(courtyard|heavy)?\s*door\b", t):
            return "unlock_courtyard_door"
        # gå tillbaka till källaren
        if re.search(r"\b(return|go\s+back|head\s+back|back)\b", t) and re.search(r"\b(coal(?:\s+cellar)?|cellar)\b", t):
            return "return_to_coal"
        
    elif current_room == "courtyard_01":

        # Exit synonyms -> crossing the bridge (validator will require gate lowered)
        if re.search(r"\b(exit|leave|head\s+out|go\s+out|get\s+out|leave\s+the\s+castle|exit\s+the\s+castle)\b", t):
            return "cross_gate_bridge"
        
        if _JUMP_TO_GROUND_RE.search(t):
            return "climb_ladder_down"

        # VATTEN först
        if _JUMP_MOAT_RE.search(t):
            return "jump_into_moat"
        # "other side" ska bara tolkas som simning om man inte pratar om stege/torn
        if _SWIM_RE.search(t) or (_OTHER_SIDE_SWIM_RE.search(t) and not re.search(r"\b(ladder|tower)\b", t)):
            return "swim_across"

        # Generiskt "jump down/off/from" → behandla som moat-försök (motorn gate:ar från marken)
        if _JUMP_GENERIC_RE.search(t):
            return "jump_into_moat"
        # Om spelaren säger "go to the tower" → behandla som att klättra upp
        if _TOWER_RE.search(t):
            return "climb_ladder_up"

        # STEGE före gate/bridge för att undvika falska "run across"
        if _LADDER_UP_RE.search(t) or re.search(r"\bclimb\b.*\bladder\b.*\bup\b", t):
            return "climb_ladder_up"
        if _LADDER_DOWN_RE.search(t) or re.search(r"\bclimb\b.*\bladder\b.*\bdown\b", t):
            return "climb_ladder_down"

        # Spak/skjuta/gå över bron (endast uttryckliga gate/bridge-ord)
        if _LEVER_RE.search(t):
            return "pull_lever"
        if _SHOOT_RE.search(t):
            return "shoot_guard"
        if _CROSS_GATE_RE.search(t):
            return "cross_gate_bridge"




    return None


_KEYS_RE = re.compile(r"\b(key|keys|keyring|keychain|key\s*ring)\b", re.IGNORECASE)
_UNLOCK_RE = re.compile(r"\b(unlock|use\s+key|use\s+keys|open\s+with\s+(?:key|keys))\b", re.IGNORECASE)

def infer_item_event(text: str) -> Optional[str]:
    t = (text or "").lower()
    mentions_torch = ("torch" in t) or ("stick" in t)  # OBS: ‘fire’ är inte längre en fackel-signal
    mentions_keys = _KEYS_RE.search(t) is not None
    mentions_crossbow = _CROSSBOW_RE.search(t) is not None
    if mentions_torch and _EXTINGUISH_RE.search(t):
        return "extinguish_torch"
    if mentions_torch and _DROP_RE.search(t):
        return "drop_torch"
    if mentions_torch and _PICK_RE.search(t):
        return "pickup_torch"
    if mentions_torch and _LIGHT_TORCH_RE.search(t):
        return "light_torch"


    if mentions_keys and _DROP_RE.search(t):
        return "drop_keys"
    if mentions_keys and _PICK_RE.search(t):
        return "pickup_keys"
    if _UNLOCK_RE.search(t) and mentions_keys:
        return "unlock_courtyard_door"
    
    if mentions_crossbow and _DROP_RE.search(t):
        return "drop_crossbow"
    if mentions_crossbow and _PICK_RE.search(t):
        return "pickup_crossbow"

    return None


def inventory_items_from_items(state: GameState) -> List[str]:
    torch = state.items.get("torch", {})
    keys = state.items.get("keys", {})
    crossbow = state.items.get("crossbow", {})
    if torch.get("location") == "player":
        return ["lit torch" if torch.get("lit") else "wooden torch"]
    if keys.get("location") == "player":
        return ["keys"]
    if crossbow.get("location") == "player":
        return ["crossbow"]
    return []



def torch_light_present_here(state: GameState) -> bool:
    """Is there torchlight in the current room (carried or placed lit here)?"""
    torch = state.items.get("torch", {})
    if not torch.get("lit"):
        return False
    loc = torch.get("location")
    return loc == "player" or loc == state.current_room


def sync_flags_with_items(state: GameState) -> None:
    """Keep legacy flags in sync for LLM context (mirror light to specific rooms)."""
    torch = state.items.get("torch", {})
    holding = (torch.get("location") == "player")
    lit_global = bool(torch.get("lit", False))

    # Ljuset i respektive rum — definieras uttryckligen per rum (inte beroende av current_room)
    lit_in_cell = lit_global and (torch.get("location") in ("player", "cell_01"))
    lit_in_coal = lit_global and (torch.get("location") in ("player", "coal_01"))
    lit_in_hall = lit_global and (torch.get("location") in ("player", "hall_01"))

    # Coal-flaggor ska spegla om det vore ljust där (oavsett vilket rum du är i)
    state.flags_coal["has_torch_stick"] = (holding and not lit_global)
    state.flags_coal["torch_lit"] = lit_in_coal

    # Spegla bara cell-flaggor när vi ÄR i cellen (för LLM-kontexten i det rummet)
    if state.current_room == "cell_01":
        state.flags_cell["has_torch_stick"] = (holding and not lit_global)
        state.flags_cell["torch_lit"] = lit_in_cell
    else:
        state.flags_cell["has_torch_stick"] = False
        state.flags_cell["torch_lit"] = False





def coerce_llm_result(raw: Dict[str, Any]) -> LLMResult:
    narration = str(raw.get("narration", "") or "").strip()

    # noise_level
    try:
        noise_level = int(raw.get("noise_level", 0) or 0)
    except Exception:
        noise_level = 0

    # hp_delta
    try:
        hp_delta = int(raw.get("hp_delta", 0) or 0)
    except Exception:
        hp_delta = 0

    # events (acceptera list eller dict{event: bool})
    ev_raw = raw.get("events", [])
    if isinstance(ev_raw, dict):
        events = [k for k, v in ev_raw.items() if v]
    elif isinstance(ev_raw, list):
        events = list(ev_raw)
    else:
        events = []

    # flags_set (acceptera list eller dict{flag: bool})
    fs_raw = raw.get("flags_set", [])
    if isinstance(fs_raw, dict):
        flags_set = [k for k, v in fs_raw.items() if v]
    elif isinstance(fs_raw, list):
        flags_set = list(fs_raw)
    else:
        flags_set = []

    # progression som text
    prog_raw = raw.get("progression", "")
    progression = prog_raw if isinstance(prog_raw, str) else ""

    safety_reason = str(raw.get("safety_reason", "") or "").strip()

    return LLMResult(narration, noise_level, hp_delta, events, flags_set, progression, safety_reason)


def print_room_banner(room_id: str) -> None:
    if room_id == "cell_01":
        print("\n— You are in the Prison Cell. —\n")
    elif room_id == "coal_01":
        print("\n— You are in the Coal Cellar. —\n")
    elif room_id == "hall_01":
        print("\n— You are in the Great Hall. —\n")
    elif room_id == "courtyard_01":
        print("\n— You are in the Castle Courtyard. —\n")
    else:
        print(f"\n— You are in: {room_id} —\n")


COAL_INTRO_TEXT = (
    "It’s pitch-black. Coal dust clings to your throat; the stone floor feels uneven underfoot. "
    "Heaps of coal hem you in on both sides. As you steady yourself, your foot bumps against something lying on the ground. "
    "You can feel it with your toes, but you can’t tell what it is until you pick it up."
)

HALL_INTRO_TEXT = (
    "A lofty chamber opens before you: five stone pillars on each side, chandeliers burning above, portraits along the walls. "
    "Near the door you entered stands a mail-clad knight with his back turned; a keyring hangs from his belt. "
    "Across the hall, a heavy door leads to the courtyard."
)

COURTYARD_INTRO_TEXT = (
    "The night air is cold. A broad grassy courtyard is ringed by torchlit stone walls under a starry sky. "
    "Fifty meters ahead a massive wooden gate stands shut. To its right, a squat wall-tower has a ladder up to a small platform. "
    "Up there you can see a lever, a crossbow leaning against the parapet, and a big heap of bolts. Beyond the outer wall yawns a moat—thirty meters down."
)



def validate_and_apply(state: GameState, llm: LLMResult, player_action_text: str) -> Dict[str, Any]:
    """
    Apply deterministic policies for the current room; validate allowed state changes; handle room transitions and victory.
    """
    notes: List[str] = []
    crossbow_nohold = False
    scrubbed_illegal_events = False

    # ----- Local helpers -----
    def _set_narr(msg: str):
        llm.narration = msg

    # ----- Local intent regex (kept local so you can paste this function alone) -----
    import re as _re

    ADVICE_QUERY_RE = _re.compile(r"\bwhat\s+should\s+i\s+do\b", _re.IGNORECASE)

    MOVE_DEEPER_RE = _re.compile(
        r"\b(walk|move|proceed|advance|head|go|step|explore|"
        r"make\s+(?:your|my)\s+way|feel\s+(?:your|my)\s+way|grope|"
        r"run|sprint|dash|charge|rush|jog|hurry|bolt|dive|burrow|crawl|roll|tumble)\b",
        _re.IGNORECASE
    )
    LOOK_RE = _re.compile(
        r"\b(look(?:\s+around|\s+about)?|examine|inspect|scan|peer|peek|observe|search)\b",
        _re.IGNORECASE
    )
    USE_TORCH_RE = _re.compile(
        r"(?:\b(use|shine|raise|brandish|wave|aim|point)\b.*\btorch\b|\btorch\b.*\b(use|shine|raise|brandish|wave|aim|point)\b)",
        _re.IGNORECASE
    )
    SEE_QUERY_RE = _re.compile(r"\bwhat\s+(?:can|do)\s+i\s+see\b", _re.IGNORECASE)

    # Courtyard-specific intents
    _SHOOT_RE = _re.compile(r"\b(shoot|fire|loose|let\s+fly|squeeze(?:\s+the)?\s+trigger|aim\s+and\s+fire)\b", _re.IGNORECASE)
    _ALL_THREE_RE = _re.compile(r"\b(all\s+three|all\s+3|three|3|shoot\s+them\s+all|shoot\s+all)\b", _re.IGNORECASE)
    _TWO_RE = _re.compile(r"\b(two|2|both)\b", _re.IGNORECASE)
    _LADDER_UP_RE = _re.compile(r"\b(climb|go|head|get)\b.*\b(up|ladder\s+up|up\s+the\s+ladder)\b", _re.IGNORECASE)
    _LADDER_DOWN_RE = _re.compile(r"\b(climb|go|head|get)\b.*\b(down|ladder\s+down|down\s+the\s+ladder)\b", _re.IGNORECASE)
    _LEVER_RE = _re.compile(r"\b(pull|yank|throw|press)\b.*\b(lever)\b", _re.IGNORECASE)
    _JUMP_MOAT_RE = _re.compile(r"\b(jump)\b.*\b(moat)\b", _re.IGNORECASE)
    _SWIM_RE = _re.compile(r"\b(swim|swim\s+across|cross\s+the\s+moat)\b", _re.IGNORECASE)
    _CROSSBOW_RE = _re.compile(r"\b(crossbow|bow|weapon)\b", _re.IGNORECASE)

    # --- snapshots/intent för stenlogik i cellen ---
    was_stone_moved_before = False
    attempted_move_stone_input = False
    if state.current_room == "cell_01":
        was_stone_moved_before = bool(state.flags_cell.get("stone_moved", False))
        if _STONE_ACT_RE1.search(player_action_text or "") or _STONE_ACT_RE2.search(player_action_text or ""):
            attempted_move_stone_input = True
        # snapshot – hade vi redan hittat den lösa stenen?
        was_found_before = bool(state.flags_cell.get("found_loose_stone", False))
    else:
        was_found_before = bool(state.flags_cell.get("found_loose_stone", False))

    # --- snapshot: var riddaren redan utslagen före denna tur? ---
    was_knight_out_before = bool(state.flags_hall.get("knight_knocked_out", False))

    room_transition: Optional[str] = None
    game_won: bool = False

    # Snapshot: holding torch at start of this turn (used for hall combat damage)
    torch_snapshot = state.items.get("torch", {"location": None, "lit": False})
    holding_torch_snapshot = (torch_snapshot.get("location") == "player")

    # Bound noise level
    noise = max(0, min(3, int(llm.noise_level)))
    if noise != llm.noise_level:
        notes.append(f"Noise coerced from {llm.noise_level} to {noise}.")

    # Strictly filter flags reported by the LLM to what is allowed for the current room
    if state.current_room == "cell_01":
        allowed_flags = ALLOWED_CELL_FLAGS
    elif state.current_room == "coal_01":
        allowed_flags = ALLOWED_COAL_FLAGS
    elif state.current_room == "hall_01":
        allowed_flags = ALLOWED_HALL_FLAGS
    elif state.current_room == "courtyard_01":
        allowed_flags = globals().get("ALLOWED_COURTYARD_FLAGS", {"at_tower_top", "gate_lowered", "guards_present"})
    else:
        allowed_flags = set()
    if llm.flags_set:
        original_flags = list(llm.flags_set)
        llm.flags_set = [f for f in llm.flags_set if f in allowed_flags]
        if llm.flags_set != original_flags:
            notes.append(f"Filtered LLM flags from {original_flags} to {llm.flags_set} for room {state.current_room}.")

    # Events sanitized
    events = [e for e in llm.events if e in ALLOWED_EVENTS]

    # 'dark_stumble' ska bara finnas i coal_01
    if state.current_room != "coal_01" and "dark_stumble" in events:
        events = [e for e in events if e != "dark_stumble"]
        notes.append("Removed 'dark_stumble' outside the coal cellar.")
        scrubbed_illegal_events = True

        # Coal cellar correction: if torchlight present, scrub any dark stumble
    if state.current_room == "coal_01" and torch_light_present_here(state):
        if "dark_stumble" in events:
            events = [e for e in events if e != "dark_stumble"]
            notes.append("Removed dark_stumble since torchlight is present in coal cellar.")
        if llm.hp_delta < 0:
            llm.hp_delta = 0
        if llm.safety_reason.lower().startswith("dark"):
            llm.safety_reason = ""


    # Intent inference from raw input
    ev_move = infer_move_event(state.current_room, player_action_text)
    if ev_move and ev_move not in events:
        events.append(ev_move)
        notes.append(f"Inferred movement event from input: {ev_move}")

    ev_item = infer_item_event(player_action_text)
    if ev_item and ev_item not in events:
        events.append(ev_item)
        notes.append(f"Inferred item event from input: {ev_item}")

    # Straw rummage => deterministiskt: hitta den lösa stenen denna tur
    if state.current_room == "cell_01" and not state.flags_cell["found_loose_stone"]:
        if _re.search(r"\b(straw|hay|straw\s*bed|bed)\b", player_action_text or "", flags=_re.IGNORECASE):
            if "straw_rummaged" not in events:
                events.append("straw_rummaged")
            state.flags_cell["found_loose_stone"] = True
            # Force en konsekvent rad så vi inte får "inget" + "du hittade" samtidigt
            # Keep LLM's first sentence as flavor (if any), then add the fixed discovery line.
            orig = (llm.narration or "").strip()
            first = ""
            if orig:
                parts = re.split(r"(?<=[.!?])\s+", orig)
                first = (parts[0] or "").strip()
            fixed = "You notice a loose cobblestone under the straw."

            # Avoid duplicating if the LLM already said it
            already_said = re.search(r"\bloose\s+(?:stone|cobblestone)\b", orig, re.IGNORECASE)
            if first and not already_said and not re.search(r"\bNothing special happened\.\s*$", first):
                llm.narration = f"{first} {fixed}"
            else:
                llm.narration = fixed

            
            

    # Dedupe while preserving order
    events = list(dict.fromkeys(events))

    # Coal cellar: resolve contradictory movement intents in one turn
    if state.current_room == "coal_01":
        if ("return_to_cell" in events) and ("open_hall_door" in events):
            inferred = infer_move_event("coal_01", player_action_text)
            if inferred == "return_to_cell":
                events = [e for e in events if e != "open_hall_door"]
                notes.append("Kept 'return_to_cell' and removed 'open_hall_door' based on input.")
            elif inferred == "open_hall_door":
                events = [e for e in events if e != "return_to_cell"]
                notes.append("Kept 'open_hall_door' and removed 'return_to_cell' based on input.")
            else:
                if not torch_light_present_here(state):
                    events = [e for e in events if e != "open_hall_door"]
                    notes.append("Preferred safe 'return_to_cell' over 'open_hall_door' in darkness.")
                else:
                    events = [e for e in events if e != "return_to_cell"]
                    notes.append("Preferred 'open_hall_door' with light present.")


    # Entering the cellar this turn cancels any stray dark_stumble from the LLM
    if "enter_coal_cellar" in events and "dark_stumble" in events:
        events = [e for e in events if e != "dark_stumble"]
        notes.append("Suppressed 'dark_stumble' on the same turn as entering the cellar.")

    # Returning from the hall into the cellar the same turn should also never stumble
    if "return_to_coal" in events and "dark_stumble" in events:
        events = [e for e in events if e != "dark_stumble"]
        notes.append("Suppressed 'dark_stumble' on the same turn as returning to the cellar.")


    # Ignorera 'unlock_courtyard_door' i andra rum än hallen
    if state.current_room != "hall_01" and "unlock_courtyard_door" in events:
        events = [e for e in events if e != "unlock_courtyard_door"]
        notes.append("Removed 'unlock_courtyard_door' outside the Great Hall.")
        scrubbed_illegal_events = True

    # Courtyard gate flag hygiene: require lever event from tower top
    if state.current_room == "courtyard_01":
        if ("gate_lowered" in llm.flags_set) and ("pull_lever" not in events or not state.flags_courtyard.get("at_tower_top", False)):
            llm.flags_set = [f for f in llm.flags_set if f != "gate_lowered"]
            notes.append("Ignored 'gate_lowered' flag without pulling the lever at the tower top.")

    # Convert drop+extinguish user intent into 'extinguish_torch'
    if "drop_torch" in events and _re.search(r"\b(extinguish|snuff|put\s+out|douse|blow\s+out|quench)\b", player_action_text or "", _re.IGNORECASE):
        events = ["extinguish_torch" if e == "drop_torch" else e for e in events]
        notes.append("Converted 'drop_torch' to 'extinguish_torch' based on player intent.")

    # Guard: disallow straw rummage unless player actually mentions straw/hay/bed
    if state.current_room == "cell_01":
        straw_mentioned = _re.search(r"\b(straw|hay|straw\s*bed|bed)\b", player_action_text or "", flags=_re.IGNORECASE) is not None
        if ("straw_rummaged" in events) and (not straw_mentioned):
            notes.append("Removed 'straw_rummaged' since player did not mention straw/hay/bed.")
            events = [e for e in events if e != "straw_rummaged"]
            if "found_loose_stone" in llm.flags_set:
                llm.flags_set = [f for f in llm.flags_set if f != "found_loose_stone"]
            if not state.flags_cell["found_loose_stone"]:
                llm.narration = "You look around the cell. Nothing special happened."

    # Cell: infer lifting the loose stone deterministically
    if state.current_room == "cell_01" and state.flags_cell.get("found_loose_stone", False) and not state.flags_cell.get("stone_moved", False):
        if _STONE_ACT_RE1.search(player_action_text or "") or _STONE_ACT_RE2.search(player_action_text or ""):
            if "stone_lifted" not in events:
                events.append("stone_lifted")
                notes.append("Inferred 'stone_lifted' from player intent.")

    # ---------------- Darkness intent inference (engine-level guard) ----------------
    dark_here = (state.current_room == "coal_01" and not torch_light_present_here(state))
    if dark_here:
        # If the player is trying to move deeper in the cellar (not returning up), force a stumble.
        # But don't punish stationary groping/feeling/picking up the nearby object.
        local_probe = _re.search(r"\b(pick\s*up|grab|feel|grope|pat|reach|search)\b", player_action_text or "", _re.IGNORECASE)
        if MOVE_DEEPER_RE.search(player_action_text or "") and "return_to_cell" not in events and not local_probe:
            if "dark_stumble" not in events:
                events.append("dark_stumble")
                notes.append("Auto-injected 'dark_stumble' due to moving in darkness in coal_01.")
                # Give a clear stumble line if the model didn't.
                if not llm.narration or "stumble" not in llm.narration.lower():
                    llm.narration = "You edge forward in the pitch black, stumble hard, and stay put."


    cancel_movement_this_turn = False
    # In the coal cellar, stumbling in darkness cancels moving deeper,
    # but MUST NOT cancel returning to the cell.
    if state.current_room == "coal_01" and "dark_stumble" in events and "return_to_cell" not in events:
        cancel_movement_this_turn = True

    # Försök att tända kolhögarna i källaren: alltid realistiskt avslag
    if state.current_room == "coal_01":
        if _re.search(r"\b(coal|coal\s+piles?)\b", player_action_text or "", _re.IGNORECASE) and \
           _re.search(r"\b(ignite|light|set|burn)\b", player_action_text or "", _re.IGNORECASE):
            llm.narration = ("You try to ignite the coal, but the dust is damp and there's no draft—"
                             "nothing catches. Nothing special happened.")

    # ---------------- HP / Punishments ----------------
    enforced_hp_delta = 0
    cause = ""

    if state.current_room == "cell_01":
        if noise >= 2:
            enforced_hp_delta = -20
            # Scrub contradictory 'oblivious/unaware' lines this turn (guard reacts)
            llm.narration = re.sub(
                r"(?:but\s+)?the\s+guard\s+(?:remains\s+)?(?:oblivious|unaware|does(?:\s+not)?\s+notice)[^.]*\.",
                "", llm.narration, flags=re.IGNORECASE
            ).strip()
            llm.narration = re.sub(r"\s*Nothing special happened\.\s*$", "", llm.narration).strip()

            cause = "Guard strikes you for disturbing his slumber"
            if llm.hp_delta != -20:
                notes.append(f"HP delta overridden to -20 due to guard punishment (noise={noise}) in cell.")
            if "guard_punishes" not in events:
                events.append("guard_punishes")
        else:
            # Rensa felaktigt LLM-påhittat straff vid låg noise
            if "guard_punishes" in events:
                events = [e for e in events if e != "guard_punishes"]
                notes.append("Removed 'guard_punishes' because noise < 2 in cell.")


            if llm.hp_delta != 0:
                notes.append(f"Ignored non-guard hp_delta={llm.hp_delta} (noise={noise}) in cell.")

    elif state.current_room == "coal_01":
        if "dark_stumble" in events:
            if torch_light_present_here(state):
                notes.append("Ignored 'dark_stumble' because torchlight is present in this room.")
                events = [e for e in events if e != "dark_stumble"]
            else:
                enforced_hp_delta = -10
                cause = "You stumble in the dark"
                if llm.hp_delta != -10:
                    notes.append("HP delta overridden to -10 for dark stumble in coal cellar.")
        else:
            if llm.hp_delta != 0:
                notes.append(f"Ignored hp_delta in coal without 'dark_stumble': {llm.hp_delta}")

    elif state.current_room == "hall_01":

                # Normalisera spelarens input för hall-intent
        lower_action = (player_action_text or "").lower()
        # Räknas som icke-attack (t.ex. "drop the torch and take the keys")
        non_attack_item_action = (_DROP_RE.search(lower_action) is not None) or (_PICK_RE.search(lower_action) is not None)
        # Riktiga våldsintentioner (verb), inte bara ordet "torch"
        attack_intent = (HIT_VERBS_RE.search(lower_action) is not None) or re.search(r"\b(attack|smash|bash|stab|burn|swing)\b", lower_action)


        # Rensa bort falsk riddarstrid om noise < 2
        if noise < 2:
            if any(e in {"knight_notice", "combat_knock_guard"} for e in events):
                events = [e for e in events if e not in {"knight_notice", "combat_knock_guard"}]
                notes.append("Removed knight events because noise < 2 in hall.")


        # 'open_hall_door' hör ENDAST hemma i coal_01 => scrub i hallen
        if "open_hall_door" in events:
            events = [e for e in events if e != "open_hall_door"]
            notes.append("Removed 'open_hall_door' in hall (only valid from coal_01).")
            scrubbed_illegal_events = True

            if not state.flags_hall["courtyard_door_unlocked"]:
                llm.narration = "The courtyard door is locked. Nothing special happened."

        # Stage 2: Knight combat if too noisy and not already knocked out
        if noise >= 2 and not state.flags_hall["knight_knocked_out"]:
            # ensure events are present
            if "knight_notice" not in events:
                events.append("knight_notice")
            if "combat_knock_guard" not in events:
                events.append("combat_knock_guard")

            # damage rule: -30 if torch in hand this turn, else -50
            enforced_hp_delta = -30 if holding_torch_snapshot else -50
            cause = "The knight hears you; you struggle and knock him out"
            state.flags_hall["knight_knocked_out"] = True  # permanently unconscious

            # strengthen narration
            if llm.narration and llm.narration[-1] not in ".!?":
                llm.narration += "."
            # Keep up to one short flavor sentence, then canonical KO line
            orig = (llm.narration or "").strip()
            flavor = ""
            parts = re.split(r"(?<=[.!?])\s+", orig) if orig else []
            if parts and parts[0]:
                first = parts[0].strip()
                if not re.search(r"\bThe knight whirls at the noise; you clash briefly and you knock him out cold\.\b", first, re.IGNORECASE):
                    flavor = first[:140].strip()  # short flavor

            fixed = "The knight whirls at the noise; you clash briefly and you knock him out cold."
            llm.narration = (f"{flavor} {fixed}".strip() if flavor else fixed)


        # If the knight is already out, scrub any new notice/combat and correct the tone
        if was_knight_out_before:
            # Ta bort stridshändelser
            if any(e in {"knight_notice", "combat_knock_guard"} for e in events):
                events = [e for e in events if e not in {"knight_notice", "combat_knock_guard"}]
                notes.append("Removed knight events after he was already unconscious.")

            # Korrigera all text som påstår att han reagerar, står upp, duckar etc.
            REACTION_RE = re.compile(
                r"\b("
                r"turns?|stumbles?|spins?|lurch(?:es|ed)?|reacts?|flinches?|"
                r"eyes\s+(?:narrow|widen|wide(?:ns)?)|notic\w*|"
                r"ducks?|dodg(?:e|es|ed)|avoids?|parries|blocks|"
                r"stands?|standing|upright|rises?|gets?\s+up|with\s+his\s+back\s+turned"
                r")\b",
                re.IGNORECASE
            )

            if re.search(r"\bknight\b", llm.narration or "", re.IGNORECASE) and REACTION_RE.search(llm.narration or ""):
                llm.narration = "The knight is unconscious on the ground. Further actions against him have no effect."

                        # Om spelaren verkligen försöker attackera igen (inte lägga/släppa/plocka),
            # skriv över narrationen. Att bara nämna "torch" räcker inte.
            if attack_intent and not non_attack_item_action:
                llm.narration = "The knight is unconscious on the ground. Further actions against him have no effect."





        # Clamp narration claiming to pass through the courtyard door without actually unlocking
        if not state.flags_hall["courtyard_door_unlocked"] and not room_transition:
            if re.search(r"\b(walk|go|step|move|slip)\b.*\b(through|into|out)\b.*\b(courtyard|door)\b", llm.narration, re.IGNORECASE):
                llm.narration = "The courtyard door is locked. Nothing special happened."

        # Rörelse tillbaka
                # Rörelse tillbaka
        can_go_back = "coal_01" in ROOM_GRAPH.get("hall_01", [])
        if "return_to_coal" in events and can_go_back:
            torch_obj = state.items.get("torch", {})
            carrying_lit_torch = (torch_obj.get("location") == "player" and torch_obj.get("lit", False))
            if not carrying_lit_torch:
                events = [e for e in events if e != "return_to_coal"]
                llm.narration = "Too dangerous to go back into the coal cellar without a lit torch."
                notes.append("Blocked return to coal cellar due to not carrying a lit torch.")
            else:
                room_transition = "coal_01"



        # Unlock/open courtyard (requires keys in inventory) -> transition to courtyard_01
        if "unlock_courtyard_door" in events:
            # Kräv uttrycklig input-intent för att låsa upp/öppna ut till gården
            inferred = infer_move_event("hall_01", player_action_text)
            if inferred != "unlock_courtyard_door":
                notes.append("Blocked courtyard unlock due to missing explicit input intent.")
                events = [e for e in events if e != "unlock_courtyard_door"]
                llm.narration = "You’re by a heavy door, but you haven’t committed to unlocking it. Nothing special happened."
            else:
                keys = state.items.get("keys", {})
                if keys.get("location") == "player":
                    if not state.flags_hall["courtyard_door_unlocked"]:
                        state.flags_hall["courtyard_door_unlocked"] = True
                    room_transition = "courtyard_01"
                    if llm.narration and llm.narration[-1] not in ".!?":
                        llm.narration += "."
                    llm.narration = (llm.narration + " You pull the heavy door open and step out into the night air of the courtyard.").strip()
                else:
                    notes.append("Tried to unlock courtyard door without keys.")
                    llm.narration = "The door is locked and you have no keys. Nothing special happened."


    elif state.current_room == "courtyard_01":

        # Rensa 'guards_arrive' om inte levern dragits från tower top just nu
        if "guards_arrive" in events and ("pull_lever" not in events or not state.flags_courtyard.get("at_tower_top", False)):
            events = [e for e in events if e != "guards_arrive"]
            notes.append("Removed 'guards_arrive' without valid 'pull_lever' at tower top.")


        if "swim_across" in events:
            events = [e for e in events if e not in {"climb_ladder_up", "climb_ladder_down"}]

        # --- Infer shooting count from player's text ---
        t = (player_action_text or "").lower()
        shoot_count = 0
        if "shoot_guard" in events:
            if _ALL_THREE_RE.search(t):
                shoot_count = 3
            elif _TWO_RE.search(t):
                shoot_count = 2
            else:
                shoot_count = 1

        # Cannot reach the ladder from the moat
        if state.flags_courtyard.get("in_moat", False) and (
            "climb_ladder_up" in events or "climb_ladder_down" in events
        ):
            notes.append("Denied ladder movement from the moat.")
            events = [e for e in events if e not in {"climb_ladder_up", "climb_ladder_down"}]
            _set_narr("The ladder is well above the waterline. You can’t reach it from the moat. Nothing special happened.")

        # Movement on the ladder
        if "climb_ladder_up" in events:
            state.flags_courtyard["at_tower_top"] = True
            state.flags_courtyard["in_moat"] = False
            if llm.narration and llm.narration[-1] not in ".!?":
                llm.narration += "."
            llm.narration = "You climb the ladder and step onto the small platform."

        # Ignore ladder-down if already on the grass
        if "climb_ladder_down" in events and not state.flags_courtyard.get("at_tower_top", False):
            events = [e for e in events if e != "climb_ladder_down"]
            if not ("pull_lever" in events or "shoot_guard" in events or "cross_gate_bridge" in events or "jump_into_moat" in events or "swim_across" in events):
                llm.narration = "You’re already on the grass. Nothing special happened."



        if "climb_ladder_down" in events:
            state.flags_courtyard["at_tower_top"] = False
            state.flags_courtyard["in_moat"] = False
            if llm.narration and llm.narration[-1] not in ".!?":
                llm.narration += "."
            if _re.search(r"\bjump\b", player_action_text or "", _re.IGNORECASE):
                llm.narration = "You jump down to the grass, knees jolting."
            else:
                llm.narration = "You climb back down to the grass."

        # Pulling the lever: only from the tower platform; lowers gate and summons guards
        if "pull_lever" in events:
            if not state.flags_courtyard.get("at_tower_top", False):
                notes.append("Tried to pull lever from the ground; denied.")
                llm.narration = "The lever is out of reach from the ground. You’ll need to climb up the ladder first. Nothing special happened."
                # ta bort lever-relaterade events den här turen
                events = [e for e in events if e not in {"pull_lever", "guards_arrive"}]
            else:
                noise = max(noise, 3)
                gate_was_lowered = bool(state.flags_courtyard.get("gate_lowered", False))
                state.flags_courtyard["gate_lowered"] = True
                if not gate_was_lowered:
                    if not state.flags_courtyard["guards_present"]:
                        state.flags_courtyard["guards_present"] = True
                        state.flags_courtyard["guards_remaining"] = 3
                    if "guards_arrive" not in events:
                        events.append("guards_arrive")
                    if llm.narration and llm.narration[-1] not in ".!?":
                        llm.narration += "."
                    llm.narration = (llm.narration + " With a thunderous slam the gate drops into a bridge, and three armored guards sprint across the lawn toward you.").strip()
                else:
                    notes.append("Lever pulled again; gate already lowered — no new guards spawned.")


        # Shooting (allowed from tower or ground) but requires crossbow in hand
        if shoot_count > 0:
            holding_crossbow_now = (state.items.get("crossbow", {}).get("location") == "player")
            if not holding_crossbow_now:
                # Remove illegal shoot so engine/state stays consistent; narration is exactly per spec
                events = [e for e in events if e != "shoot_guard"]
                llm.narration = "You are not holding the crossbow."
                crossbow_nohold = True
            elif not state.flags_courtyard.get("guards_present", False) or int(state.flags_courtyard.get("guards_remaining", 0)) <= 0:
                llm.narration = "There are no guards to shoot."
            else:
                kills = min(shoot_count, int(state.flags_courtyard["guards_remaining"]))
                state.flags_courtyard["guards_remaining"] -= kills
                if state.flags_courtyard["guards_remaining"] <= 0:
                    state.flags_courtyard["guards_present"] = False

                # Bestäm platsfras
                locus = "from the platform" if state.flags_courtyard.get("at_tower_top", False) else "from the grass"
                shot_line = (
                    f"You fire the crossbow {locus}; a guard drops."
                    if kills == 1 else
                    f"You fire the crossbow {locus}; {kills} guards drop."
                )

                # Om LLM inte redan uttryckligen beskriver att en vakt faller – lägg till vår tydliga rad
                if not re.search(r"\b(guard|guards)\b.*\b(drop|falls?|fall|crumple|collapse|go\s+down|die|dies)\b",
                                llm.narration or "", re.IGNORECASE):
                    if llm.narration and llm.narration.strip() and llm.narration.strip()[-1] not in ".!?":
                        llm.narration += "."
                    llm.narration = (llm.narration + " " + shot_line).strip() if llm.narration else shot_line

                # Lägg alltid till återstående antal/”lawn falls silent” om det inte redan sägs
                if state.flags_courtyard["guards_remaining"] > 0:
                    if not re.search(r"\b(remain|left)\b", llm.narration, re.IGNORECASE):
                        llm.narration += f" {state.flags_courtyard['guards_remaining']} remain."
                else:
                    if not re.search(r"\bfalls silent\b|\bno (?:one|guards) remain\b", llm.narration, re.IGNORECASE):
                        llm.narration += " The lawn falls silent."


        # Jump into moat: only from tower top; -40 HP, then you are in the moat
        if "jump_into_moat" in events:
            if state.flags_courtyard.get("at_tower_top", False):
                enforced_hp_delta = -40
                if llm.hp_delta != -40:
                    notes.append("HP delta overridden to -40 for moat jump.")
                cause = "You plunge into the moat"
                state.flags_courtyard["in_moat"] = True
                state.flags_courtyard["at_tower_top"] = False
                if llm.narration and llm.narration[-1] not in ".!?":
                    llm.narration += "."
                llm.narration = (llm.narration + " You leap from the platform, plummet thirty meters, and crash into the cold water.").strip()
            else:
                notes.append("Jump into moat attempted from ground; denied.")
                llm.narration = "Jumping here would be pointless—and painful. Climb the tower and jump into the moat if you dare. Nothing special happened."
                events = [e for e in events if e != "jump_into_moat"]


        # Swimming across from moat => victory (if still alive)
        pending_victory_via_swim = False
        if "swim_across" in events:
            if state.flags_courtyard.get("in_moat", False):
                pending_victory_via_swim = True
            else:
                notes.append("Tried to swim across while not in the moat; denied.")
                llm.narration = "You're not in the moat. Nothing special happened."
                events = [e for e in events if e != "swim_across"]


        # Crossing the gate bridge => victory (must be lowered; must be on ground and not in moat)
        pending_victory_via_gate = False
        if "cross_gate_bridge" in events:
            if not state.flags_courtyard.get("gate_lowered", False):
                notes.append("Tried to cross but gate not lowered.")
                _set_narr("The gate is still up; there is no bridge to cross. Nothing special happened.")
            elif state.flags_courtyard.get("in_moat", False):
                notes.append("Tried to cross from the moat.")
                _set_narr("You’re in the moat; you’ll need to reach the bank first. Nothing special happened.")
            elif state.flags_courtyard.get("at_tower_top", False):
                notes.append("Tried to cross from tower top.")
                _set_narr("You’ll have to climb down to the grass first. Nothing special happened.")
            else:
                pending_victory_via_gate = True

        # Ground damage each turn while guards remain and player ends turn on grass
        # (skip this damage if the player is achieving guaranteed victory this turn)
        if not (pending_victory_via_gate or pending_victory_via_swim):
            if (not state.flags_courtyard.get("at_tower_top", False)
                and not state.flags_courtyard.get("in_moat", False)
                and state.flags_courtyard.get("guards_present", False)
                and state.flags_courtyard.get("guards_remaining", 0) > 0):
                enforced_hp_delta += -60
                cause = "The charging guards batter you on the grass"

        # Schedule victory flags (main loop resolves death before victory)
        if pending_victory_via_gate:
            game_won = True
            if llm.narration and llm.narration[-1] not in ".!?":
                llm.narration += "."
            llm.narration = "You sprint across the lowered gate and vanish into the treeline beyond."

        if pending_victory_via_swim:
            game_won = True
            if llm.narration and llm.narration[-1] not in ".!?":
                llm.narration += "."
            llm.narration = "You swim hard, scramble up the far bank, and disappear into the forest."

    # ---------------- Derive flags from events ----------------
    if state.current_room == "cell_01":
        event_to_flag = {
            "stone_lifted": "stone_moved",
            "stone_moved": "stone_moved",
            "straw_rummaged": "found_loose_stone",
            "enter_coal_cellar": "entered_hole",
            "light_torch": "torch_lit",
        }
    else:
        event_to_flag = {
            "pickup_stick": "has_torch_stick",
            "pickup_torch": "has_torch_stick",
            "light_torch": "torch_lit",
            "pull_lever": "gate_lowered",  # Courtyard convenience
        }
    for e in list(events or []):
        f = event_to_flag.get(e)
        if f and f not in llm.flags_set:
            llm.flags_set.append(f)

    # ---------------- Apply cell flags (ordered) ----------------
    new_flags: List[str] = []

    if state.current_room == "cell_01":
        # found_loose_stone requires actual straw interaction
        if ("found_loose_stone" in llm.flags_set) and (not state.flags_cell["found_loose_stone"]):
            if ("straw_rummaged" in events) or ("stone_revealed" in events) or _re.search(r"\b(straw|hay|straw\s*bed|bed)\b", player_action_text or "", flags=_re.IGNORECASE):
                state.flags_cell["found_loose_stone"] = True
                new_flags.append("found_loose_stone")
            else:
                notes.append("Ignored 'found_loose_stone' without straw interaction.")

        # stone_moved requires found_loose_stone first
        if "stone_moved" in llm.flags_set:
            if state.flags_cell["found_loose_stone"] and not state.flags_cell["stone_moved"]:
                state.flags_cell["stone_moved"] = True
                new_flags.append("stone_moved")
            elif not state.flags_cell["found_loose_stone"]:
                notes.append("Cannot set 'stone_moved' before 'found_loose_stone'. Ignored.")

        if "entered_hole" in llm.flags_set:
            if state.flags_cell["stone_moved"] and not state.flags_cell["entered_hole"]:
                state.flags_cell["entered_hole"] = True
                new_flags.append("entered_hole")
            elif not state.flags_cell["stone_moved"]:
                notes.append("Cannot set 'entered_hole' before 'stone_moved'. Ignored.")

    # ---------------- Inventory & item events ----------------
    torch = state.items.get("torch", {"location": "coal_01", "lit": False})
    keys = state.items.get("keys", {"location": "hall_01"})
    crossbow = state.items.get("crossbow", {"location": "courtyard_tower_top"})
    holding_torch = (torch["location"] == "player")
    holding_keys  = (keys["location"]  == "player")
    holding_crossbow = (crossbow["location"] == "player")

    # 1) Process DROPS first (to free hands for same-turn pickups)
    if "extinguish_torch" in events:
        if holding_torch and torch.get("lit"):
            torch["lit"] = False
            state.items["torch"] = torch
            notes.append("Torch extinguished while held.")
            llm.narration = "You cup the flame and snuff it out."
        elif torch.get("location") == state.current_room and torch.get("lit"):
            torch["lit"] = False
            state.items["torch"] = torch
            notes.append("Torch extinguished on the ground in this room.")
            llm.narration = "You snuff the torch; darkness creeps back."
        else:
            notes.append("No lit torch to extinguish; ignoring.")
            llm.narration = "There’s no lit torch to extinguish. Nothing special happened."

    if "drop_torch" in events:
        if holding_torch:
            torch["location"] = state.current_room
            state.items["torch"] = torch
        else:
            notes.append("Tried to drop torch while not holding it; ignored.")

    if "drop_keys" in events:
        if holding_keys:
            keys["location"] = state.current_room
            state.items["keys"] = keys
        else:
            notes.append("Tried to drop keys while not holding them; ignored.")

    if "drop_crossbow" in events:
        if holding_crossbow:
            if state.current_room == "courtyard_01" and state.flags_courtyard.get("in_moat", False):
                notes.append("Denied dropping crossbow in moat.")
                llm.narration = "If you drop it here, it sinks into the moat. You hold on. Nothing special happened."
                events = [e for e in events if e != "drop_crossbow"]

            elif state.current_room == "courtyard_01" and state.flags_courtyard.get("at_tower_top", False):
                crossbow["location"] = "courtyard_tower_top"
                state.items["crossbow"] = crossbow
            else:
                crossbow["location"] = state.current_room
                state.items["crossbow"] = crossbow
        else:
            notes.append("Tried to drop crossbow while not holding it; ignored.")


    # Recompute slot occupancy after drops
    holding_torch = (state.items.get("torch", {}).get("location") == "player")
    holding_keys  = (state.items.get("keys", {}).get("location") == "player")
    holding_crossbow = (state.items.get("crossbow", {}).get("location") == "player")
    slot_occupied = holding_torch or holding_keys or holding_crossbow

    def _recalc_slot():
        nonlocal holding_torch, holding_keys, holding_crossbow, slot_occupied
        holding_torch = (state.items.get("torch", {}).get("location") == "player")
        holding_keys = (state.items.get("keys", {}).get("location") == "player")
        holding_crossbow = (state.items.get("crossbow", {}).get("location") == "player")
        slot_occupied = holding_torch or holding_keys or holding_crossbow

    # 2) Process PICKUPS after drops
    if ("pickup_stick" in events) or ("pickup_torch" in events) or ("has_torch_stick" in llm.flags_set):
        if holding_torch:
            notes.append("Already holding the torch; pickup ignored.")
        elif slot_occupied:
            notes.append("Hands full; cannot pick up torch while holding another item.")
            llm.narration = "Your hands are full. Drop what you're holding first. Nothing special happened."
        else:
            if torch["location"] == state.current_room:
                torch["location"] = "player"
                state.items["torch"] = torch
                new_flags.append("has_torch_stick")
                _recalc_slot()
            else:
                if state.current_room == "cell_01":
                    # Spelaren försöker ta väggfacklan i cellen (den är fastsatt)
                    notes.append("Tried to take the fixed wall torch in the cell; denied.")
                    llm.narration = "The wall torch is fixed in its bracket, and there isn’t a loose torch here. Nothing special happened."
                else:
                    notes.append("Torch is not in this room; pickup ignored.")
                    llm.narration = "You feel around, but there’s no torch here. Nothing special happened."

    if "pickup_keys" in events:
        if holding_keys:
            notes.append("Already holding the keys; pickup ignored.")
        elif slot_occupied:
            notes.append("Hands full; cannot pick up keys while holding another item.")
            llm.narration = "Your hands are full. Drop what you're holding first. Nothing special happened."
        else:
            if keys["location"] == state.current_room:
                keys["location"] = "player"
                state.items["keys"] = keys
                _recalc_slot()

            
            else:
                notes.append("Keys are not in this room; pickup ignored.")
                llm.narration = "You don't find any keys here. Nothing special happened."

    if "pickup_crossbow" in events:
        if holding_crossbow:
            notes.append("Already holding the crossbow; pickup ignored.")
        elif slot_occupied:
            notes.append("Hands full; cannot pick up the crossbow while holding another item.")
            llm.narration = "Your hands are full. Drop what you're holding first. Nothing special happened."
        else:
            in_moat = state.flags_courtyard.get("in_moat", False)
            at_top = state.flags_courtyard.get("at_tower_top", False)
            if state.current_room == "courtyard_01" and in_moat:
                notes.append("Crossbow pickup attempted from the moat; denied.")
                llm.narration = "The crossbow is out of reach from the water. Get onto the grass or the platform first. Nothing special happened."
            elif state.current_room == "courtyard_01" and at_top and crossbow["location"] == "courtyard_tower_top":
                crossbow["location"] = "player"
                state.items["crossbow"] = crossbow
                _recalc_slot()
            elif state.current_room == "courtyard_01" and (not at_top) and (not in_moat) and crossbow["location"] == "courtyard_01":
                crossbow["location"] = "player"
                state.items["crossbow"] = crossbow
                _recalc_slot()
            else:
                notes.append("Crossbow not reachable here; you need to be at the same elevation.")
                llm.narration = "You reach out, but the crossbow is not within reach here. Nothing special happened."


    # 3) Lighting attempts (only in the cell and only if holding an unlit torch)
    if "light_torch" in events:
        holding_torch_now = (state.items["torch"]["location"] == "player")
        if state.current_room != "cell_01":
            notes.append("Ignored 'light_torch' outside Prison Cell.")
            events = [e for e in events if e != "light_torch"]
            if "torch_lit" in llm.flags_set:
                llm.flags_set = [f for f in llm.flags_set if f != "torch_lit"]
            if not holding_torch_now:
                _set_narr("You’re not holding the torch, and you can only light it on the wall torch in the prison cell. Nothing special happened.")
            else:
                _set_narr("You can only light the torch on the wall torch in the prison cell. Nothing special happened.")
        elif not holding_torch_now:
            notes.append("Ignored 'light_torch' without holding the torch.")
            events = [e for e in events if e != "light_torch"]
            _set_narr("You lean toward the wall flame with empty hands; heat licks your knuckles and you flinch back. Nothing special happened.")

            
        elif state.items["torch"].get("lit"):
            notes.append("Torch already lit; ignoring duplicate.")
        else:
            state.items["torch"]["lit"] = True
            notes.append("Torch lit.")
            new_flags.append("torch_lit")
            llm.narration = "You touch the stick to the wall flame; the torch flares to life."

    # Keep flags in sync with items for LLM context and engine logic
    sync_flags_with_items(state)

    prog_norm = (llm.progression or "").strip().lower()

    if state.current_room == "cell_01":
        can_go = "coal_01" in ROOM_GRAPH["cell_01"] and state.flags_cell["stone_moved"]
        wants_go = (
            ("enter_coal_cellar" in events)
            or (prog_norm in {"next_room", "coal_01", "coal", "coal_cellar"})
            or ("entered_hole" in llm.flags_set)
        )
        # NYTT: endast tillåt traversal om spelarens INPUT uttryckte rörelse in i hålet
        input_move_intent = (infer_move_event("cell_01", player_action_text) == "enter_coal_cellar")

        if can_go and wants_go and input_move_intent and not cancel_movement_this_turn:
            room_transition = "coal_01"
        elif wants_go and not input_move_intent:
            # Icke-rörelseinteraktioner med hålet ska inte traversera
            llm.narration = "You peer into the tight opening; coal dust rasps at your nose. You’d have to crawl to go anywhere. Nothing special happened."
        elif wants_go and not can_go:
            narration_tmp = (llm.narration.strip() if llm.narration else "")
            if narration_tmp and narration_tmp[-1] not in ".!?":
                narration_tmp += "."
            denial_line = " The stone still blocks the opening; you can’t squeeze through."
            llm.narration = (narration_tmp + denial_line).strip()
            notes.append("Attempted to enter coal cellar but stone not moved; movement denied.")


    elif state.current_room == "coal_01":
        can_go = "cell_01" in ROOM_GRAPH["coal_01"]
        wants_go = ("return_to_cell" in events) or (prog_norm in {"cell_01", "cell", "prison_cell"})
        if can_go and wants_go and not cancel_movement_this_turn:
            room_transition = "cell_01"
        if cancel_movement_this_turn and wants_go:
            notes.append("Movement canceled due to darkness; you remain where you are.")

        # Hantera källardörren oberoende av uppgångslogik
        if "open_hall_door" in events:
            # Kräv uttrycklig input-intent för att faktiskt öppna dörren
            inferred = infer_move_event("coal_01", player_action_text)
            if inferred != "open_hall_door":
                notes.append("Blocked open_hall_door due to missing explicit input intent.")
                events = [e for e in events if e != "open_hall_door"]
                llm.narration = "Your hand pauses on cold stone—first decide if you’re actually opening the door. Nothing special happened."
            elif torch_light_present_here(state):  # tillåter även tänd fackla nedlagd i rummet
                room_transition = "hall_01"
            else:
                notes.append("Ignored 'open_hall_door' without light present in this room.")
                llm.narration = "You grope toward the far door, but in pitch-black you can't find the handle. Better find light first. Nothing special happened."



    # ---------------- Apply HP ----------------
    prev_hp = state.hp
    state.hp = max(0, min(100, state.hp + enforced_hp_delta))
    if state.hp != prev_hp:
        notes.append(f"HP changed {prev_hp} -> {state.hp} (delta {enforced_hp_delta}).")

    # ---------------- Narration & cues (initial composition) ----------------
    narration = llm.narration.strip() or "Nothing happens. The scene remains as it was. Nothing special happened."

    if state.current_room == "cell_01":
        nl = narration.lower()
        cues: List[str] = []

        # hinta baserat på state-delta (inte new_flags)
        if (not was_found_before) and state.flags_cell.get("found_loose_stone", False):
            if ("loose stone" not in nl) and ("loose cobblestone" not in nl):
                cues.append("You notice a loose cobblestone under the straw.")


        # hål-cue också baserad på delta i state
        if (not was_stone_moved_before) and state.flags_cell.get("stone_moved", False):
            if ("crawlable hole" not in nl) and (" a hole" not in nl and " the hole" not in nl):
                cues.append("You reveal a crawlable hole.")

        # Behåll torchraden om den precis tändes denna tur
        if "torch_lit" in new_flags and ("torch" not in nl or "lit" not in nl):
            cues.append("Your torch catches fire and burns steadily.")

        if cues:
            narration = _re.sub(r'\s*Nothing special happened\.\s*$', '', narration).strip()
            if narration and narration[-1] not in ".!?":
                narration += "."
            narration += " " + " ".join(cues)

        # deterministisk, tydlig rad första gången stenen verkligen lyfts
        if (not was_stone_moved_before) and state.flags_cell.get("stone_moved", False):
            if "straw_rummaged" in events and (not was_found_before):
                narration = "You brush the straw aside and spot a loose stone. You carefully lift it aside, revealing a crawlable hole."
            else:
                narration = "You carefully lift the loose cobblestone aside, revealing a crawlable hole."
    # Om stenen redan var åt sidan före turen och spelaren försöker igen:
    if state.current_room == "cell_01" and was_stone_moved_before and attempted_move_stone_input and not room_transition:
        narration = _re.sub(r'\s*Nothing special happened\.\s*$', '', narration).strip()
        narration = "The loose stone is already aside."

    if state.current_room == "coal_01":
        nl = narration.lower()
        cues: List[str] = []
        if "has_torch_stick" in new_flags and ("wooden torch" not in nl and "torch stick" not in nl and "torch" not in nl):
            cues.append("You pick up a wooden torch.")
        if cues:
            if narration and narration[-1] not in ".!?":
                narration += "."
            narration += " " + " ".join(cues)

    # Ensure guard strike narration line if needed (cell) — no duplicates
    if enforced_hp_delta == -20 and state.current_room == "cell_01":
        nl2 = narration.lower()
        has_guard = ("guard" in nl2)
        has_hit_verb = HIT_VERBS_RE.search(nl2) is not None
        if not (has_guard and has_hit_verb):
            if narration and narration[-1] not in ".!?":
                narration += "."
            narration += " The guard gets annoyed and unlocks the door, strikes you with his fist, locks the door, and then returns to his bench."


    # Guard-scrub i alla rum utom cell_01
    if state.current_room != "cell_01" and "guard_punishes" in events:
        events = [e for e in events if e != "guard_punishes"]
        notes.append("Removed 'guard_punishes' outside the prison cell.")

    # Knight-scrub i rum där riddaren inte finns (cell_01 & coal_01)
    if state.current_room in {"cell_01", "coal_01"}:
        knight_mentioned = ("knight_notice" in events) or ("combat_knock_guard" in events) or _re.search(r"\bknight\b", narration, _re.IGNORECASE)
        if knight_mentioned:
            events = [e for e in events if e not in {"knight_notice", "combat_knock_guard"}]
            other_meaningful = any(e in {
                "straw_rummaged","stone_lifted","enter_coal_cellar","return_to_cell","open_hall_door",
                "pickup_stick","pickup_torch","drop_torch","extinguish_torch","light_torch",
                "drop_keys","pickup_keys",
                "dark_stumble"
            } for e in events)
            if not (state.current_room == "cell_01" and enforced_hp_delta == -20):
                if other_meaningful:
                    if narration and narration[-1] not in ".!?":
                        narration += "."
                    narration += " There’s no knight here."
                else:
                    narration = "There’s no knight here. Nothing special happened."
            notes.append("Removed knight-related narration/events outside the hall.")

    # ---------------- Darkness-specific narrative guards ----------------
    # Deny "using the torch" if not carried/present lit here.
    if state.current_room == "coal_01" and USE_TORCH_RE.search(player_action_text or ""):
        if not (state.items.get("torch", {}).get("location") == "player") and not torch_light_present_here(state):
            if narration and narration[-1] not in ".!?":
                narration += "."
            narration += " You don't have a torch here."

    # Darkness "look" hint (adds a brief safety nudge).
    if state.current_room == "coal_01" and dark_here and LOOK_RE.search(player_action_text or ""):
        if narration and narration[-1] not in ".!?":
            narration += "."
        narration += " Better not move while in darkness—find some light first."

    # Hard darkness clamp — om LLM påstår ljus/detaljer i beckmörker, ersätt men bevara pickups
    if state.current_room == "coal_01" and dark_here and "return_to_cell" not in events and "open_hall_door" not in events:
        if (_TORCHLIGHT_WORDS_RE.search(narration) or
            _TORCHLIGHT_WORDS_RE.search(player_action_text or "") or
            _TORCH_POSSESSION_CLAIM_RE.search(narration) or
            _SEEING_DETAILS_IN_COAL_RE.search(narration)):
            picked = ("pickup_stick" in events) or ("pickup_torch" in events)
            if picked:
                narration = "You pick up a wooden torch. It is pitch-black. Better not move while in darkness—find some light first."
            else:
                narration = "It is pitch-black. Better not move while in darkness—find some light first."
            notes.append("Replaced narration due to light/seeing claims while in total darkness (pickup preserved if present).")


        # Hard light clamp — om fackelljus finns HÄR, men LLM påstår mörker/snubbel, korrigera texten.
    if state.current_room == "coal_01" and torch_light_present_here(state):
        if re.search(r"\b(pitch[- ]?black|total\s+dark(ness)?|can't\s+see|cannot\s+see|blindly|grope|stumble)\b",
                    narration, re.IGNORECASE):
            narration = ("With the light from your torch, the cramped coal heaps come into view; "
                        "at the far end a short staircase leads to a closed door.")
            notes.append("Replaced contradictory darkness narration because torchlight is present in coal_01.")


    # ---------------- Commit room transition ----------------
    if room_transition:
        src = state.current_room
        dst = room_transition

        # Sätt deterministisk färd-narration
        canonical_move_lines = {
            ("cell_01", "coal_01"): "You carefully crawl into the hole and drop into the coal cellar.",
            ("coal_01", "cell_01"): "You climb back up through the crawl opening into the prison cell.",
            ("coal_01", "hall_01"): "You push the far door open and step into the great hall.",
            ("hall_01", "coal_01"): "You slip back through the door into the coal cellar.",
            ("hall_01", "courtyard_01"): "You pull the door open and step into the castle courtyard.",
        }
        move_line = canonical_move_lines.get((src, dst))
        if move_line:
            llm.narration = move_line
            narration = move_line

        notes.append(f"Room transition: {src} -> {dst}")
        state.current_room = dst
        # After moving rooms, sync visibility flag again (light may or may not be present here)
        sync_flags_with_items(state)

    # ---------------- Inventory UI ----------------
    state.inventory = inventory_items_from_items(state)

    # ---------------- Final sanity: if nothing meaningful happened, enforce "Nothing special happened."
    meaningful_events = {
        "drop_torch", "pickup_torch", "pickup_stick", "light_torch", "extinguish_torch",
        "drop_keys", "pickup_keys", "unlock_courtyard_door",
        "drop_crossbow", "pickup_crossbow", "shoot_guard",
        "dark_stumble", "guard_punishes",
        "enter_coal_cellar", "return_to_cell", "return_to_coal", "open_hall_door",
        "straw_rummaged", "stone_lifted",
        "knight_notice", "combat_knock_guard",
        "climb_ladder_up", "climb_ladder_down", "pull_lever", "guards_arrive",
        "cross_gate_bridge", "jump_into_moat", "swim_across"
    }

    something_happened = (
        bool(room_transition) or
        bool(game_won) or
        (state.hp != prev_hp) or
        any(e in events for e in meaningful_events)
    )

    # Om ingen transition skedde men narrationen hävdar att du "är i" ett annat rum: klampa
    if not room_transition:
        claim_hall = _re.search(r"\b(Great Hall)\b", narration, _re.IGNORECASE)
        claim_coal = _re.search(r"\b(Coal Cellar)\b", narration, _re.IGNORECASE)
        claim_cell = _re.search(r"\b(Prison Cell)\b", narration, _re.IGNORECASE)
        claimed = None
        if claim_hall:
            claimed = "hall_01"
        elif claim_coal:
            claimed = "coal_01"
        elif claim_cell:
            claimed = "cell_01"
        if claimed and claimed != state.current_room:
            narration = "You stay where you are. Nothing special happened."

        # Sub-location clamp: tower top claims när man står på gräset
        if (
            state.current_room == "courtyard_01"
            and not state.flags_courtyard.get("at_tower_top", False)
            and not state.flags_courtyard.get("in_moat", False)
            and "jump_into_moat" not in events
            and "climb_ladder_up" not in events
            and "climb_ladder_down" not in events
            and "pull_lever" not in events
        ):
            if re.search(r"\b(platform|tower\s+top|top\s+of\s+the\s+tower)\b", narration, re.IGNORECASE):
                narration = "You are on the grass below the tower. Nothing special happened."

    # Suppress "Nothing special happened." on pure LOOK/describe turns
    look_only = (not something_happened) and (
        LOOK_RE.search(player_action_text or "") is not None
        or SEE_QUERY_RE.search(player_action_text or "") is not None
        or ADVICE_QUERY_RE.search(player_action_text or "") is not None
    )

    # If something meaningful happened but earlier narration injected "Nothing special happened.", strip it
    if something_happened:
        narration = _re.sub(r'\s*Nothing (?:special )?happened\.\s*$', '', narration).strip()
        narration = _re.sub(r'(?<!\S)Nothing (?:special )?happened\.(?:\s+)?', '', narration).strip()

    if not something_happened and not look_only:
        if scrubbed_illegal_events:
            narration = "You stay where you are. Nothing special happened."
        elif not crossbow_nohold:
            if narration and narration[-1] not in ".!?":
                narration += "."
            if "Nothing special happened." not in narration:
                narration += " Nothing special happened."
                    # Normalize boring-outcome wording globally
                narration = narration.replace("Nothing special happened.", NOTHING_LINE)

        # else: crossbow_nohold => exakt fras, ingen "Nothing special happened."

    # ---------------- Result ----------------
    progression = "stay"

    return {
        "narration": narration,
        "applied_noise": noise,
        "applied_hp_delta": enforced_hp_delta,
        "events": events,
        "new_flags": new_flags,
        "progression": progression,
        "notes": notes,
        "cause": cause,
        "room_transition": room_transition,
        "game_won": game_won,
    }





# -----------------------------
# Game loop (Rooms 1–2 prototype)
# -----------------------------

WELCOME_TEXT = (
    "You stand in a dark stone cell. A torch burns low on one wall. An iron door with a barred window faces a torchlit corridor "
    "where a drowsy guard slumps on a bench. He wears full armor; there is no realistic chance to trick him or defeat him. He gets angry if you make too much noise or annoy him."
    "High above, a tiny window reveals a foggy night sky and distant battlements—falling from there would be certain death. "
    "A thin straw bed lies on the floor. There is said to be a loose cobblestone somewhere in the cell—if you can find it.\n\n"
    "Advice: Try to specify your prompt, like 'take the torch' insead of 'take it' for best result. Try to limit yourself to one or two actions per prompt. For exampel: 'Open the door.', or 'Drop the hammer and pick up the flower.' "
    "\n"
)

def print_status(hp: int, noise_level: int, hp_delta: int, inventory: List[str], cause: str = "") -> None:
    if hp_delta != 0 and cause:
        print(f"[HP {hp} ({hp_delta}) — {cause}]")
    elif hp_delta != 0:
        print(f"[HP {hp} ({hp_delta})]")
    else:
        print(f"[HP {hp}]")
    print(f"[Noise this turn: {noise_level}]")
    inv_text = ", ".join(inventory) if inventory else "empty"
    print(f"[Inventory: {inv_text}]")


def game_over_line() -> str:
    return "\n*** GAME OVER ***\n"


def print_room_intro_if_needed(state: GameState) -> None:
    """Print a longer intro only the first time a room is entered."""
    if state.current_room == "coal_01" and not state.flags_coal.get("coal_intro_shown", False):
        if torch_light_present_here(state):
            print("Torchlight reveals cramped coal heaps and, at the far end, a short staircase leading to a closed door.\n")
        else:
            print(COAL_INTRO_TEXT + "\n")
        state.flags_coal["coal_intro_shown"] = True

    if state.current_room == "hall_01" and not state.flags_hall.get("hall_intro_shown", False):
        print(HALL_INTRO_TEXT + "\n")
        state.flags_hall["hall_intro_shown"] = True

    if state.current_room == "courtyard_01" and not state.flags_courtyard.get("courtyard_intro_shown", False):
        print(COURTYARD_INTRO_TEXT + "\n")
        state.flags_courtyard["courtyard_intro_shown"] = True



def print_coal_lit_entry_hint_if_applicable(state: GameState) -> None:
    """When entering the coal cellar with torchlight present, print what is now visible."""
    if state.current_room == "coal_01" and torch_light_present_here(state):
        print("Torchlight pushes back the darkness: heaps of coal crowd the tight stone floor, and at the far end a short staircase leads to a closed door.\n")


def main() -> None:
    dev_log("PROMPT_VERSION: 3.3 (... )")

    print("\n=== ESCAPE THE CASTLE — That is your only mission. ===\n")
    print(WELCOME_TEXT)
    print_room_banner("cell_01")
    print("What do you do?\n")

    state = GameState()
    client = OllamaChat()

    while True:
        try:
            player_action = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye.")
            return

        if not player_action:
            print("(Say what you do.)")
            continue

        if player_action.lower() in {"inventory", "inv"}:
            state.inventory = inventory_items_from_items(state)
            print_status(state.hp, noise_level=0, hp_delta=0, inventory=state.inventory, cause="")
            continue

        if player_action.lower() in {"help", "?"}:
            print("Commands: type what you do in plain English. Useful: 'inventory', 'quit'. Keep quiet in the cell.")
            continue

        if player_action.lower() in {"quit", "exit"}:
            print("You give up. The castle remains your world.")
            print(game_over_line())
            return

        # Build scene card for the current room
        scene_card = SCENES[state.current_room]
        scene_json = json.dumps(scene_card, ensure_ascii=False, indent=2)
        sync_flags_with_items(state)

        flags_cell_str = json.dumps(state.flags_cell, ensure_ascii=False)
        # Sync flags with items before sending to model (for accurate context)
        
        flags_coal_str = json.dumps(state.flags_coal, ensure_ascii=False)
        items_str = json.dumps(state.items, ensure_ascii=False)
        inventory_str = json.dumps(state.inventory, ensure_ascii=False)
        flags_hall_str = json.dumps(state.flags_hall, ensure_ascii=False)
        flags_courtyard_str = json.dumps(state.flags_courtyard, ensure_ascii=False)

        user_prompt = USER_INSTRUCTION_TEMPLATE.format(
            room_id=state.current_room,
            room_title=scene_card.get("title", state.current_room),
            scene_card_json=scene_json,
            hp=state.hp,
            flags_cell=flags_cell_str,
            flags_coal=flags_coal_str,
            flags_hall=flags_hall_str,
            flags_courtyard=flags_courtyard_str,   # <-- ny rad
            items=items_str,
            inventory=inventory_str,
            player_action=player_action
        )


        # LLM call
        try:
            raw = client.chat_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            dev_log(f"LLM_ERROR: {e}")
            print("The torch sputters; your thoughts blur. (LLM error). Try again.")
            continue

        # Coerce and validate
        llm = coerce_llm_result(raw)
        result = validate_and_apply(state, llm, player_action)

        # Dev log details
        dev_log("LLM_PARSED: " + json.dumps(raw, ensure_ascii=False))
        dev_log("ENGINE_NOTES: " + "; ".join(result["notes"]))
        dev_log(f"ENGINE_ROOM: {state.current_room}")
        dev_log(f"ENGINE_FLAGS_CELL: {state.flags_cell}")
        dev_log(f"ENGINE_FLAGS_COAL: {state.flags_coal}")
        dev_log(f"ENGINE_FLAGS_HALL: {state.flags_hall}")
        dev_log(f"ENGINE_FLAGS_COURTYARD: {state.flags_courtyard}")

        dev_log(f"ENGINE_ITEMS: {state.items}")
        dev_log(f"ENGINE_INV: {state.inventory}")
        dev_log(f"ENGINE_HP: {state.hp}")

        # Print narration (model's voice + deterministic cues)
        print("\n" + result["narration"].strip() + "\n")

        # Print concise status (with cause + inventory)
        print_status(state.hp, result["applied_noise"], result["applied_hp_delta"], state.inventory, result.get("cause", ""))

        # Death check
        if state.hp <= 0:
            if not re.search(r"\b(die|dies|dead|death|lifeless|darkness takes you|your last breath)\b",
                             result["narration"], re.IGNORECASE):
                print("\nYour knees buckle as the world narrows to a single, fading point of light.")
            print(game_over_line())
            return

        # Room transition banner + first-time intro + lit entry hint
        if result.get("room_transition"):
            print_room_banner(state.current_room)
            print_room_intro_if_needed(state)
            print_coal_lit_entry_hint_if_applicable(state)

        # Win check (final escape)
        if result.get("game_won"):
            print("\n*** YOU ESCAPED! ***")
            print("You slip beyond the castle walls and vanish into the night.\n")
            return


        # Continue loop


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        sys.stderr.write(
            "\n[ConnectionError] Could not reach Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running (open the Ollama app or run `ollama serve`) and that you've pulled the model:\n"
            "    ollama pull llama3.1:8b\n\n"
        )
        sys.exit(1)
