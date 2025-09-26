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

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CHAT_API = f"{OLLAMA_HOST}/api/chat"
MODEL_NAME = "llama3.1:8b"

TEMPERATURE = 0.2
TOP_P = 0.9

DEV_LOG_PATH = "game_dev.log"


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
    # location: "player" | "cell_01" | "coal_01"
    items: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "torch": {"location": "coal_01", "lit": False}  # starts as a wooden torch stick on the coal cellar floor
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
        "rounded cobblestones about 20–30 centimeters across. A single torch flickers on one wall. "
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
        # cross-room flags that may change here:
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
        "Any careful inspection/searching/looking under/moving/rummaging/removing of the straw MUST be recorded as event 'straw_rummaged' and MUST reveal the loose stone if not already (set flag: found_loose_stone). This includes actions that mention 'straw' or the 'straw bed'.",
        "If the player lifts/moves/pries/drags the loose stone (once discovered), set flag: stone_moved and reveal a crawlable hole.",
        "If the player crawls into/goes through the hole (once available), set flag: entered_hole and add event 'enter_coal_cellar'.",
        # Lighting rule (only here):
        "If the player holds a wooden torch (has_torch_stick) and tries to light it on the wall torch, add event 'light_torch' and set flag 'torch_lit' (the wall torch itself cannot be removed).",
        # Inventory (torch):
        "If the player drops/throws/places their torch here, add event 'drop_torch'. If they pick it up, add event 'pickup_torch'.",
        "If an action is impossible, produce a playful grounded refusal and end with 'Nothing happened.'",
        "Never invent items/exits beyond the scene; never escape via the high window."
    ],
    "style_and_tone": [
        "Second person, immersive, concise, vivid.",
        "Maximum ~3 sentences per turn.",
        "Announce each key outcome exactly once; do not repeat the same fact.",
        "If nothing meaningful happens, end with 'Nothing happened.'"
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
        # Darkness & movement
        "If the player attempts to walk/explore/move deeper while in darkness (no 'torch_lit'), they immediately stumble in the dark: set event 'dark_stumble' and hp_delta -10. Warn them it's unwise to move without light and that they remain where they are.",
        "If 'torch_lit' is true, never set 'dark_stumble'; movement is safe and surroundings are visible.",
        # Ground object
        "If the player feels/picks up the unknown object at their feet, reveal it as a wooden torch stick: set flag 'has_torch_stick' and add event 'pickup_stick' (or 'pickup_torch').",
        # Returning to cell
        "If the player intends to return through the crawl opening, add event 'return_to_cell'.",
        # Inventory (torch)
        "If the player drops/throws/places their torch here, add event 'drop_torch'. If they pick it up, add event 'pickup_torch'.",
        # Visibility-gated exploration
        "If 'torch_lit' is true, you may describe coal heaps, the cramped stone floor, and a short stone staircase leading to a door at the far end.",
        # Door interaction (win for now)
        "If the player opens the door at the far end (requires being able to find it; typically with 'torch_lit'), add event 'open_hall_door'.",
        # Safety
        "If an action is impossible, produce a playful grounded refusal and end with 'Nothing happened.'",
        "Never invent items or exits not stated."
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
}

# Static room graph (engine controls legal travel)
ROOM_GRAPH: Dict[str, List[str]] = {
    "cell_01": ["coal_01"],   # via crawlable hole
    "coal_01": ["cell_01"],
}


# -----------------------------
# Prompt templates
# -----------------------------

SYSTEM_PROMPT = """You are the Game Master for a grounded, text-based escape adventure.
Follow the CURRENT ROOM's scene card and the rules below. Respond in VALID JSON only.

Global rules:
- English only. Interpret player intent semantically; no verb lists.
- Evaluate noise_level (0–3) using the current room's scale; resets next turn.
- Guard punishment applies ONLY in cell_01: if noise_level >= 2 this turn, the guard MUST strike once (-20 HP) and you MUST narrate it. In other rooms, ignore guard/noise punishment.
- Hard rule (straw in cell_01): If the player's action mentions the straw or the straw bed and the loose stone isn't discovered yet, you MUST add 'straw_rummaged' to events AND 'found_loose_stone' to flags_set this turn.
- Straw-only rule: When the player only interacts with the straw (inspect/move/remove/search), you MUST NOT set 'stone_moved'. Moving the stone requires an explicit attempt to lift/pry/drag it.
- Multi-action rule: If the player does both in one prompt (e.g., “clear the straw AND lift the stone”), include events 'straw_rummaged' AND 'stone_lifted', and set flags 'found_loose_stone' AND 'stone_moved' in the same turn.
- Never invent items, tools, magic, or exits beyond the active scene card.
- Narration: immersive 2nd person, soft limit ~3 sentences, announce each key outcome ONCE (no repetition).
- If the action is impossible, reply briefly and end with "Nothing happened."
- If nothing meaningful changes, end narration with "Nothing happened."
- When an event happens, set relevant flags in "flags_set". Use semantic events for traversal or special effects (e.g., 'enter_coal_cellar','return_to_cell','dark_stumble','open_hall_door','light_torch','pickup_stick','pickup_torch','drop_torch').

Coal cellar specific:
- If the player tries to move around without a lit torch, you MUST set event 'dark_stumble' and hp_delta -10, warn them, and say they remain where they are.
- If 'torch_lit' is true, you MUST NOT set 'dark_stumble'. Movement is safe and you should describe visible surroundings instead.
- The unknown ground object is a wooden torch stick; only reveal that identity on pickup (set 'has_torch_stick'; using event 'pickup_stick' or 'pickup_torch').
- Lighting the torch is ONLY possible in the Prison Cell by using the wall torch; if attempted there and the player has the stick, set 'torch_lit' (event 'light_torch').
- With 'torch_lit' true, you may reveal the staircase and the door; opening the door sets event 'open_hall_door'.

Inventory protocol:
- The player can carry ONLY ONE item. If they try to pick up an item while already carrying one, you MUST refuse and suggest dropping the current item. End with "Nothing happened."
- If the player drops/throws/places their torch, you MUST add event 'drop_torch' and treat it as no longer in their inventory (but it remains in the current room).

Return a single JSON object with this schema:
{
  "narration": string,
  "noise_level": integer,
  "hp_delta": integer,
  "events": string[],
  "flags_set": string[],
  "progression": string,
  "safety_reason": string
}
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
- Items: {items}
- Inventory (one slot): {inventory}

REMINDERS:
- Prison Cell only: noise >= 2 ⇒ guard unlocks, strikes (-20 HP), returns (must narrate).
- Traversal (MANDATORY): use 'enter_coal_cellar' when crawling down; use 'return_to_cell' when crawling back up.
- Straw in the Prison Cell: searching/looking under/moving reveals the loose stone if not already.
- Straw-only: do NOT set 'stone_moved' from straw actions alone.
- To move the stone, the player must explicitly try to lift/pry/drag it; when you do that, include event 'stone_lifted'.
- Inventory: only ONE item carried. To pick up another, the player must drop the current one (event 'drop_torch' for torch).
- In the Coal Cellar: without a lit torch, moving around causes 'dark_stumble' (-10 HP) and you remain where you are. With a lit torch, do not describe darkness; describe visible details (coal heaps, cramped stone floor, short staircase to a door).
- Lighting the torch works ONLY in the Prison Cell using the wall torch (if you hold the stick).

PLAYER ACTION:
{player_action}

IMPORTANT:
- Output VALID JSON ONLY matching the schema above. No backticks, no code fences, no commentary.
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
        data = resp.json()

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
            data2 = resp2.json()
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

ALLOWED_CELL_FLAGS = {"found_loose_stone", "stone_moved", "entered_hole"}
ALLOWED_COAL_FLAGS = {"has_torch_stick", "torch_lit"}
ALLOWED_EVENTS = {
    "hay_moved", "straw_rummaged", "stone_revealed", "stone_lifted",
    "enter_coal_cellar", "return_to_cell",
    "dark_stumble", "pickup_stick", "pickup_torch", "drop_torch", "light_torch",
    "open_hall_door",
    "guard_punishes"
}

# Robust hit-verb detection to avoid duplicate guard line
HIT_VERBS_RE = re.compile(
    r"\b(strike|strikes|struck|striking|hit|hits|hitting|"
    r"smack|smacks|smacked|punch|punches|punched|"
    r"kick|kicks|kicked|club|clubs|clubbed)\b",
    re.IGNORECASE
)


def inventory_items_from_items(state: GameState) -> List[str]:
    """UI text derived from world items."""
    torch = state.items.get("torch", {})
    if torch.get("location") == "player":
        return ["lit torch" if torch.get("lit") else "wooden torch"]
    return []


def torch_light_present_here(state: GameState) -> bool:
    """Is there torchlight in the current room (carried or placed lit here)?"""
    torch = state.items.get("torch", {})
    if not torch.get("lit"):
        return False
    loc = torch.get("location")
    return loc == "player" or loc == state.current_room


def sync_flags_with_items(state: GameState) -> None:
    """Keep legacy flags in sync for LLM context."""
    torch = state.items.get("torch", {})
    holding = torch.get("location") == "player"
    state.flags_coal["has_torch_stick"] = (holding and not torch.get("lit", False))
    # torch_lit reflects light in *current room* (so the model doesn't assume light elsewhere)
    state.flags_coal["torch_lit"] = torch_light_present_here(state)


def coerce_llm_result(raw: Dict[str, Any]) -> LLMResult:
    narration = str(raw.get("narration", "")).strip()
    noise_level = int(raw.get("noise_level", 0))
    hp_delta = int(raw.get("hp_delta", 0))
    events = list(raw.get("events", []) or [])
    flags_set = list(raw.get("flags_set", []) or [])
    progression = str(raw.get("progression", "stay")).strip() or "stay"
    safety_reason = str(raw.get("safety_reason", "")).strip()
    return LLMResult(narration, noise_level, hp_delta, events, flags_set, progression, safety_reason)


def print_room_banner(room_id: str) -> None:
    if room_id == "cell_01":
        print("\n— You are in the Prison Cell. —\n")
    elif room_id == "coal_01":
        print("\n— You are in the Coal Cellar. —\n")
    else:
        print(f"\n— You are in: {room_id} —\n")


COAL_INTRO_TEXT = (
    "It’s pitch-black. Coal dust clings to your throat; the stone floor feels uneven underfoot. "
    "Heaps of coal hem you in on both sides. As you steady yourself, your foot bumps against something lying on the ground. "
    "You can feel it with your toes, but you can’t tell what it is until you pick it up."
)


def validate_and_apply(state: GameState, llm: LLMResult) -> Dict[str, Any]:
    """
    Apply deterministic policies for the current room; validate allowed state changes; handle room transitions and victory.
    """
    notes: List[str] = []

    # Bound noise level
    noise = max(0, min(3, int(llm.noise_level)))
    if noise != llm.noise_level:
        notes.append(f"Noise coerced from {llm.noise_level} to {noise}.")

    # Events sanitized
    events = [e for e in llm.events if e in ALLOWED_EVENTS]

    # ---------------- HP / Punishments ----------------
    enforced_hp_delta = 0
    cause = ""

    if state.current_room == "cell_01":
        # Guard punishment
        if noise >= 2:
            enforced_hp_delta = -20
            cause = "Guard strikes you for making too much noise"
            if llm.hp_delta != -20:
                notes.append(f"HP delta overridden to -20 due to guard punishment (noise={noise}) in cell.")
            if "guard_punishes" not in events:
                events.append("guard_punishes")
        else:
            if llm.hp_delta != 0:
                notes.append(f"Ignored non-guard hp_delta={llm.hp_delta} (noise={noise}) in cell.")

    elif state.current_room == "coal_01":
        # Darkness rule — but only if there is *no* light in this room
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

    # ---------------- Derive flags from events ----------------
    if state.current_room == "cell_01":
        event_to_flag = {
            "stone_moved": "stone_moved",
            "stone_revealed": "found_loose_stone",
            "straw_rummaged": "found_loose_stone",
            "enter_coal_cellar": "entered_hole",
            "light_torch": "torch_lit",
        }
    else:
        event_to_flag = {
            "pickup_stick": "has_torch_stick",
            "pickup_torch": "has_torch_stick",
            "light_torch": "torch_lit",
        }
    for e in list(events or []):
        f = event_to_flag.get(e)
        if f and f not in llm.flags_set:
            llm.flags_set.append(f)

    # ---------------- Apply cell flags (ordered) ----------------
    new_flags: List[str] = []

    if state.current_room == "cell_01":
        if "found_loose_stone" in llm.flags_set and not state.flags_cell["found_loose_stone"]:
            state.flags_cell["found_loose_stone"] = True
            new_flags.append("found_loose_stone")

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
    holding_torch = (torch["location"] == "player")

    # Pickup attempts
    if ("pickup_stick" in events) or ("pickup_torch" in events) or ("has_torch_stick" in llm.flags_set):
        if holding_torch:
            notes.append("Already holding the torch; pickup ignored.")
        else:
            if torch["location"] == state.current_room:
                # Enforce one-slot inventory (only torch exists for now, so slot is free)
                torch["location"] = "player"
                state.items["torch"] = torch
                new_flags.append("has_torch_stick")
            else:
                notes.append("Torch is not in this room; pickup ignored.")

    # Drop attempts
    if "drop_torch" in events:
        if holding_torch:
            torch["location"] = state.current_room
            state.items["torch"] = torch
        else:
            notes.append("Tried to drop torch while not holding it; ignored.")

    # Lighting attempts (only in the cell and only if holding an unlit torch)
    if "light_torch" in events:
        if state.current_room != "cell_01":
            notes.append("Ignored 'light_torch' outside Prison Cell.")
        elif not holding_torch:
            notes.append("Ignored 'light_torch' without holding the torch.")
        elif torch.get("lit"):
            notes.append("Torch already lit; ignoring duplicate.")
        else:
            torch["lit"] = True
            state.items["torch"] = torch
            new_flags.append("torch_lit")

    # Keep flags in sync with items for LLM context and engine logic
    sync_flags_with_items(state)

    # ---------------- Room transitions (engine-gated) ----------------
    room_transition: Optional[str] = None
    game_won: bool = False

    prog_norm = (llm.progression or "").strip().lower()

    if state.current_room == "cell_01":
        can_go = "coal_01" in ROOM_GRAPH["cell_01"] and state.flags_cell["stone_moved"]
        wants_go = ("enter_coal_cellar" in events) or (prog_norm in {"next_room", "coal_01", "coal", "coal_cellar"})
        if can_go and wants_go:
            room_transition = "coal_01"
    elif state.current_room == "coal_01":
        can_go = "cell_01" in ROOM_GRAPH["coal_01"]
        wants_go = ("return_to_cell" in events) or (prog_norm in {"cell_01", "cell", "prison_cell"})
        if can_go and wants_go:
            room_transition = "cell_01"

        if "open_hall_door" in events:
            if torch_light_present_here(state):
                game_won = True
            else:
                notes.append("Ignored 'open_hall_door' without light present in this room.")

    # ---------------- Apply HP ----------------
    prev_hp = state.hp
    state.hp = max(0, min(100, state.hp + enforced_hp_delta))
    if state.hp != prev_hp:
        notes.append(f"HP changed {prev_hp} -> {state.hp} (delta {enforced_hp_delta}).")

    # ---------------- Narration & cues ----------------
    narration = llm.narration.strip() or "Nothing happens. The scene remains as it was. Nothing happened."

    if state.current_room == "cell_01":
        nl = narration.lower()
        cues: List[str] = []
        if "found_loose_stone" in new_flags and ("loose stone" not in nl) and ("loose cobblestone" not in nl):
            cues.append("You notice a loose stone beneath the straw bed.")
        if "stone_moved" in new_flags and ("crawlable hole" not in nl) and (" a hole" not in nl and " the hole" not in nl):
            cues.append("You reveal a crawlable hole.")
        if "torch_lit" in new_flags and ("torch" not in nl or "lit" not in nl):
            cues.append("Your torch catches fire and burns steadily.")
        if cues:
            if narration and narration[-1] not in ".!?":
                narration += "."
            narration += " " + " ".join(cues)

    if state.current_room == "coal_01":
        nl = narration.lower()
        cues: List[str] = []
        # If you just picked it up and model didn't narrate the identity
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
            narration += " The guard unlocks the door, strikes you, then returns to his bench."

    # ---------------- Commit room transition ----------------
    if room_transition:
        notes.append(f"Room transition: {state.current_room} -> {room_transition}")
        state.current_room = room_transition
        # After moving rooms, sync visibility flag again (light may or may not be present here)
        sync_flags_with_items(state)

    # ---------------- Inventory UI ----------------
    state.inventory = inventory_items_from_items(state)

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
    "where a drowsy guard slumps on a bench. He wears full armor; there is no realistic chance to trick him or defeat him. "
    "High above, a tiny window reveals a foggy night sky and distant battlements—falling from there would be certain death. "
    "A thin straw bed lies on the floor. There is said to be a loose cobblestone somewhere in the cell - if you can find it, that'll say.\n\n"
    "Advice: Keep quiet. If you make noticeable noise in the cell, the guard will wake, unlock the door, and strike you before returning to his post."
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
    """Print a longer intro only the first time a room is entered (coal cellar for now)."""
    if state.current_room == "coal_01" and not state.flags_coal.get("coal_intro_shown", False):
        print(COAL_INTRO_TEXT + "\n")
        state.flags_coal["coal_intro_shown"] = True


def print_coal_lit_entry_hint_if_applicable(state: GameState) -> None:
    """When entering the coal cellar with torchlight present, print what is now visible."""
    if state.current_room == "coal_01" and torch_light_present_here(state):
        print("Torchlight pushes back the darkness: heaps of coal crowd the tight stone floor, and at the far end a short staircase leads to a closed door.\n")


def main() -> None:
    dev_log("PROMPT_VERSION: 3.0 (room graph, item system, single-slot inventory, per-room light, guard dedupe)")
    print("\n=== ESCAPE THE CASTLE — ROOMS 1–2 (Prototype Stage 2) ===\n")
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
            # Quick inventory peek without LLM call
            state.inventory = inventory_items_from_items(state)
            print_status(state.hp, noise_level=0, hp_delta=0, inventory=state.inventory, cause="")
            continue

        if player_action.lower() in {"quit", "exit"}:
            print("You give up. The castle remains your world.")
            print(game_over_line())
            return

        # Build scene card for the current room
        scene_card = SCENES[state.current_room]
        scene_json = json.dumps(scene_card, ensure_ascii=False, indent=2)
        flags_cell_str = json.dumps(state.flags_cell, ensure_ascii=False)
        # Sync flags with items before sending to model (for accurate context)
        sync_flags_with_items(state)
        flags_coal_str = json.dumps(state.flags_coal, ensure_ascii=False)
        items_str = json.dumps(state.items, ensure_ascii=False)
        inventory_str = json.dumps(state.inventory, ensure_ascii=False)

        user_prompt = USER_INSTRUCTION_TEMPLATE.format(
            room_id=state.current_room,
            room_title=scene_card.get("title", state.current_room),
            scene_card_json=scene_json,
            hp=state.hp,
            flags_cell=flags_cell_str,
            flags_coal=flags_coal_str,
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
        result = validate_and_apply(state, llm)

        # Dev log details
        dev_log("LLM_PARSED: " + json.dumps(raw, ensure_ascii=False))
        dev_log("ENGINE_NOTES: " + "; ".join(result["notes"]))
        dev_log(f"ENGINE_ROOM: {state.current_room}")
        dev_log(f"ENGINE_FLAGS_CELL: {state.flags_cell}")
        dev_log(f"ENGINE_FLAGS_COAL: {state.flags_coal}")
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

        # Win check
        if result.get("game_won"):
            print("\nYou push the door open and step into the great hall. Fresh air washes over the coal dust on your skin.")
            print("*** YOU ESCAPED THE COAL CELLAR! (Next room coming soon) ***")
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
