# ui_server.py
# Flask-baserad web-UI för Escape the Castle med CRT-look.
# ► Endast följande art-filer används (lägg i static/art/):
#   cell.png
#   climb_ladder_down.png
#   climbing_up_tower.png
#   cross_gate_bridge.png
#   dark_stumble.png
#   drop_crossbow.png
#   pickup_crossbow.png
#   drop_keys.png
#   pickup_keys.png
#   enter_coal_cellar.png
#   fight_knight_with_hands.png
#   fight_knight_with_torch.png
#   guard_punishes_you_in_prison_cell.png
#   guards_arrive_when_pulling_lever.png
#   hall.png
#   jump_into_moat.png
#   just_unlocked_courtyard_door.png
#   movestoneaside.png
#   nothing_happened.png
#   opening_door_from_coal_cellar_to_hall.png
#   return_to_cell.png
#   setting_fire_to_torch_in_prison_cell.png
#   shooting_crossbow_from_tower.png
#   straw_rummaged.png

from __future__ import annotations

import json
import os
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# Importera motor och resurser från ditt spel
from escape_castle import (
    GameState,
    OllamaChat,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_TEMPLATE,
    SCENES,
    sync_flags_with_items,
    coerce_llm_result,
    validate_and_apply,
    inventory_items_from_items,
    torch_light_present_here,
    WELCOME_TEXT,
    COAL_INTRO_TEXT,
    HALL_INTRO_TEXT,
    COURTYARD_INTRO_TEXT,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- Enkel global spelsession ----------
STATE = GameState()
CLIENT = OllamaChat()
SESSION_STARTED = False  # visa välkomsttext en gång

ROOM_TITLES = {
    "cell_01": "Prison Cell",
    "coal_01": "Coal Cellar",
    "hall_01": "Great Hall",
    "courtyard_01": "Castle Courtyard",
}

# Basbilder per rum (endast tillåtna filer)
# OBS: vi har INGEN default för courtyard/coal (för att inte använda bannade filer).
ART_DEFAULT = {
    "cell_01": "cell.png",
    "hall_01": "hall.png",
}

# Hjälpare: finns en art-fil?
def _exists_art(filename: str) -> bool:
    return os.path.isfile(os.path.join(app.static_folder, "art", filename))

# Välj första befintliga fil i en lista kandidatnamn
def _first_existing(*names: str) -> str | None:
    for n in names:
        if n and _exists_art(n):
            return n
    return None

# Mapping för varje event → exakt fil (endast dina filer)
EVENT_ART = {
    # Cell
    "straw_rummaged":       ["straw_rummaged.png"],
    "stone_lifted":         ["movestoneaside.png"],  # ← stavning fixad
    "enter_coal_cellar":    ["enter_coal_cellar.png"],  # visas också som kontext i mörk källare
    "return_to_cell":       ["return_to_cell.png"],

    # Fackla (enbart 'light_torch' har bild enligt listan)
    "light_torch":          ["setting_fire_to_torch_in_prison_cell.png"],

    # Källardörr (till hall)
    "open_hall_door":       ["opening_door_from_coal_cellar_to_hall.png"],

    # Hall & nycklar
    "pickup_keys":          ["pickup_keys.png"],
    "drop_keys":            ["drop_keys.png"],
    "unlock_courtyard_door":["just_unlocked_courtyard_door.png"],

    # Cell-straff
    "guard_punishes":       ["guard_punishes_you_in_prison_cell.png"],

    # Hall-strid (specialfall nedan väljer hands/torch-bild)
    "knight_notice":        [],  # styrs i speciallogik
    "combat_knock_guard":   [],  # styrs i speciallogik

    # Gården
    "climb_ladder_up":      ["climbing_up_tower.png"],
    "climb_ladder_down":    ["climb_ladder_down.png"],
    "pull_lever":           ["guards_arrive_when_pulling_lever.png"],
    "guards_arrive":        ["guards_arrive_when_pulling_lever.png"],
    "pickup_crossbow":      ["pickup_crossbow.png"],
    "drop_crossbow":        ["drop_crossbow.png"],
    # 'shoot_guard': endast bild om man är uppe i tornet (speciallogik)
    "shoot_guard":          [],

    "cross_gate_bridge":    ["cross_gate_bridge.png"],  # vinst
    "jump_into_moat":       ["jump_into_moat.png"],

    # Övrigt / säkerhet
    "dark_stumble":         ["dark_stumble.png"],
}

# Rumskontext-bilder (när inget specifikt event tog över)
# Endast dina filer: cell.png / hall.png som bas, samt:
# - enter_coal_cellar.png när man är i coal cellar och det är mörkt
# - opening_door_from_coal_cellar_to_hall.png när man är i coal cellar med ljus/fackla
CONTEXT_ART = {
    "cell_01": [
        (lambda s: True, ["cell.png"]),
    ],
    "coal_01": [
        (lambda s: not s.flags_coal.get("torch_lit", False),
         ["enter_coal_cellar.png"]),
        (lambda s: s.flags_coal.get("torch_lit", False),
         ["opening_door_from_coal_cellar_to_hall.png"]),
    ],
    "hall_01": [
        (lambda s: True, ["hall.png"]),
    ],
    # Ingen generell courtyard-bild i din lista → bara event-bilder där
    "courtyard_01": []
}

def pick_art_filename(state: GameState, result: dict, narration: str) -> str | None:
    """
    Välj rätt bild för denna tick utifrån:
    1) Vinst (cross_gate_bridge.png),
    2) "Nothing special happened." om inget event,
    3) Specialfall (hall-strid & tower-shoot),
    4) Event-prioritet,
    5) Rumskontext,
    6) Fallback: ART_DEFAULT för cell/hall, annars None (behåll senaste bild).
    """
    events = list(result.get("events", []))
    room   = state.current_room

        # --- COAL WITH LIGHT OVERRIDE ---
    # If we are in the coal cellar and there is torchlight present here,
    # always show the "coal-with-torch" image (character with torch in cellar).
    if room == "coal_01" and torch_light_present_here(state):
        art = _first_existing("opening_door_from_coal_cellar_to_hall.png")
        if art:
            return art
        
            # --- JUST ENTERED HALL OVERRIDE ---
    # On room transition into the hall, show the hall scene (not the door-from-coal image).
    if result.get("room_transition") == "hall_01":
        art = _first_existing("hall.png")
        if art:
            return art



    # 1) Victory
    if result.get("game_won"):
        art = _first_existing("cross_gate_bridge.png")
        if art: return art

    # 2) "Nothing special happened." – om inga events
    # (motorn normaliserar till "Nothing special happened.")
    if (("Nothing special happened." in (narration or "")) or ("Nothing special happened" in (narration or ""))) and not events:
        art = _first_existing("nothing_happened.png")
        if art:
            return art

    # 3) Specialfall
    # 3a) Hall-strid: välj rätt variant beroende på om torch hålls i handen
    if any(ev in events for ev in ("knight_notice", "combat_knock_guard")):
        torch_in_hand = (state.items.get("torch", {}).get("location") == "player")
        art = _first_existing("fight_knight_with_torch.png" if torch_in_hand else "fight_knight_with_hands.png")
        if art: return art

    # 3b) Skjuta: visa bara tower-varianten om man står på plattformen
    if "shoot_guard" in events and state.flags_courtyard.get("at_tower_top", False):
        art = _first_existing("shooting_crossbow_from_tower.png")
        if art: return art
    # Skjuter man från gräset finns ingen tillåten bild → inget art-byte här

    # 4) Event-prioritet (endast de events som har explicit art-lista)
    PRIORITY = [
        # Courtyard vinst/hopp/lever/vakter/stege/vapen
        "cross_gate_bridge", "jump_into_moat",
        "guards_arrive", "pull_lever",
        "climb_ladder_up", "climb_ladder_down",
        "pickup_crossbow", "drop_crossbow",

        # Hall
        "unlock_courtyard_door", "pickup_keys", "drop_keys",

        # Coal/Cell
        "open_hall_door", "light_torch",
        "stone_lifted", "straw_rummaged",
        "enter_coal_cellar", "return_to_cell",

        # Safety/damage
        "guard_punishes", "dark_stumble",
    ]
    for ev in PRIORITY:
        if ev in events:
            art = _first_existing(*EVENT_ART.get(ev, []))
            if art:
                return art

    # 5) Rumskontext (t.ex. cell/hall bas, kolkällare mörk/ljus)
    for predicate, candidates in CONTEXT_ART.get(room, []):
        try:
            if predicate(state):
                art = _first_existing(*candidates)
                if art:
                    return art
        except Exception:
            pass

    # 6) Fallback — endast cell/hall har default enligt din lista.
    return ART_DEFAULT.get(room, None)


def append_room_entry_text(state: GameState) -> str:
    """Banner + intro + hint (coal-ljus) för aktuellt rum, samt sätt motsv. flaggor."""
    room = state.current_room
    title = ROOM_TITLES.get(room, room)
    out = []
    out.append(f"\n— You are in the {title}. —\n")

    if room == "coal_01":
        if not state.flags_coal.get("coal_intro_shown", False):
            out.append(COAL_INTRO_TEXT + "\n")
            state.flags_coal["coal_intro_shown"] = True
        if torch_light_present_here(state):
            out.append("Torchlight pushes back the darkness: heaps of coal crowd the tight stone floor, and at the far end a short staircase leads to a closed door.\n")

    elif room == "hall_01":
        if not state.flags_hall.get("hall_intro_shown", False):
            out.append(HALL_INTRO_TEXT + "\n")
            state.flags_hall["hall_intro_shown"] = True

    elif room == "courtyard_01":
        if not state.flags_courtyard.get("courtyard_intro_shown", False):
            out.append(COURTYARD_INTRO_TEXT + "\n")
            state.flags_courtyard["courtyard_intro_shown"] = True

    return "".join(out)


# ---------- HTML (inline) ----------
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Escape the Castle — CRT UI</title>
<style>
  :root{ --bg:#0b0f0b; --phosphor:#9cff57; --accent:#b6ff7b; }
  *{ box-sizing:border-box; }
  html,body{ height:100%; margin:0; background:#000; }
  body{ display:flex; align-items:center; justify-content:center; padding:12px; }
  .shell{ position:relative; width:min(1200px, 96vw); height:min(720px, 92vh);
    background: radial-gradient(120% 140% at 50% 50%, #0a0a0a 0%, #000 70%, #000 100%);
    border-radius:22px; box-shadow: 0 8px 40px rgba(0,0,0,.65), inset 0 0 140px rgba(0,0,0,.9); overflow:hidden;}
  .crt{ position:absolute; inset:16px; border-radius:14px; overflow:hidden;
    background: linear-gradient(180deg, rgba(12,18,12,.65), rgba(12,18,12,.65));
    filter: saturate(0.9) contrast(1.02); transform: perspective(1100px) translateZ(0) scale(1.002);}
  .crt::before{ content:""; position:absolute; inset:0; pointer-events:none;
    background: repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px,
      rgba(0,0,0,.16) 3px, rgba(0,0,0,.16) 4px); mix-blend-mode:multiply;}
  .crt::after{ content:""; position:absolute; inset:-1px; pointer-events:none;
    background: radial-gradient(120% 180% at 50% 50%, rgba(255,255,255,.05), rgba(255,255,255,0) 60%);
    animation: flicker 6.5s infinite, glow 7.2s infinite;}
  @keyframes flicker { 0%,94%,100%{opacity:.06;} 96%{opacity:.11;} 98%{opacity:.03;} }
  @keyframes glow { 0%,100%{filter:brightness(1.00);} 45%{filter:brightness(1.02);} 70%{filter:brightness(0.98);} }

  .screen{ position:relative; display:grid; grid-template-columns: 1.4fr 0.9fr; gap:14px;
    padding:16px; width:100%; height:100%; color: var(--phosphor);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    text-shadow: 0 0 6px rgba(156,255,87,.25); }
  .left{ display:flex; flex-direction:column; min-width:0; overflow:hidden;
    background: linear-gradient(180deg, rgba(0,0,0,.35), rgba(0,0,0,.35)),
      repeating-linear-gradient(90deg, rgba(10,20,10,.16) 0 1px, transparent 1px 2px);
    border:1px solid rgba(156,255,87,.15); border-radius:12px; box-shadow: inset 0 0 18px rgba(0,0,0,.65);}
  .header{ padding:10px 12px; border-bottom:1px solid rgba(156,255,87,.14); font-weight:700; letter-spacing:.5px; }
  .log{ flex:1; overflow:auto; padding:12px; line-height:1.35; white-space:pre-wrap;
        scrollbar-color: rgba(156,255,87,.35) rgba(0,0,0,.2); scrollbar-width: thin; }
  .log .user{ color: var(--accent); }
  .line{ margin: 0 0 4px 0; }
  .glowline{ text-shadow: 0 0 8px rgba(156,255,87,.85), 0 0 18px rgba(156,255,87,.45), 0 0 32px rgba(156,255,87,.25); animation: pulse 2.1s ease-in-out 2;}
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.85} }
  .status{ display:flex; gap:12px; flex-wrap:wrap; padding:8px 12px 0 12px; font-size:13px; opacity:.9; }
  .badge{ padding:2px 8px; border-radius:999px; border:1px solid rgba(156,255,87,.25); background: rgba(0,0,0,.35); }
  .inputbar{ position: sticky; bottom: 0; padding:10px; display:flex; gap:8px; align-items:flex-end;
    background: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.75));
    border-top:1px solid rgba(156,255,87,.14);}
  .inputbar textarea{ flex:1; min-height:42px; max-height:120px; overflow:auto; resize:vertical;
    padding:10px 10px 10px 12px; font:inherit; color:var(--phosphor); background: rgba(0,0,0,.45);
    border:1px solid rgba(156,255,87,.22); border-radius:8px; outline:none; box-shadow: inset 0 0 10px rgba(0,0,0,.55);}
  .inputbar button{ padding:10px 14px; color:#051; background: linear-gradient(180deg, #baff7a, #8de354);
    border:none; border-radius:8px; font-weight:700; cursor:pointer; box-shadow: 0 2px 0 #5ca534, 0 6px 14px rgba(0,0,0,.25); }
  .inputbar button:active{ transform: translateY(1px); box-shadow: 0 1px 0 #5ca534, 0 4px 10px rgba(0,0,0,.3); }

  .right{ min-width:0; display:flex; flex-direction:column; overflow:hidden;
    background: radial-gradient(140% 140% at 50% 50%, rgba(10,20,10,.5), rgba(0,0,0,.7));
    border:1px solid rgba(156,255,87,.15); border-radius:12px; box-shadow: inset 0 0 18px rgba(0,0,0,.65); }
  .imgwrap{ flex:1; display:flex; align-items:center; justify-content:center; padding: 12px; }
  .imgwrap img{ width:100%; height:100%; object-fit:contain; image-rendering: pixelated; filter: contrast(1.05) saturate(0.9) brightness(0.92); }
  .right .caption{ padding:10px 12px; font-size:18px; font-weight:800; letter-spacing:.8px; text-align:center; border-top:1px solid rgba(156,255,87,.14); }

  
    /* Victory overlay */
  .overlay-victory{
    position:absolute; inset:16px;
    display:none; align-items:center; justify-content:center; text-align:center;
    background: radial-gradient(120% 140% at 50% 50%, rgba(0,0,0,.0), rgba(0,0,0,.65));
    z-index: 10;
  }
  .overlay-victory .box{
    padding: 24px 28px;
    border:2px solid rgba(156,255,87,.45);
    border-radius:14px;
    background: rgba(0,0,0,.55);
    box-shadow: 0 0 24px rgba(156,255,87,.25), inset 0 0 18px rgba(0,0,0,.6);
  }
  .overlay-victory h1{
    margin:0 0 8px 0;
    font-size:36px; letter-spacing:1.2px; color:var(--accent);
    text-shadow: 0 0 12px rgba(156,255,87,.7), 0 0 32px rgba(156,255,87,.35);
  }
  .overlay-victory p{
    margin:0; opacity:.9;
  }
  .overlay-victory.show{ display:flex; }





  @media (max-width: 900px){
    .screen{ grid-template-columns: 1fr; }
    .right{ order:-1; height: 36vh; }
  }
</style>
</head>
<body>
  <div class="shell">
    <div class="crt">

      <div id="victoryOverlay" class="overlay-victory">
        <div class="box">
          <h1>YOU ESCAPED!</h1>
          <p> </p>
        </div>
      </div>
      <div class="screen">
        <div class="left">
          <div class="header">ESCAPE THE CASTLE — TERMINAL</div>
          <div class="log" id="log"></div>

          <div class="status" id="status">
            <div class="badge" id="hp">HP: —</div>
            <div class="badge" id="noise">Noise: —</div>
            <div class="badge" id="room">Room: —</div>
            <div class="badge" id="inv">Inventory: —</div>
          </div>

          <div class="inputbar">
            <textarea id="cmd" rows="2" placeholder="> type what you do… (Enter to send)"></textarea>
            <button id="send">Send</button>
          </div>
        </div>

        <div class="right">
          <div class="imgwrap">
            <img id="roomimg" src="/static/art/cell.png" alt="scene">
          </div>
          <div class="caption" id="caption">Prison Cell</div>
        </div>
      </div>
    </div>
  </div>

<script>
  const victoryOverlay = document.getElementById('victoryOverlay');

  const log = document.getElementById('log');
  const cmd = document.getElementById('cmd');
  const send = document.getElementById('send');

  const hp = document.getElementById('hp');
  const noise = document.getElementById('noise');
  const room = document.getElementById('room');
  const inv = document.getElementById('inv');
  const img = document.getElementById('roomimg');
  const caption = document.getElementById('caption');

  // ---- Tangentbordsljud ----
  const keyAudio = new Audio('/static/sfx/keyboard.mp3');
  keyAudio.loop = true;
  keyAudio.volume = 0.6;

  function startKeyAudio(){
    try { keyAudio.currentTime = 0; keyAudio.play().catch(()=>{}); } catch(e){}
  }
  function stopKeyAudio(){
    try { keyAudio.pause(); keyAudio.currentTime = 0; } catch(e){}
  }

  // ---- Bakgrundsmusik per rum ----
  const bgmMap = {
    "cell_01": new Audio('/static/sfx/prison_music.mp3'),
    "coal_01": new Audio('/static/sfx/coalroom_music.mp3'),
    "hall_01": new Audio('/static/sfx/hallway_music.mp3'),
    "courtyard_01": new Audio('/static/sfx/courtyard_music.mp3')
  };
  for (const [k, a] of Object.entries(bgmMap)) { a.loop = true; a.volume = (k === "cell_01") ? 0.275 : 0.55; }

  // ---- SFX ----
  const sfx = {
    door:  new Audio('/static/sfx/opening_doors.mp3'),
    punch: new Audio('/static/sfx/punch_sound.mp3'),
    lose:  new Audio('/static/sfx/losing_sound.mp3'),
    win:   new Audio('/static/sfx/vinning_sound.mp3')
  };
  sfx.door.volume = 0.5; sfx.punch.volume = 0.7; sfx.lose.volume = 0.6; sfx.win.volume = 0.5;

  // ---- Preload ----
  keyAudio.preload = 'auto';
  Object.values(bgmMap).forEach(a => a.preload = 'auto');
  Object.values(sfx).forEach(a => a.preload = 'auto');

  // ---- BGM-state + unlock ----
  let currentBgmKey = null;
  let audioReady = false;
  let pendingBgmKey = 'cell_01';

  function setBgm(roomId){
    pendingBgmKey = roomId;
    if (!audioReady) return;
    if (roomId === currentBgmKey) return;

    if (currentBgmKey && bgmMap[currentBgmKey]) {
      const cur = bgmMap[currentBgmKey];
      cur.pause(); try { cur.currentTime = 0; } catch(e){}
    }
    const next = bgmMap[roomId];
    if (next){ next.play().then(()=>{ currentBgmKey = roomId; }).catch(()=>{}); }
  }

  function unlockAudioOnce(){
    if (audioReady) return;
    audioReady = true;

    const all = [keyAudio, ...Object.values(bgmMap), ...Object.values(sfx)];
    all.forEach(a => { try { a.load(); } catch(e){} });

    Promise.all(all.map(a=>{
      a.muted = true;
      return a.play().then(()=>{ a.pause(); a.currentTime = 0; a.muted = false; }).catch(()=>{ a.muted = false; });
    })).finally(()=>{ setBgm(pendingBgmKey); });
  }
  window.addEventListener('pointerdown', unlockAudioOnce, { once:true });
  window.addEventListener('keydown',     unlockAudioOnce, { once:true });

  function playOnce(audio){
    const a = audio.cloneNode(true);
    a.play().catch(()=>{});
  }

  // ---- Key-event detection (glow) ----
  const KEY_EVENT_PATTERNS = [
    /Torchlight pushes back the darkness/i,
    /You reveal a crawlable hole/i,
    /You carefully lift the loose cobblestone/i,
    /You (?:push|pull) the .* door open/i,
    /You (?:climb|step) onto the small platform/i,
    /With a thunderous slam the gate drops/i,
    /three armored guards/i,
    /You (?:loose|fire) .*guard/i,
    /The lawn falls silent/i,
    /You sprint across the lowered gate/i,
    /You swim hard, scramble up the far bank/i,
    /The knight whirls .* you knock him out/i,
    /You push the far door open and step into the great hall/i,
    /You pull the door open and step into the castle courtyard/i,
    /You carefully crawl into the hole and drop into the coal cellar/i
  ];
  function isKeyEventLine(text){ return KEY_EVENT_PATTERNS.some(re => re.test(text)); }

  function updateStatus(s){
    hp.textContent = `HP: ${s.hp}` + (s.hp_delta !== 0 ? ` (${s.hp_delta > 0 ? '+' : ''}${s.hp_delta})` : '');
    if (s.cause) hp.textContent += ` — ${s.cause}`;
    noise.textContent = `Noise: ${s.noise}`;
    room.textContent = `Room: ${s.room_title}`;
    inv.textContent = `Inventory: ${s.inventory && s.inventory.length ? s.inventory.join(', ') : 'empty'}`;

    // Endast byt bild om servern skickar en art. Annars behåll förra (inga bannade fallbacks).
    if (s.art){
      img.src = `/static/art/${s.art}`;
    }
    caption.textContent = s.room_title;
        // Victory overlay toggle
    if (s.game_won) {
      victoryOverlay.classList.add('show');
    } else {
      victoryOverlay.classList.remove('show');
    }

  }

  const DOOR_EVENTS = new Set(["open_hall_door", "unlock_courtyard_door"]);
  function handleAudioFromState(s){
    setBgm(s.room_id);

    if (s.events && s.events.some(e => DOOR_EVENTS.has(e))){ playOnce(sfx.door); }
    if (typeof s.hp_delta === 'number' && s.hp_delta < 0){ playOnce(sfx.punch); }
    if (s.dead){
      playOnce(sfx.lose);
      if (currentBgmKey && bgmMap[currentBgmKey]) bgmMap[currentBgmKey].pause();
    } else if (s.game_won){
      playOnce(sfx.win);
      if (currentBgmKey && bgmMap[currentBgmKey]) bgmMap[currentBgmKey].pause();
    }
  }

  const BASE_DELAY = 14;
  const PUNCT_PAUSE = { ',': 80, '.': 120, '!': 140, '?': 140, ';': 100, ':': 100 };

  function typeLine(lineText, parentEl, isKey){
    return new Promise(async (resolve) => {
      const line = document.createElement('div');
      line.className = 'sys line' + (isKey ? ' glowline' : '');
      parentEl.appendChild(line);

      for (let i=0; i<lineText.length; i++){
        line.textContent += lineText[i];
        log.scrollTop = log.scrollHeight;

        const ch = lineText[i];
        const extra = PUNCT_PAUSE[ch] || 0;
        await new Promise(r => setTimeout(r, BASE_DELAY));
        if (extra){ await new Promise(r => setTimeout(r, extra)); }
      }
      resolve();
    });
  }

  async function typeNarration(text){
    if (!text) return;
    const lines = text.split(/\r?\n/);
    for (const rawLine of lines){
      const t = rawLine;
      if (t.trim().length === 0){
        const spacer = document.createElement('div'); spacer.className = 'line'; spacer.innerHTML = '&nbsp;';
        log.appendChild(spacer); log.scrollTop = log.scrollHeight; continue;
      }
      const key = isKeyEventLine(t);
      await typeLine(t, log, key);
    }
  }

  function appendUser(text){
    const div = document.createElement('div');
    div.className = 'user line';
    div.textContent = '> ' + text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  const THINK_LINES = [
    "Gathering strength to do your action","Thinking about what to eat later","You forgot for a second what you were supposed to do",
    "Consulting the imaginary map","Counting cobblestones for luck","Practicing a heroic eyebrow raise",
    "Untangling the narrative threads","Rolling a mental d20","Quietly negotiating with physics","Rewinding the scene to check continuity"
  ];
  let thinkingTimer = null; let thinkingEl = null;

  function startThinking(){
    const msg = THINK_LINES[Math.floor(Math.random()*THINK_LINES.length)];
    thinkingEl = document.createElement('div'); thinkingEl.className = 'sys line thinking'; thinkingEl.textContent = msg;
    log.appendChild(thinkingEl); log.scrollTop = log.scrollHeight;

    let dots = 0;
    thinkingTimer = setInterval(()=>{ dots = (dots + 1) % 4; thinkingEl.textContent = msg + '.'.repeat(dots); log.scrollTop = log.scrollHeight; }, 350);
  }

  function stopThinking(){
    if (thinkingTimer){ clearInterval(thinkingTimer); thinkingTimer = null; }
    if (thinkingEl){ thinkingEl.textContent = thinkingEl.textContent.replace(/\.*$/, '...'); thinkingEl = null; }
  }

  async function fetchState(){
    const r = await fetch('/state');
    const s = await r.json();
    if (s.narration) await typeNarration(s.narration.trim());
    updateStatus(s);
    handleAudioFromState(s);
    cmd.focus();
  }

  async function sendCmd(){
    const text = cmd.value.trim();
    if(!text) return;
    appendUser(text);
    cmd.value = "";

    startThinking();
    try{
      const r = await fetch('/act', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ text }) });
      const s = await r.json();
      stopThinking();

      if (s.narration) await typeNarration(s.narration.trim());
      updateStatus(s);
      handleAudioFromState(s);
      cmd.focus();
    } catch(err){
      stopThinking();
      console.error(err);
    }
  }

  send.addEventListener('click', sendCmd);
  cmd.addEventListener('keydown', (e)=>{ if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendCmd(); } });

  window.addEventListener('pointerdown', unlockAudioOnce, { once:true });
  window.addEventListener('keydown',     unlockAudioOnce, { once:true });

  fetchState();
</script>
</body>
</html>
"""

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/state")
def get_state():
    """Första laddningen: returnera välkomsttext + banner + ev. intro för startrummet."""
    global SESSION_STARTED
    sync_flags_with_items(STATE)
    STATE.inventory = inventory_items_from_items(STATE)

    narration = ""
    if not SESSION_STARTED:
        SESSION_STARTED = True
        narration += WELCOME_TEXT.strip() + "\n"
        narration += append_room_entry_text(STATE)

    # Starta i cellen med cell.png (tillåten)
    start_art = ART_DEFAULT.get(STATE.current_room, "cell.png")

    return jsonify({
        "narration": narration,
        "hp": STATE.hp,
        "hp_delta": 0,
        "noise": 0,
        "inventory": STATE.inventory,
        "room_id": STATE.current_room,
        "room_title": ROOM_TITLES.get(STATE.current_room, STATE.current_room),
        "cause": "",
        "events": [],
        "game_won": False,
        "dead": STATE.hp <= 0,
        "art": start_art,
    })

@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(force=True)
    player_action = (data.get("text") or "").strip()
    if not player_action:
        return jsonify({"error": "empty"}), 400

    # Bygg prompt (samma som terminal-versionen)
    scene_card = SCENES[STATE.current_room]
    scene_json = json.dumps(scene_card, ensure_ascii=False, indent=2)

    sync_flags_with_items(STATE)
    flags_cell_str = json.dumps(STATE.flags_cell, ensure_ascii=False)
    flags_coal_str = json.dumps(STATE.flags_coal, ensure_ascii=False)
    flags_hall_str = json.dumps(STATE.flags_hall, ensure_ascii=False)
    flags_courtyard_str = json.dumps(STATE.flags_courtyard, ensure_ascii=False)
    items_str = json.dumps(STATE.items, ensure_ascii=False)
    inventory_str = json.dumps(STATE.inventory, ensure_ascii=False)

    user_prompt = USER_INSTRUCTION_TEMPLATE.format(
        room_id=STATE.current_room,
        room_title=scene_card.get("title", STATE.current_room),
        scene_card_json=scene_json,
        hp=STATE.hp,
        flags_cell=flags_cell_str,
        flags_coal=flags_coal_str,
        flags_hall=flags_hall_str,
        flags_courtyard=flags_courtyard_str,
        items=items_str,
        inventory=inventory_str,
        player_action=player_action
    )

    raw = CLIENT.chat_json(SYSTEM_PROMPT, user_prompt)
    llm = coerce_llm_result(raw)
    result = validate_and_apply(STATE, llm, player_action)

    narration = (result.get("narration") or "").strip()
    if result.get("room_transition"):
        narration = narration.rstrip() + "\n" + append_room_entry_text(STATE)

    STATE.inventory = inventory_items_from_items(STATE)

    # Art-val
    art_file = pick_art_filename(STATE, result, narration)

    return jsonify({
        "narration": narration,
        "hp": STATE.hp,
        "hp_delta": result.get("applied_hp_delta", 0),
        "noise": result.get("applied_noise", 0),
        "inventory": STATE.inventory,
        "room_id": STATE.current_room,
        "room_title": ROOM_TITLES.get(STATE.current_room, STATE.current_room),
        "cause": result.get("cause", ""),
        "events": result.get("events", []),
        "game_won": bool(result.get("game_won", False)),
        "dead": STATE.hp <= 0,
        "art": art_file,  # kan vara None → klienten behåller förra bilden
    })

# Statik (art)
@app.route("/static/art/<path:filename>")
def art_file(filename):
    return send_from_directory(os.path.join(app.static_folder, "art"), filename)

if __name__ == "__main__":
    # SFX läggs i static/sfx/...
    # Art läggs i static/art/ (endast de filer som listas högst upp)
    app.run(host="127.0.0.1", port=5000, debug=True)
