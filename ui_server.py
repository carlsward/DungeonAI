# ui_server.py
# Flask-baserad web-UI för Escape the Castle med CRT-look, rums-intros,
# stabil inputrad, glow-highlight för key events och typskrivaranimation + ljud + musik.

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

# Basbilder per rum
ART_DEFAULT = {
    "cell_01": "cell.png",
    "coal_01": "coal.png",
    "hall_01": "hall.png",
    "courtyard_01": "courtyard.png",
}

def _exists_art(filename: str) -> bool:
    return os.path.exists(os.path.join(app.static_folder, "art", filename))

def pick_art_filename(state: GameState, result: dict, narration: str) -> str:
    """
    Välj rätt bild för denna tick, utifrån aktuellt rum, events och flaggor.
    Prioriterad ordning per rum. Fallback till ART_DEFAULT[room].
    """
    events = set(result.get("events", []))
    room   = state.current_room

    # Neutral fallback om inget hände
    if ("Nothing happened." in (narration or "")) and not events:
        if _exists_art("nothing_happened.png"):
            return "nothing_happened.png"

    # ----- Prison Cell -----
    if room == "cell_01":
        if "light_torch" in events:
            if _exists_art("setting_fire_to_torch.png"):
                return "setting_fire_to_torch.png"
        if "stone_lifted" in events or state.flags_cell.get("stone_moved"):
            if _exists_art("movestonesaside.png"):
                return "movestonesaside.png"
        if "straw_rummaged" in events or state.flags_cell.get("found_loose_stone"):
            if _exists_art("finding_cobblestone_2.png"):
                return "finding_cobblestone_2.png"
        return ART_DEFAULT["cell_01"]

    # ----- Coal Cellar -----
    if room == "coal_01":
        if not state.flags_coal.get("torch_lit", False):
            if _exists_art("first_time_in_coalroom_without_lit_torch.png"):
                return "first_time_in_coalroom_without_lit_torch.png"
        if state.flags_coal.get("torch_lit", False):
            if _exists_art("in_coalroom_with_torch.png"):
                return "in_coalroom_with_torch.png"
        return ART_DEFAULT["coal_01"]

    # ----- Great Hall -----
    if room == "hall_01":
        if "combat_knock_guard" in events:
            # Stöder både korrekt och felstavat filnamn
            if _exists_art("fight_with_torch.png"):
                return "fight_with_torch.png"
            if _exists_art("fiht_with_torch.png"):
                return "fiht_with_torch.png"
            if _exists_art("fight_with_hands.png"):
                return "fight_with_hands.png"
        if "pickup_keys" in events and _exists_art("smygande_bakom2.png"):
            return "smygande_bakom2.png"
        return ART_DEFAULT["hall_01"]

    # ----- Courtyard -----
    if room == "courtyard_01":
        # Vinst (både bro och simma – använder samma flyktbild)
        if "cross_gate_bridge" in events or "swim_across" in events or result.get("game_won"):
            if _exists_art("running_away.png"):
                return "running_away.png"
        if "shoot_guard" in events and _exists_art("shooting_from_tower2.png"):
            return "shooting_from_tower2.png"
        if "climb_ladder_up" in events or state.flags_courtyard.get("at_tower_top"):
            if _exists_art("going_up_tower.png"):
                return "going_up_tower.png"
        return ART_DEFAULT["courtyard_01"]

    # Fallback
    return ART_DEFAULT.get(room, "cell.png")


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
            # Den här raden vill du uttryckligen highlighta
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
  :root{
    --bg:#0b0f0b;
    --phosphor:#9cff57;
    --accent:#b6ff7b;
  }
  *{ box-sizing:border-box; }
  html,body{ height:100%; margin:0; background:#000; }
  body{ display:flex; align-items:center; justify-content:center; padding:12px; }

  .shell{
    position:relative;
    width:min(1200px, 96vw);
    height:min(720px, 92vh);
    background: radial-gradient(120% 140% at 50% 50%, #0a0a0a 0%, #000 70%, #000 100%);
    border-radius:22px;
    box-shadow: 0 8px 40px rgba(0,0,0,.65), inset 0 0 140px rgba(0,0,0,.9);
    overflow:hidden;
  }
  .crt{
    position:absolute; inset:16px; border-radius:14px; overflow:hidden;
    background: linear-gradient(180deg, rgba(12,18,12,.65), rgba(12,18,12,.65));
    filter: saturate(0.9) contrast(1.02);
    transform: perspective(1100px) translateZ(0) scale(1.002);
  }
  .crt::before{
    content:""; position:absolute; inset:0; pointer-events:none;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,.16) 3px,
        rgba(0,0,0,.16) 4px
    );
    mix-blend-mode:multiply;
  }
  .crt::after{
    content:""; position:absolute; inset:-1px; pointer-events:none;
    background: radial-gradient(120% 180% at 50% 50%, rgba(255,255,255,.05), rgba(255,255,255,0) 60%);
    animation: flicker 6.5s infinite, glow 7.2s infinite;
  }
  @keyframes flicker { 0%,94%,100%{opacity:.06;} 96%{opacity:.11;} 98%{opacity:.03;} }
  @keyframes glow { 0%,100%{filter:brightness(1.00);} 45%{filter:brightness(1.02);} 70%{filter:brightness(0.98);} }

  .screen{
    position:relative;
    display:grid; grid-template-columns: 1.4fr 0.9fr; gap:14px;
    padding:16px; width:100%; height:100%;
    color: var(--phosphor);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    text-shadow: 0 0 6px rgba(156,255,87,.25);
  }

  .left{
    display:flex; flex-direction:column; min-width:0; overflow:hidden;
    background: linear-gradient(180deg, rgba(0,0,0,.35), rgba(0,0,0,.35)),
                repeating-linear-gradient(90deg, rgba(10,20,10,.16) 0 1px, transparent 1px 2px);
    border:1px solid rgba(156,255,87,.15); border-radius:12px;
    box-shadow: inset 0 0 18px rgba(0,0,0,.65);
  }
  .header{ padding:10px 12px; border-bottom:1px solid rgba(156,255,87,.14); font-weight:700; letter-spacing:.5px; }
  .log{ flex:1; overflow:auto; padding:12px; line-height:1.35; white-space:pre-wrap;
        scrollbar-color: rgba(156,255,87,.35) rgba(0,0,0,.2); scrollbar-width: thin; }
  .log .user{ color: var(--accent); }
  .log .sys{}
  .line{ margin: 0 0 4px 0; }

  /* Key-event highlight (glow + pulser) */
  .glowline{
    text-shadow:
      0 0 8px rgba(156,255,87,.85),
      0 0 18px rgba(156,255,87,.45),
      0 0 32px rgba(156,255,87,.25);
    animation: pulse 2.1s ease-in-out 2;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.85} }

  .status{ display:flex; gap:12px; flex-wrap:wrap; padding:8px 12px 0 12px; font-size:13px; opacity:.9; }
  .badge{ padding:2px 8px; border-radius:999px; border:1px solid rgba(156,255,87,.25); background: rgba(0,0,0,.35); }

  .inputbar{
    position: sticky; bottom: 0; padding:10px; display:flex; gap:8px; align-items:flex-end;
    background: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.75));
    border-top:1px solid rgba(156,255,87,.14);
  }
  .inputbar textarea{
    flex:1; min-height:42px; max-height:120px; overflow:auto; resize:vertical;
    padding:10px 10px 10px 12px; font:inherit; color:var(--phosphor);
    background: rgba(0,0,0,.45); border:1px solid rgba(156,255,87,.22); border-radius:8px;
    outline:none; box-shadow: inset 0 0 10px rgba(0,0,0,.55);
  }
  .inputbar button{
    padding:10px 14px; color:#051; background: linear-gradient(180deg, #baff7a, #8de354);
    border:none; border-radius:8px; font-weight:700; cursor:pointer;
    box-shadow: 0 2px 0 #5ca534, 0 6px 14px rgba(0,0,0,.25);
  }
  .inputbar button:active{ transform: translateY(1px); box-shadow: 0 1px 0 #5ca534, 0 4px 10px rgba(0,0,0,.3); }

  .right{
    min-width:0; display:flex; flex-direction:column; overflow:hidden;
    background: radial-gradient(140% 140% at 50% 50%, rgba(10,20,10,.5), rgba(0,0,0,.7));
    border:1px solid rgba(156,255,87,.15); border-radius:12px;
    box-shadow: inset 0 0 18px rgba(0,0,0,.65);
  }
  .imgwrap{ flex:1; display:flex; align-items:center; justify-content:center; padding: 12px; }
  .imgwrap img{ width:100%; height:100%; object-fit:contain; image-rendering: pixelated; filter: contrast(1.05) saturate(0.9) brightness(0.92); }

  /* STÖRRE & CENTRERAD platsrubrik */
  .right .caption{
    padding:10px 12px; font-size:18px; font-weight:800; letter-spacing:.8px; text-align:center;
    border-top:1px solid rgba(156,255,87,.14);
  }

  @media (max-width: 900px){
    .screen{ grid-template-columns: 1fr; }
    .right{ order:-1; height: 36vh; }
  }
</style>
</head>
<body>
  <div class="shell">
    <div class="crt">
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
    keyAudio.currentTime = 0;
    keyAudio.play().catch(()=>{ /* autoplay kan vara blockerad, ignoreras */ });
  }
  function stopKeyAudio(){
    keyAudio.pause();
    keyAudio.currentTime = 0;
  }

  // ---- Bakgrundsmusik per rum ----
  const bgmMap = {
    "cell_01": new Audio('/static/sfx/prison_music.mp3'),
    "coal_01": new Audio('/static/sfx/coalroom_music.mp3'),
    "hall_01": new Audio('/static/sfx/hallway_music.mp3'),
    "courtyard_01": new Audio('/static/sfx/courtyard_music.mp3')
  };
  for (const [key, a] of Object.entries(bgmMap)) {
  a.loop = true;
  a.volume = (key === "cell_01") ? 0.275 : 0.55; // 0.55 / 2
}


  let currentBgmKey = null;
  function setBgm(roomId){
    if (currentBgmKey && bgmMap[currentBgmKey]){
      bgmMap[currentBgmKey].pause();
      try { bgmMap[currentBgmKey].currentTime = 0; } catch(e){}
    }
    const next = bgmMap[roomId];
    if (next){
      currentBgmKey = roomId;
      next.play().catch(()=>{ /* kräver user gesture första gången */ });
    }
  }

  // ---- SFX ----
  const sfx = {
    door: new Audio('/static/sfx/opening_doors.mp3'),
    punch: new Audio('/static/sfx/punch_sound.mp3'),
    lose: new Audio('/static/sfx/losing_sound.mp3'),
    win: new Audio('/static/sfx/vinning_sound.mp3')
  };
  sfx.door.volume = 0.9;
  sfx.punch.volume = 0.9;
  sfx.lose.volume = 0.95;
  sfx.win.volume = 0.95;

  function playOnce(audio){
    // skapa en klon så flera kan överlappa om det behövs
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
  function isKeyEventLine(text){
    return KEY_EVENT_PATTERNS.some(re => re.test(text));
  }

  // ---- Hjälp: status & bilder ----
  function updateStatus(s){
    hp.textContent = `HP: ${s.hp}` + (s.hp_delta !== 0 ? ` (${s.hp_delta > 0 ? '+' : ''}${s.hp_delta})` : '');
    if (s.cause) hp.textContent += ` — ${s.cause}`;
    noise.textContent = `Noise: ${s.noise}`;
    room.textContent = `Room: ${s.room_title}`;
    inv.textContent = `Inventory: ${s.inventory && s.inventory.length ? s.inventory.join(', ') : 'empty'}`;

    // NYTT: använd bild som servern valt, annars fallback per rum
    if (s.art){
      img.src = `/static/art/${s.art}`;
    } else {
      const map = {
        "cell_01": "/static/art/cell.png",
        "coal_01": "/static/art/coal.png",
        "hall_01": "/static/art/hall.png",
        "courtyard_01": "/static/art/courtyard.png"
      };
      img.src = map[s.room_id] || "/static/art/cell.png";
    }
    caption.textContent = s.room_title;
  }

  // ---- Ljudlogik kopplat till state från servern ----
  const DOOR_EVENTS = new Set(["open_hall_door", "unlock_courtyard_door", "return_to_coal"]);
  function handleAudioFromState(s, prev){
    // Start/byt BGM vid rum
    setBgm(s.room_id);

    // Dörrljud om events innehåller dörr-aktiviteter
    if (s.events && s.events.some(e => DOOR_EVENTS.has(e))){
      playOnce(sfx.door);
    }

    // Punch varje gång hp_delta < 0
    if (typeof s.hp_delta === 'number' && s.hp_delta < 0){
      playOnce(sfx.punch);
    }

    // Losing/winning – spela en gång när det inträffar
    if (s.dead){
      playOnce(sfx.lose);
      if (currentBgmKey && bgmMap[currentBgmKey]) bgmMap[currentBgmKey].pause();
    } else if (s.game_won){
      playOnce(sfx.win);
      if (currentBgmKey && bgmMap[currentBgmKey]) bgmMap[currentBgmKey].pause();
    }
  }

  // ---- Typskrivaranimation för SYS-rader ----
  const BASE_DELAY = 14; // ms per tecken
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
        const delay = BASE_DELAY;
        await new Promise(r => setTimeout(r, delay));
        if (extra){ await new Promise(r => setTimeout(r, extra)); }
      }
      resolve();
    });
  }

  async function typeNarration(text){
    if (!text) return;
    startKeyAudio();
    const lines = text.split(/\r?\n/);
    for (const rawLine of lines){
      const t = rawLine;
      if (t.trim().length === 0){
        const spacer = document.createElement('div');
        spacer.className = 'line';
        spacer.innerHTML = '&nbsp;';
        log.appendChild(spacer);
        log.scrollTop = log.scrollHeight;
        continue;
      }
      const key = isKeyEventLine(t);
      await typeLine(t, log, key);
    }
    stopKeyAudio();
  }

  // ---- Vanliga utskrifter (user prompt)—utan animation
  function appendUser(text){
    const div = document.createElement('div');
    div.className = 'user line';
    div.textContent = '> ' + text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  // ---- API-anrop ----
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

    const r = await fetch('/act', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text })
    });
    const s = await r.json();
    if (s.narration) await typeNarration(s.narration.trim());
    updateStatus(s);
    handleAudioFromState(s);
    cmd.focus();
  }

  send.addEventListener('click', sendCmd);
  cmd.addEventListener('keydown', (e)=>{
    if (e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendCmd();
    }
  });

  // Init
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

    # första laddningen – inget hp_delta/noise, och inga events ännu
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
        "art": ART_DEFAULT.get(STATE.current_room, "cell.png"),  # <-- bild för startscreen
    })

@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(force=True)
    player_action = (data.get("text") or "").strip()
    if not player_action:
        return jsonify({"error": "empty"}), 400

    # Förbered prompten (samma som i terminal-versionen)
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

    # Välj rätt art för klienten
    art_file = pick_art_filename(STATE, result, narration)

    # skicka events + vinst/död i svaret så klienten kan trigga ljud
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
        "art": art_file,
    })

# Statik (art)
@app.route("/static/art/<path:filename>")
def art_file(filename):
    return send_from_directory(os.path.join(app.static_folder, "art"), filename)

if __name__ == "__main__":
    # Lägg filer i:
    # static/sfx/{keyboard.mp3, prison_music.mp3, coalroom_music.mp3, hallway_music.mp3, courtyard_music.mp3,
    #            opening_doors.mp3, punch_sound.mp3, losing_sound.mp3, vinning_sound.mp3}
    # Lägg scenbilder i static/art/{cell.png, coal.png, hall.png, courtyard.png, ...}
    app.run(host="127.0.0.1", port=5000, debug=True)
