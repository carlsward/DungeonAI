# ui_server.py
# Flask-baserad web-UI för Escape the Castle med CRT-look, rums-intros och stabil inputrad.
from __future__ import annotations

import json
import os
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# Importera din motor och resurser
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

# ---------- Global enkel spelsession (single-user lokalt) ----------
STATE = GameState()
CLIENT = OllamaChat()
SESSION_STARTED = False  # för att ge uppstarts-intro exakt en gång


# ---------- Hjälp: generera intro/banner på rumsbyte ----------
ROOM_TITLES = {
    "cell_01": "Prison Cell",
    "coal_01": "Coal Cellar",
    "hall_01": "Great Hall",
    "courtyard_01": "Castle Courtyard",
}

def append_room_entry_text(state: GameState) -> str:
    """Returnerar banner + ev. intro/hint när man befinner sig i rummet nu."""
    room = state.current_room
    title = ROOM_TITLES.get(room, room)
    out = []

    # Banner
    out.append(f"\n— You are in the {title}. —\n")

    # Första-gången-intros + sätt flaggarna (samma semantik som terminalversionen)
    if room == "coal_01":
        if not state.flags_coal.get("coal_intro_shown", False):
            out.append(COAL_INTRO_TEXT + "\n")
            state.flags_coal["coal_intro_shown"] = True
        # Hint vid ljus: återskapa samma rad som i print_coal_lit_entry_hint_if_applicable()
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


# ---------- HTML (inline template) ----------
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
    --glass:#0b120b;
    --scan:#0c120c;
    --phosphor:#9cff57;  /* retrogrönt */
    --accent:#b6ff7b;
    --grid:#0a140a;
    --shadow: rgba(0,0,0,.7);
  }
  *{ box-sizing:border-box; }
  html,body{ height:100%; margin:0; background: #000; }
  body{
    display:flex;
    align-items:center;
    justify-content:center;
    padding:12px;
  }

  /* Ytterhölje (CRT-kåpa) */
  .shell{
    position:relative;
    width:min(1200px, 96vw);
    height:min(720px, 92vh);
    background: radial-gradient(120% 140% at 50% 50%, #0a0a0a 0%, #000 70%, #000 100%);
    border-radius: 22px;
    box-shadow:
      0 8px 40px rgba(0,0,0,.65),
      inset 0 0 140px rgba(0,0,0,.9);
    overflow:hidden;
  }

  /* Svag kromatisk “glas”-yta med scanlines och subtil böjning */
  .crt{
    position:absolute; inset:16px;
    background: linear-gradient(180deg, rgba(12,18,12,.65), rgba(12,18,12,.65));
    border-radius: 14px;
    padding:0;
    overflow:hidden;
    /* lätt utåtbuktning via filter + transform */
    filter: saturate(0.9) contrast(1.02);
    transform: perspective(1100px) translateZ(0) scale(1.002);
  }
  /* Scanlines-overlay */
  .crt::before{
    content:"";
    position:absolute; inset:0;
    pointer-events:none;
    background:
      repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,.16) 3px,
        rgba(0,0,0,.16) 4px
      );
    mix-blend-mode:multiply;
  }
  /* Subtil “flicker” och brightness-vobble */
  .crt::after{
    content:"";
    position:absolute; inset:-1px;
    pointer-events:none;
    background: radial-gradient(120% 180% at 50% 50%, rgba(255,255,255,.05), rgba(255,255,255,0) 60%);
    animation: flicker 6.5s infinite, glow 7.2s infinite;
  }
  @keyframes flicker {
    0%, 94%, 100% { opacity: .06; }
    96% { opacity: .11; }
    98% { opacity: .03; }
  }
  @keyframes glow {
    0%,100% { filter: brightness(1.00); }
    45% { filter: brightness(1.02); }
    70% { filter: brightness(0.98); }
  }

  /* Layout: vänster textpanel + höger bild */
  .screen{
    position:relative;
    display:grid;
    grid-template-columns: 1.4fr 0.9fr;
    gap:14px;
    padding:16px;
    width:100%;
    height:100%;
    color: var(--phosphor);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    text-shadow: 0 0 6px rgba(156,255,87,.25);
  }

  /* Textkolumn är en flex-kolumn som fyller höjd: log scrollar; input sticky i botten */
  .left{
    display:flex;
    flex-direction:column;
    min-width:0;
    background: linear-gradient(180deg, rgba(0,0,0,.35), rgba(0,0,0,.35)),
                repeating-linear-gradient(90deg, rgba(10,20,10,.16) 0 1px, transparent 1px 2px);
    border:1px solid rgba(156,255,87,.15);
    border-radius:12px;
    box-shadow: inset 0 0 18px rgba(0,0,0,.65);
    overflow:hidden;
  }

  .header{
    padding:10px 12px;
    border-bottom:1px solid rgba(156,255,87,.14);
    font-weight:700;
    letter-spacing:.5px;
  }

  .log{
    flex:1;
    overflow:auto;
    padding:12px;
    line-height:1.35;
    white-space:pre-wrap;
    scrollbar-color: rgba(156,255,87,.35) rgba(0,0,0,.2);
    scrollbar-width: thin;
  }
  .log .user{ color: var(--accent); }
  .log .sys { color: var(--phosphor); opacity:.95; }

  /* Sticky inputbar: försvinner inte, växer ej oändligt */
  .inputbar{
    position: sticky;
    bottom: 0;
    padding: 10px;
    background: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.75));
    border-top:1px solid rgba(156,255,87,.14);
    display:flex;
    gap:8px;
    align-items:flex-end;
  }
  .inputbar textarea{
    flex:1;
    min-height: 42px;
    max-height: 120px;   /* <-- STOPP: väx inte utanför skärmen */
    overflow:auto;       /* intern scrollbar */
    resize: vertical;    /* kan dras lite, men begränsas av max-height */
    padding:10px 10px 10px 12px;
    font: inherit;
    color: var(--phosphor);
    background: rgba(0,0,0,.45);
    border:1px solid rgba(156,255,87,.22);
    border-radius:8px;
    outline:none;
    box-shadow: inset 0 0 10px rgba(0,0,0,.55);
  }
  .inputbar button{
    padding:10px 14px;
    color:#051;
    background: linear-gradient(180deg, #baff7a, #8de354);
    border:none;
    border-radius:8px;
    font-weight:700;
    cursor:pointer;
    box-shadow: 0 2px 0 #5ca534, 0 6px 14px rgba(0,0,0,.25);
  }
  .inputbar button:active{ transform: translateY(1px); box-shadow: 0 1px 0 #5ca534, 0 4px 10px rgba(0,0,0,.3); }

  /* Statusrad ovanför input */
  .status{
    display:flex; gap:12px; flex-wrap:wrap;
    padding: 8px 12px 0 12px;
    font-size: 13px;
    opacity:.9;
  }
  .badge{
    padding:2px 8px; border-radius: 999px;
    border:1px solid rgba(156,255,87,.25);
    background: rgba(0,0,0,.35);
  }

  /* Bildpanel */
  .right{
    min-width:0;
    background: radial-gradient(140% 140% at 50% 50%, rgba(10,20,10,.5), rgba(0,0,0,.7));
    border:1px solid rgba(156,255,87,.15);
    border-radius:12px;
    box-shadow: inset 0 0 18px rgba(0,0,0,.65);
    display:flex; flex-direction:column; overflow:hidden;
  }
  .imgwrap{
    flex:1; display:flex; align-items:center; justify-content:center;
    padding: 12px;
  }
  .imgwrap img{
    width:100%; height:100%; object-fit:contain;
    image-rendering: pixelated; /* retro */
    filter: contrast(1.05) saturate(0.9) brightness(0.92);
  }
  .right .caption{
    padding:8px 10px; font-size:12px; opacity:.9;
    border-top:1px solid rgba(156,255,87,.14);
  }

  /* Smal layout */
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

  function appendLine(text, cls="sys"){
    const div = document.createElement('div');
    div.className = cls;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function updateStatus(s){
    hp.textContent = `HP: ${s.hp}` + (s.hp_delta !== 0 ? ` (${s.hp_delta > 0 ? '+' : ''}${s.hp_delta})` : '');
    if (s.cause) hp.textContent += ` — ${s.cause}`;
    noise.textContent = `Noise: ${s.noise}`;
    room.textContent = `Room: ${s.room_title}`;
    inv.textContent = `Inventory: ${s.inventory && s.inventory.length ? s.inventory.join(', ') : 'empty'}`;

    // byt bild
    const map = {
      "cell_01": "/static/art/cell.png",
      "coal_01": "/static/art/coal.png",
      "hall_01": "/static/art/hall.png",
      "courtyard_01": "/static/art/courtyard.png"
    };
    img.src = map[s.room_id] || "/static/art/cell.png";
    caption.textContent = s.room_title;
  }

  async function fetchState(){
    const r = await fetch('/state');
    const s = await r.json();
    if (s.narration) appendLine(s.narration.trim());
    updateStatus(s);
  }

  async function sendCmd(){
    const text = cmd.value.trim();
    if(!text) return;
    appendLine("> " + text, "user");
    cmd.value = "";

    const r = await fetch('/act', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text })
    });
    const s = await r.json();
    if (s.narration) appendLine(s.narration.trim(), "sys");
    updateStatus(s);
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
    """Första laddningen: returnera välkomsttext + banner + ev. intro för start-rummet."""
    global SESSION_STARTED
    sync_flags_with_items(STATE)
    STATE.inventory = inventory_items_from_items(STATE)

    narration = ""
    if not SESSION_STARTED:
        SESSION_STARTED = True
        # Samma känsla som terminalen:
        narration += WELCOME_TEXT.strip() + "\n"
        narration += append_room_entry_text(STATE)

    return jsonify({
        "narration": narration,
        "hp": STATE.hp,
        "hp_delta": 0,
        "noise": 0,
        "inventory": STATE.inventory,
        "room_id": STATE.current_room,
        "room_title": ROOM_TITLES.get(STATE.current_room, STATE.current_room),
        "cause": "",
    })

@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(force=True)
    player_action = (data.get("text") or "").strip()
    if not player_action:
        return jsonify({"error":"empty"}), 400

    # Förbered prompt (samma som i din main-loop)
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

    # LLM-call + motorns validering
    raw = CLIENT.chat_json(SYSTEM_PROMPT, user_prompt)
    llm = coerce_llm_result(raw)
    result = validate_and_apply(STATE, llm, player_action)

    # Bygg narration + lägg på rums-entry (banner/intro/hint) vid transition
    narration = (result.get("narration") or "").strip()
    if result.get("room_transition"):
        narration = narration.rstrip() + "\n" + append_room_entry_text(STATE)

    # Uppdatera inventory UI
    STATE.inventory = inventory_items_from_items(STATE)

    return jsonify({
        "narration": narration,
        "hp": STATE.hp,
        "hp_delta": result.get("applied_hp_delta", 0),
        "noise": result.get("applied_noise", 0),
        "inventory": STATE.inventory,
        "room_id": STATE.current_room,
        "room_title": ROOM_TITLES.get(STATE.current_room, STATE.current_room),
        "cause": result.get("cause", ""),
    })


# ---------- Statik (valfritt, enkla platshållare) ----------
@app.route("/static/art/<path:filename>")
def art_file(filename):
    # Servera bilder från ./static/art
    return send_from_directory(os.path.join(app.static_folder, "art"), filename)


if __name__ == "__main__":
    # Starta Flask. Sätt debug=False om du inte vill ha auto-reload/import två gånger.
    app.run(host="127.0.0.1", port=5000, debug=True)
