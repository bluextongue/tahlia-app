# app.py
# Tahlia: Flask + Google Gemini (2.0 Flash) + ElevenLabs (per-user session state)
# - Per-client (cid) memory so intro/dedup/double-speak are isolated per user
# - Always-on ASR with final-only interrupt + echo filter
# - Faster feel: lower interrupt grace (700ms), quick send debounce (220ms)
# - Robust LLM error logging + safe fallback that avoids duplicate blocking
# - Modal/backdrop fix + button click fixes
# - Minimal change: /api/intro always sends an intro (no double-intro warning)
# - NEW: Adjacent (anticipatory) processing — quick teaser reply on interim ASR

import base64, re, time, random, os, threading, json, sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from flask import Flask, request, jsonify, make_response, send_file
from io import BytesIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= YOUR API KEYS (LOCAL ONLY) =========
ELEVEN_API_KEY  = "3e7c3a7c14cec12c34324bd0d25a063ae44b3f4c09b1d25ac1dbcd5a606652d8"
ELEVEN_VOICE_ID = "X03mvPuTfprif8QBAVeJ"

# Google Gemini API (AI Studio)
GOOGLE_API_KEY  = "AIzaSyBqKNG0-SEQapQBtZaHa_YdiabBfm_ADLY"
GEMINI_MODEL    = "gemini-2.0-flash"

ASSISTANT_NAME  = "Tahlia"

# ========= HTTP session with retries =========
session = requests.Session()
retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.35,
              status_forcelist=[429,500,502,503,504], allowed_methods=["GET","POST"], raise_on_status=False)
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
session.mount("https://", adapter); session.mount("http://", adapter)
DEFAULT_TIMEOUT = (5, 55)

# ========= Flask =========
app = Flask(__name__)

# ========= Per-client state =========
@dataclass
class ClientState:
    history: deque = field(default_factory=lambda: deque(maxlen=16))
    intro_sent: bool = False
    last_spoke: str | None = None
    last_reply: str = ""
    recent_assist: deque = field(default_factory=lambda: deque(maxlen=6))
    recent_style: deque = field(default_factory=lambda: deque(maxlen=4))
    last_adjacent_ts: float = 0.0  # cooldown for teaser generation

state_lock = threading.Lock()
clients: dict[str, ClientState] = defaultdict(ClientState)

def get_state(cid: str) -> ClientState:
    if not cid:
        cid = "anon"
    with state_lock:
        return clients[cid]

def reset_state(cid: str):
    with state_lock:
        clients[cid] = ClientState()

# ========= Prompts and rules =========
SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a warm, natural, therapist-like conversational partner. "
    "Default stance: be genuinely curious and specific to the user's situation—ask focused, non-generic questions that help get to the root of what’s going on. "
    "Keep 2–3 sentences by default (up to 5 if the user shares details). "
    "Use one of these styles per turn (don’t stack them back-to-back): "
    "• Inquire: reflect briefly, then ask 1 precise question that narrows the real issue (e.g., where/when/who/what made it harder). "
    "• Tip (sometimes): offer a tailored, bite-sized suggestion tied to what the user said—no menus of exercises, just one concrete move. "
    "• Story (rarely): share a very short, relatable vignette ('some people find…') to normalize their experience, then invite them back. "
    "Avoid generic platitudes and the phrases “I’m here with you” and “Let’s take it one step at a time.” "
    "Crisis: if the user indicates imminent self-harm, advise calling 911 or contacting/texting 988 (U.S. crisis line) immediately."
)

CRISIS_TRIGGERS = ("kill myself","suicide","hurt myself","harm myself","overdose","end my life","take my life","self harm","self-harm")
STOP_WORDS = ("stop","quit","end","goodbye","end call","terminate")

# ========= Helpers =========
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def contains_stop_command(text: str) -> bool:
    lw = norm(text)
    return lw in STOP_WORDS or (len(lw) <= 24 and any(w in lw for w in STOP_WORDS))

def detect_crisis(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in CRISIS_TRIGGERS)

def concise(text: str, max_chars: int = 520, max_sents: int = 5) -> str:
    t = (text or "").strip()
    if not t: return t
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:max_sents]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"\s+\S*$", "", t).rstrip(",;:.!? ")
    return t

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    return inter / (len(a | b) or 1)

def tokset(s: str) -> set:
    return set(re.findall(r"[a-z']+", (s or "").lower()))

def too_similar(new: str, refs: list[str], threshold: float = 0.90) -> bool:
    A = tokset(new)
    for r in refs:
        if not r: continue
        if jaccard(A, tokset(r)) >= threshold:
            return True
    return False

BANNED_PREFIXES = ("i’m here with you", "let’s take it one step at a time")

def not_duplicate_user(st: ClientState, text: str) -> bool:
    if not st.history: return True
    last = st.history[-1]
    return not (last[0] == "user" and norm(last[1]) == norm(text))

def not_duplicate_bot(st: ClientState, text: str) -> bool:
    return norm(st.last_reply) != norm(text)

def ends_with_question(s: str) -> bool:
    return bool(re.search(r"\?\s*$", (s or "")))

def choose_style(st: ClientState) -> str:
    last = st.recent_style[-1] if st.recent_style else None
    weights = {"inquire": 0.60, "tip": 0.30, "story": 0.10}
    if last in ("tip", "story"):
        weights = {"inquire": 0.75, "tip": 0.18, "story": 0.07}
    r = random.random()
    if r < weights["inquire"]:
        return "inquire"
    elif r < weights["inquire"] + weights["tip"]:
        return "tip"
    return "story"

# ========= Google Gemini Chat =========
def _to_gemini_contents(messages):
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "") or ""
        if role == "system":
            role = "user"
            text = f"[SYSTEM INSTRUCTIONS]\n{text}"
        elif role == "assistant":
            role = "model"
        else:
            role = "user"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents

def gemini_chat(messages, model=GEMINI_MODEL, temperature=0.55, max_tokens=360, timeout=DEFAULT_TIMEOUT):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": _to_gemini_contents(messages),
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "topP": 0.9,
            "topK": 40
        }
    }
    headers = {"Content-Type": "application/json"}
    r = session.post(url, headers=headers, json=payload, timeout=timeout)

    if r.status_code >= 400:
        sys.stderr.write(f"\n[GEMINI HTTP {r.status_code}] {r.text[:500]}\n")
        raise RuntimeError(f"Gemini HTTP {r.status_code}")

    data = r.json() or {}
    cands = data.get("candidates") or []
    if not cands:
        pf = data.get("promptFeedback")
        sys.stderr.write(f"\n[GEMINI EMPTY] feedback={json.dumps(pf)[:500]} raw={json.dumps(data)[:500]}\n")
        return "", "empty"

    first = cands[0]
    parts = ((first.get("content") or {}).get("parts") or [])
    text = (parts[0].get("text") if parts else "") or ""
    if not text.strip():
        safety = first.get("safetyRatings") or []
        sys.stderr.write(f"\n[GEMINI NO TEXT] safety={json.dumps(safety)[:500]} raw={json.dumps(first)[:500]}\n")
        return "", "no_text"
    return text.strip(), "ok"

def llm_reply(st: ClientState, user_text: str) -> tuple[str, str]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in list(st.history)[-12:]:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    style = choose_style(st)
    if style == "inquire":
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Inquire style. Give a short reflection tied to their words, "
            "then ask ONE precise, non-generic question that helps pinpoint the root cause or a specific blocker. "
            "No exercise suggestions in this turn."
        )})
    elif style == "tip":
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Tip style. Offer ONE tailored, concrete suggestion tightly linked to what they said. "
            "Keep it tiny and situational. Optionally end with a short follow-up question."
        )})
    else:  # story
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Story style. Share a very brief, relatable vignette ('some people find…') "
            "that normalizes their experience, then invite them back. Keep it 2–3 sentences."
        )})

    dbg = "ok"
    try:
        text, status = gemini_chat(msgs)
        if not text:
            dbg = f"llm_empty_{status}"
            text = ("Got it—when does school feel toughest: during classes, homework load, or dealing with people there?")
    except Exception as e:
        dbg = "llm_error"
        sys.stderr.write(f"\n[LLM ERROR] {e}\n")
        text = ("Quick check-in: what part of school is spiking the stress most today—time pressure, a specific class, or something social?")

    final = concise(text)

    # Avoid 3 questions in a row
    recent_qs = sum(1 for r in list(st.recent_assist)[-3:] if ends_with_question(r))
    if ends_with_question(final) and recent_qs >= 2:
        msgs.append({"role": "system", "content": "Regenerate without ending in a question. Keep it specific and user-focused."})
        try:
            alt, _ = gemini_chat(msgs)
            if alt: final = concise(alt); dbg = "regen_no_question"
        except Exception:
            pass

    low = norm(final)
    if low.startswith(BANNED_PREFIXES) or too_similar(final, list(st.recent_assist)):
        msgs.append({"role": "system", "content": "Regenerate with different wording; be specific to the user's last message. Allow up to 5 sentences if helpful."})
        try:
            alt, _ = gemini_chat(msgs)
            if alt: final = concise(alt); dbg = "regen_diversity"
        except Exception:
            pass

    st.recent_style.append(style)
    st.history.append(("user", user_text))
    st.history.append(("assistant", final))
    st.recent_assist.append(final)
    return final, dbg

# ---- Quick teaser (adjacent) ----
ADJ_MIN_CHARS = 14
ADJ_COOLDOWN_S = 1.8

def llm_teaser(st: ClientState, user_prefix: str) -> tuple[str, str]:
    # Do NOT add to history. This is a provisional one-liner based on the prefix only.
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content":
         ("For THIS reply only: Generate a SINGLE short sentence as a provisional response to the user's partial utterance. "
          "Be concrete and specific to the snippet; avoid long windups and hedging. "
          "Prefer statements over questions unless a question is obvious and crisp. "
          "Do not use generic empathy phrases or platitudes. "
          "This will be spoken immediately and may be replaced by a fuller answer.")},
        {"role": "user", "content": user_prefix.strip()}
    ]
    try:
        text, status = gemini_chat(msgs, temperature=0.4, max_tokens=40, timeout=(3, 12))
        if not text:
            return "Okay—go on.", "adj_empty"
        return concise(text, max_chars=140, max_sents=1), "adj_ok"
    except Exception as e:
        sys.stderr.write(f"\n[ADJ ERROR] {e}\n")
        return "", "adj_err"

# ========= ElevenLabs TTS =========
def tts_b64(text: str):
    if not ELEVEN_API_KEY: return "", "Missing ELEVENLABS_API_KEY"
    safe_text = (text or "").strip()
    if not safe_text: return "", ""
    safe_text = safe_text[:650]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {
        "text": safe_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.85, "style": 0.25, "use_speaker_boost": True},
        "output_format": "mp3_22050_64"
    }

    for i, timeout in enumerate([(5, 20), (5, 25), (6, 35)]):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                if i < 2: time.sleep(0.15 * (i + 1)); continue
                return "", f"TTS HTTP {r.status_code}: {r.text[:200]}"
            b64 = base64.b64encode(r.content).decode("utf-8")
            return "data:audio/mpeg;base64," + b64, ""
        except Exception as e:
            if i == 2: return "", f"TTS exception: {e}"
            time.sleep(0.2 * (i + 1))
    return "", "TTS unknown error"

# ========= API: introduction =========
@app.post("/api/intro")
def api_intro():
    data = request.get_json(force=True, silent=False) or {}
    cid = (data.get("cid") or "").strip()

    reset_state(cid)
    st = get_state(cid)
    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health assistant."
    st.intro_sent = True
    st.last_spoke = None
    st.last_reply = intro
    st.recent_assist.append(intro)

    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro"}), 200

# ========= API: chat reply (final) =========
@app.post("/api/reply")
def api_reply():
    data = request.get_json(force=True, silent=False) or {}
    cid = (data.get("cid") or "").strip()
    st = get_state(cid)

    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "empty_text"}), 400

    lower_text = norm(user_text)

    if contains_stop_command(lower_text):
        reset_state(cid)
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "stop_word"}), 200

    if lower_text == norm(st.last_reply):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "echo_bot_line"}), 200

    if len(lower_text) < 1:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "too_short"}), 200

    st.last_spoke = "user"

    if not not_duplicate_user(st, lower_text):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "duplicate_user"}), 200

    if detect_crisis(lower_text):
        reply = concise(
            "I’m really glad you told me. If you’re in immediate danger, call 911. "
            "In the U.S., you can call or text 988 for the Suicide & Crisis Lifeline. "
            "Would you like resources now?"
        )
        st.history.append(("user", user_text)); st.history.append(("assistant", reply))
        dbg = "crisis"
    else:
        reply, dbg = llm_reply(st, user_text)

    if not not_duplicate_bot(st, reply):
        reply = reply.rstrip(".") + " — when did that start showing up for you?"
        dbg = "dedup_softened"

    audio, tts_err = tts_b64(reply)
    st.last_reply = reply
    return jsonify({"reply": reply, "audio": audio, "tts_error": tts_err, "dbg": dbg}), 200

# ========= API: adjacent teaser (provisional) =========
@app.post("/api/adjacent")
def api_adjacent():
    data = request.get_json(force=True, silent=False) or {}
    cid = (data.get("cid") or "").strip()
    prefix = (data.get("prefix") or "").strip()
    if not prefix or len(prefix) < ADJ_MIN_CHARS:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "adj_short"}), 200

    st = get_state(cid)
    # Cooldown so we don't spam multiple teasers per breath
    now = time.time()
    if now - (st.last_adjacent_ts or 0) < ADJ_COOLDOWN_S:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "adj_cooldown"}), 200

    # Crisis/stop checks on prefix just in case
    lw = norm(prefix)
    if contains_stop_command(lw):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "adj_stop"}), 200
    if detect_crisis(lw):
        teaser = concise("If you’re in immediate danger, call 911. In the U.S., text or call 988 for the Suicide & Crisis Lifeline.")
        audio, tts_err = tts_b64(teaser)
        st.last_adjacent_ts = time.time()
        return jsonify({"reply": teaser, "audio": audio, "tts_error": tts_err, "dbg": "adj_crisis"}), 200

    teaser, dbg = llm_teaser(st, prefix)
    if not teaser:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": dbg}), 200

    # Don't store in history; but keep last_reply for echo filter on client
    audio, tts_err = tts_b64(teaser)
    st.last_adjacent_ts = time.time()
    return jsonify({"reply": teaser, "audio": audio, "tts_error": tts_err, "dbg": dbg}), 200

# ========= API: assistant speech ACK =========
@app.post("/api/ack_assistant")
def api_ack_assistant():
    data = request.get_json(force=True, silent=False) or {}
    cid = (data.get("cid") or "").strip()
    st = get_state(cid)
    st.last_spoke = "assistant"
    return jsonify({"ok": True})

# ========= API: reset memory =========
@app.post("/api/reset")
def api_reset():
    data = request.get_json(force=True, silent=False) or {}
    cid = (data.get("cid") or "").strip()
    reset_state(cid)
    return jsonify({"ok": True})

# ========= Health/ping =========
@app.get("/api/ping")
def api_ping():
    cid = request.args.get("cid", "").strip()
    st = get_state(cid) if cid else ClientState()
    return jsonify({"ok": True, "ts": time.time(), "intro_sent": st.intro_sent, "last_spoke": st.last_spoke})

# ========= Favicon =========
@app.get("/favicon.ico")
def favicon():
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    return send_file(BytesIO(png), mimetype="image/x-icon")

# ========= UI (client) =========
@app.get("/")
def index():
    html = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Tahlia – Voice Companion</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:#5b5bd6; --border:#1b1f24; }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { background:var(--bg); color:var(--fg); margin:0; font-family:system-ui, sans-serif; display:flex; min-height:100vh; width:100vw; overflow:hidden; position:relative; }

  #left { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:24px; min-width:0; position:relative; z-index:1; }
  #ball { width:120px; height:120px; border-radius:50%; background:var(--accent); transition:transform 0.08s ease; box-shadow:0 10px 30px rgba(0,0,0,.3); }
  button { cursor:pointer; }
  #startBtn, #logsBtn { position:relative; z-index:2; }
  #startBtn { padding:14px 20px; border:none; border-radius:12px; background:#161a1f; color:#fff; font-weight:600; font-size:16px; box-shadow:0 2px 10px rgba(0,0,0,.25); }
  #logsBtn { padding:10px 14px; border:1px solid var(--border); border-radius:10px; background:#0e1116; color:#cfd8e3; font-weight:600; font-size:13px; }

  #modalBackdrop { position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; pointer-events:none; align-items:center; justify-content:center; z-index:10; }
  #modalBackdrop.show { display:flex; pointer-events:auto; }
  #modal { width:min(720px, 92vw); height:min(70vh, 86vh); background:var(--panel); border:1px solid var(--border); border-radius:12px; display:flex; flex-direction:column; overflow:hidden; }
  #modalHeader { display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border); background:#0e1116; }
  #modalTitle { font-size:14px; color:#9aa4af; }
  #modalClose { border:none; background:transparent; color:#cfd8e3; font-size:18px; cursor:pointer; padding:4px 8px; }
  #transcript { flex:1; overflow:auto; padding:10px; font-size:14px; line-height:1.35; }
  .line { padding:6px 8px; border-bottom:1px dashed #20242a; white-space:pre-wrap; word-break:break-word; }
  .user { color:#c9f0ff; } .bot{ color:#d7ffe0; } .sys{ color:#ffd7d7; } .meta{ font-size:12px; color:#9aa4af; }

  #player { position:absolute; width:0; height:0; opacity:0; pointer-events:none; }
</style>
</head>
<body>
  <div id="left">
    <div id="ball" aria-hidden="true"></div>
    <button id="startBtn" type="button">Start Conversation</button>
    <button id="logsBtn" type="button">Transcript & Logs</button>
    <audio id="player" autoplay></audio>
  </div>

  <div id="modalBackdrop" aria-hidden="true">
    <div id="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
      <div id="modalHeader">
        <div id="modalTitle">Transcript & Logs</div>
        <button id="modalClose" type="button" title="Close">✕</button>
      </div>
      <div id="transcript"></div>
    </div>
  </div>

<script>
function uuidv4(){
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}
let clientId = localStorage.getItem("tahlia_cid");
if (!clientId){ clientId = uuidv4(); localStorage.setItem("tahlia_cid", clientId); }

const player = document.getElementById("player");
const startBtn = document.getElementById("startBtn");
const logsBtn = document.getElementById("logsBtn");
const ball = document.getElementById("ball");
const modalBackdrop = document.getElementById("modalBackdrop");
const modalClose = document.getElementById("modalClose");
const transcriptEl = document.getElementById("transcript");

function logLine(kind, text){
  const div = document.createElement("div");
  div.className = "line " + (kind || "meta");
  div.textContent = text;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}
function logUser(t){ logLine("user", "You: " + t); }
function logBot(t){ logLine("bot",  "Tahlia: " + t); }
function logErr(t){ logLine("sys",  "Error: " + t); }
function logMeta(t){ logLine("meta", t); }

function openModal(){ modalBackdrop.classList.add("show"); modalBackdrop.setAttribute("aria-hidden","false"); }
function closeModal(){ modalBackdrop.classList.remove("show"); modalBackdrop.setAttribute("aria-hidden","true"); }
logsBtn.addEventListener("click", openModal);
modalClose.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (e) => { if (e.target === modalBackdrop) closeModal(); });

let rec = null;
let session = false;
let lastBotReply = "";
let introPlayed = false;

let asrState = "idle";
let recIsStarting = false;
let lastASRStartTs = 0;
let asrWatchdog = null;

let assistantSpeaking = false;
let wantsInterrupt = false;
let botSpeakingSince = 0;
const INTERRUPT_GRACE_MS = 700;
const MIN_FINAL_LEN = 1;
let lastSendTs = 0;
const ECHO_WINDOW_MS = 1800;

let audioCtx = null, mediaSrc = null, analyser = null, dataArray = null;
let volEMA = 0;

// ---- Adjacent processing controls ----
let adjTimer = null;
let adjCooldownUntil = 0;
const ADJ_DEBOUNCE_MS = 220;
const ADJ_COOLDOWN_MS = 1800; // match server
const ADJ_MIN_CHARS = 14;

function setupAudioAnalyzer(){
  if (audioCtx) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    mediaSrc = audioCtx.createMediaElementSource(player);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const bufLen = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufLen);
    mediaSrc.connect(analyser);
    mediaSrc.connect(audioCtx.destination);
    animateBall();
  } catch(e){
    logErr("Audio analyzer init failed: " + e);
  }
}

function animateBall(){
  if (!analyser || !dataArray) { requestAnimationFrame(animateBall); return; }
  analyser.getByteTimeDomainData(dataArray);
  let sum = 0;
  for (let i=0; i<dataArray.length; i++){
    const v = (dataArray[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / dataArray.length);
  const target = assistantSpeaking ? rms : 0;
  volEMA = volEMA * 0.85 + target * 0.15;
  const scale = 1 + Math.min(0.9, volEMA * 2.2);
  ball.style.transform = "scale(" + scale.toFixed(3) + ")";
  requestAnimationFrame(animateBall);
}

function words(str){
  return (str || "").toLowerCase().replace(/[^\\w\\s']/g, " ").split(/\\s+/).filter(Boolean);
}
function jaccard(a, b){
  const A = new Set(a), B = new Set(b);
  if (!A.size && !B.size) return 0;
  let inter = 0; for (const x of A) if (B.has(x)) inter++;
  return inter / (A.size + B.size - inter);
}
function likelyEcho(userFinal, botLine){
  if (!botLine) return false;
  const now = Date.now();
  if (!assistantSpeaking || !botSpeakingSince) return false;
  if ((now - botSpeakingSince) > ECHO_WINDOW_MS) return false;
  const ua = words(userFinal), ba = words(botLine);
  const sim = jaccard(ua, ba);
  const short = userFinal.toLowerCase(), bot = botLine.toLowerCase();
  const prefixish = short.length > 6 && (bot.startsWith(short) || short.startsWith(bot));
  return sim >= 0.6 || prefixish;
}

async function startASRSafe(delay = 0) {
  if (!rec || !session) return;
  if (asrState === "running" || asrState === "starting") return;
  asrState = "starting"; recIsStarting = true;
  try {
    if (delay) await new Promise(r => setTimeout(r, delay));
    try { rec.start(); } catch(_) {}
  } finally {
    setTimeout(() => { recIsStarting = false; }, 120);
  }
}
function stopASRSafe() {
  if (!rec) return;
  if (asrState === "stopping" || asrState === "idle") return;
  asrState = "stopping";
  try { rec.stop(); } catch(_) {}
}

function ensureASR(){
  if (rec) return rec;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("SpeechRecognition not supported. Use Chrome."); return null; }
  rec = new SR();
  rec.continuous = true; rec.interimResults = true; rec.lang = "en-US";

  rec.onstart = () => { asrState = "running"; lastASRStartTs = Date.now(); logMeta("ASR started"); };
  rec.onend   = () => {
    if (!session) { asrState = "idle"; return; }
    asrState = "idle"; logMeta("ASR ended → restarting"); startASRSafe(250);
  };
  rec.onerror = (e) => {
    if (!session) return;
    asrState = "idle"; logErr("ASR error: " + (e?.error || "unknown") + " → restarting");
    setTimeout(() => startASRSafe(200), 300);
  };

  rec.onresult = (evt) => {
    let finalText = "";
    let interimBest = "";
    const now = Date.now();

    for (let i = evt.resultIndex; i < evt.results.length; i++) {
      const res = evt.results[i];
      const txt = (res[0]?.transcript || "").trim();

      if (!res.isFinal && txt) {
        // Keep the longest interim we saw in this batch
        if (txt.length > interimBest.length) interimBest = txt;
        const elapsed = now - botSpeakingSince;
        if (assistantSpeaking && botSpeakingSince && elapsed > 700) wantsInterrupt = true;
      }

      if (res.isFinal) finalText += txt + " ";
    }

    // ---- Adjacent processing trigger on confident interim ----
    if (!assistantSpeaking && interimBest && interimBest.length >= ADJ_MIN_CHARS) {
      if (!adjTimer && Date.now() > adjCooldownUntil) {
        adjTimer = setTimeout(() => {
          adjTimer = null;
          fireAdjacent(interimBest);
        }, ADJ_DEBOUNCE_MS);
      }
    }

    finalText = (finalText || "").trim();
    if (!finalText) return;

    if (likelyEcho(finalText, lastBotReply)) { logMeta("Echo filtered"); return; }
    if (assistantSpeaking && !wantsInterrupt) return;
    if (assistantSpeaking && wantsInterrupt) { stopBotAudio(); wantsInterrupt = false; }

    if (finalText.length < 1) return;
    const lw = finalText.toLowerCase();
    if (lastBotReply && lw === lastBotReply.toLowerCase()) return;
    const now2 = Date.now();
    if (now2 - lastSendTs < 220) return;
    lastSendTs = now2;

    logUser(finalText);
    sendToBot(finalText);
  };

  if (asrWatchdog) clearInterval(asrWatchdog);
  asrWatchdog = setInterval(() => {
    if (!session || !rec) return;
    const tooLongSinceStart = Date.now() - lastASRStartTs > 12000;
    if ((asrState !== "running" && !recIsStarting) || tooLongSinceStart) {
      logMeta("ASR watchdog kick"); startASRSafe();
    }
  }, 6000);

  return rec;
}

async function safeJson(resp){
  const text = await resp.text();
  try { return JSON.parse(text); } catch { return { error_text: text }; }
}

function stopBotAudio(){
  try { player.pause(); player.src = ""; player.currentTime = 0; } catch(_) {}
  assistantSpeaking = false;
  botSpeakingSince = 0;
}

async function fireAdjacent(prefix){
  // cooldown window
  adjCooldownUntil = Date.now() + ADJ_COOLDOWN_MS;

  try {
    const r = await fetch("/api/adjacent", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ cid: clientId, prefix })
    });
    const d = await safeJson(r);
    if (d.dbg) logMeta("Server: " + d.dbg);
    if (d.error_text) logErr("Server raw: " + d.error_text);
    if (d.tts_error) logErr("TTS: " + d.tts_error);

    if (d.reply) { lastBotReply = d.reply; logBot(d.reply + " (teaser)"); }
    if (d.audio) {
      try {
        player.pause(); player.src = d.audio;
        assistantSpeaking = true; botSpeakingSince = Date.now();
        setupAudioAnalyzer();
        if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
        await player.play();
        fetch("/api/ack_assistant", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ cid: clientId }) }).catch(()=>{});
      } catch(e){ logErr("Audio play failed: " + e); }
    }
  } catch(e){
    logErr("Fetch /api/adjacent failed: " + e);
  }
}

async function sendToBot(text){
  const lw = (text || "").trim().toLowerCase();

  if (["stop","quit","end","goodbye","end call","terminate"].includes(lw) ||
      (lw.length <= 24 && ["stop","quit","end","goodbye","end call","terminate"].some(k => lw.includes(k)))) {
    session = false; introPlayed = false;
    try { stopASRSafe(); } catch(_){}
    if (asrWatchdog) { clearInterval(asrWatchdog); asrWatchdog = null; }
    stopBotAudio();
    startBtn.textContent = "Start Conversation";
    try {
      const r = await fetch("/api/reply", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ cid: clientId, text }) });
      const d = await safeJson(r);
      if (d.dbg) logMeta("Server: " + d.dbg); if (d.tts_error) logErr("TTS: " + d.tts_error);
    } catch(e){ logErr("Stop send failed: " + e); }
    return;
  }

  if (lastBotReply && lw === lastBotReply.toLowerCase()) return;

  try {
    const r = await fetch("/api/reply", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ cid: clientId, text })
    });
    const d = await safeJson(r);

    if (d.dbg) logMeta("Server: " + d.dbg);
    if (d.error_text) logErr("Server raw: " + d.error_text);
    if (d.tts_error) logErr("TTS: " + d.tts_error);

    if (d.reply && d.reply === lastBotReply) {
      // identical to teaser; do nothing
    } else if (d.reply) {
      lastBotReply = d.reply; logBot(d.reply);
    }

    if (d.audio) {
      try {
        // Replace any teaser currently speaking
        player.pause(); player.src = d.audio;
        assistantSpeaking = true; botSpeakingSince = Date.now();
        setupAudioAnalyzer();
        if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
        await player.play();
        fetch("/api/ack_assistant", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ cid: clientId }) }).catch(()=>{});
      } catch(e){ logErr("Audio play failed: " + e); }
    }
  } catch(e){
    logErr("Fetch /api/reply failed: " + e);
  }
}

player.onpause = () => { assistantSpeaking = false; botSpeakingSince = 0; ball.style.transform = "scale(1)"; };
player.onended = () => { assistantSpeaking = false; botSpeakingSince = 0; ball.style.transform = "scale(1)"; };

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && session) startASRSafe(120);
});
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (session) startASRSafe(180);
  });
}

startBtn.addEventListener("click", async () => {
  if (!session) {
    try {
      await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch(e) { alert("Microphone permission required"); return; }

    const r = ensureASR(); if (!r) return;
    session = true; startBtn.textContent = "■ End Conversation";
    await startASRSafe(80);

    try {
      if (!introPlayed) {
        const resp = await fetch("/api/intro", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ cid: clientId }) });
        const d = await safeJson(resp);
        introPlayed = true;
        if (d.dbg) logMeta("Server: " + d.dbg);
        if (d.reply) { lastBotReply = d.reply; logBot(d.reply); }
        if (d.tts_error) logErr("TTS: " + d.tts_error);
        if (d.audio) {
          try {
            player.pause(); player.src = d.audio;
            assistantSpeaking = true; botSpeakingSince = Date.now();
            setupAudioAnalyzer();
            if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
            await player.play();
            fetch("/api/ack_assistant", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ cid: clientId }) }).catch(()=>{});
          } catch(e){ logErr("Audio play failed: " + e); }
        }
      }
    } catch(e){ logErr("Intro failed: " + e); }
  } else {
    session = false; introPlayed = false; startBtn.textContent = "Start Conversation";
    try { stopASRSafe(); } catch(_){}
    if (asrWatchdog) { clearInterval(asrWatchdog); asrWatchdog = null; }
    try { player.pause(); } catch(_) {}
  }
});
</script>
</body></html>
"""
    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
