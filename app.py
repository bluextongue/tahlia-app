# app.py
# Tahlia: Flask + OpenAI + ElevenLabs
# - Per-user session state (no global collisions)
# - Idempotent intro + /api/begin that always resets and plays intro
# - Always-on ASR with final-only interrupt + echo filter
# - Faster feel: lower interrupt grace (700ms), quicker ASR send (220ms), smaller TTS
# - Allows deeper responses when appropriate (up to 5 sentences)
# - Transcript/logs: modal panel (button opens, "X" closes; starts closed)
# - Voice-reactive ball: scales with assistant audio volume (and stays audible)
# - Default port: from PORT env (Render), fallback 5050

import base64, re, time, random, os
from flask import Flask, request, jsonify, make_response, session as flask_session
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= YOUR API KEYS (LOCAL ONLY) =========
ELEVEN_API_KEY  = "3e7c3a7c14cec12c34324bd0d25a063ae44b3f4c09b1d25ac1dbcd5a606652d8"
ELEVEN_VOICE_ID = "X03mvPuTfprif8QBAVeJ"

OPENAI_API_KEY  = "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"
OPENAI_MODEL    = "gpt-4o-mini"  # try "o4-mini" or "o4" for smarter

ASSISTANT_NAME  = "Tahlia"

# ========= HTTP session with retries =========
session = requests.Session()
retry = Retry(
    total=3, connect=3, read=3, status=3, backoff_factor=0.35,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"], raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
session.mount("https://", adapter); session.mount("http://", adapter)
DEFAULT_TIMEOUT = (5, 55)

# ========= Flask app =========
app = Flask(__name__)
# IMPORTANT: set a real secret in production. For now this enables per-user session state.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "please_change_this_secret_key")

# ========= Per-user session helpers =========
def sget(key, default):
    if key not in flask_session:
        flask_session[key] = default
    return flask_session[key]

def sset(key, value):
    flask_session[key] = value

def append_capped_list(key, item, cap):
    lst = sget(key, [])
    lst.append(item)
    if len(lst) > cap:
        lst = lst[-cap:]
    sset(key, lst)
    return lst

# ========= Prompts and rules =========
SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a warm, natural, therapist-like conversational partner. "
    "Default stance: be genuinely curious and specific to the user's situation—ask focused, non-generic questions that help get to the root of what’s going on. "
    "Keep 2–3 sentences by default (up to 5 if the user shares details). "
    "Use one of these styles per turn (don’t stack them back-to-back): "
    "• Inquire: reflect briefly, then ask 1 precise question that narrows the real issue (e.g., where/when/who/what made it harder). "
    "• Tip (sometimes): offer a tailored, bite-sized suggestion tied to what the user said—no menus of exercises, just one concrete move. "
    "• Story (rarely): share a very short, relatable vignette ('some people find…') to normalize or illustrate, then leave space for them. "
    "Avoid generic platitudes and generic 'how are you coping' unless the user invites it. "
    "Vary your wording; avoid repeating lines. "
    "Avoid the phrases “I’m here with you”, “Let’s take it one step at a time.”, and “Let’s zero in…”. "
    "Crisis: if the user indicates imminent self-harm, advise calling 911 or contacting/texting 988 (U.S. crisis line) immediately."
)

CRISIS_TRIGGERS = (
    "kill myself","suicide","hurt myself","harm myself","overdose",
    "end my life","take my life","self harm","self-harm"
)
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

BANNED_PREFIXES = (
    "i’m here with you", "i'm here with you",
    "let’s take it one step at a time", "let's take it one step at a time",
    "let’s zero in", "let's zero in"
)

def not_duplicate_user(text: str) -> bool:
    hist = sget("history", [])
    if not hist: return True
    last = hist[-1]
    return not (last[0] == "user" and norm(last[1]) == norm(text))

def ends_with_question(s: str) -> bool:
    return bool(re.search(r"\?\s*$", (s or "")))

def choose_style() -> str:
    recent_style = sget("RECENT_STYLE", [])
    last = recent_style[-1] if recent_style else None
    weights = {"inquire": 0.60, "tip": 0.30, "story": 0.10}
    if last in ("tip", "story"):
        weights = {"inquire": 0.75, "tip": 0.18, "story": 0.07}
    r = random.random()
    if r < weights["inquire"]:
        return "inquire"
    elif r < weights["inquire"] + weights["tip"]:
        return "tip"
    return "story"

# ========= OpenAI Chat =========
def openai_chat(messages, model=OPENAI_MODEL, temperature=0.55, max_tokens=360):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

def llm_reply(user_text: str) -> tuple[str, str]:
    hist = sget("history", [])
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in hist[-12:]:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    # Style steering for THIS reply
    style = choose_style()
    if style == "inquire":
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Inquire style. Give a short reflection tied to their words, "
            "then ask ONE precise, non-generic question that helps pinpoint the root cause or a specific blocker. "
            "No exercise suggestions in this turn."
        )})
    elif style == "tip":
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Tip style. Offer ONE tailored, concrete suggestion tightly linked to what they said. "
            "Keep it tiny and situational (no list, no menu of exercises). Optionally end with a short follow-up question."
        )})
    else:  # story
        msgs.append({"role": "system", "content": (
            "For THIS reply: Use the Story style. Share a very brief, relatable vignette ('some people find…') "
            "that normalizes their experience, then invite them back with a soft question or reflection. Keep it 2–3 sentences."
        )})

    dbg = "ok"
    try:
        reply = openai_chat(msgs) or \
                "What part of this hits the hardest for you today? If it helps, name the exact moment it spikes."
    except Exception:
        reply = ("What was the moment today where this felt most intense—who was there and what changed right before it?")
    final = concise(reply)

    # Allow more inquisitiveness; only block if last TWO also ended with questions
    recent_assist = sget("RECENT_ASSIST", [])
    recent_qs = sum(1 for r in recent_assist[-3:] if ends_with_question(r))
    if ends_with_question(final) and recent_qs >= 2:
        msgs.append({"role": "system", "content": "Regenerate without ending in a question. Keep it specific and user-focused."})
        try:
            alt = openai_chat(msgs) or final
            final = concise(alt)
            dbg = "regen_no_question"
        except Exception:
            pass

    low = norm(final)
    if low.startswith(BANNED_PREFIXES) or too_similar(final, recent_assist):
        msgs.append({"role": "system", "content": "Regenerate with different wording; be specific to the user's last message. Allow up to 5 sentences if helpful."})
        try:
            alt = openai_chat(msgs) or final
            final = concise(alt)
            dbg = "regen_diversity"
        except Exception:
            dbg = "ok"

    append_capped_list("RECENT_STYLE", style, 4)
    append_capped_list("history", ("user", user_text), 16)
    append_capped_list("history", ("assistant", final), 16)
    append_capped_list("RECENT_ASSIST", final, 6)
    return final, dbg

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

# ========= API: begin (reset + intro, always works) =========
@app.post("/api/begin")
def api_begin():
    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health companion."

    # reset per-user state
    sset("history", [])
    sset("RECENT_ASSIST", [])
    sset("RECENT_STYLE", [])
    sset("INTRO_SENT", True)
    sset("LAST_SPOKE", None)
    sset("LAST_REPLY", intro)
    sset("LAST_USER_TEXT", "")

    append_capped_list("RECENT_ASSIST", intro, 6)

    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "begin"}), 200

# ========= API: introduction (idempotent) =========
@app.post("/api/intro")
def api_intro():
    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health companion."

    if sget("INTRO_SENT", False):
        # Re-emit intro if already sent; helpful if previous audio didn't autoplay
        if norm(sget("LAST_REPLY", "")) != norm(intro):
            sset("LAST_REPLY", intro)
        audio, tts_err = tts_b64(intro)
        return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro_repeat"}), 200

    # Fresh intro (rare when using /api/begin)
    sset("history", [])
    sset("RECENT_ASSIST", [])
    sset("RECENT_STYLE", [])
    sset("INTRO_SENT", True)
    sset("LAST_SPOKE", None)
    sset("LAST_REPLY", intro)
    sset("LAST_USER_TEXT", "")

    append_capped_list("RECENT_ASSIST", intro, 6)
    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro"}), 200

# ========= API: chat reply =========
@app.post("/api/reply")
def api_reply():
    data = request.get_json(force=True, silent=False) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "empty_text"}), 400

    lower_text = norm(user_text)

    # Stop command
    if contains_stop_command(lower_text):
        sset("history", [])
        sset("RECENT_ASSIST", [])
        sset("RECENT_STYLE", [])
        sset("INTRO_SENT", False)
        sset("LAST_SPOKE", None)
        sset("LAST_REPLY", "")
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "stop_word"}), 200

    # Ignore if client accidentally posts the bot's last reply
    if lower_text == norm(sget("LAST_REPLY", "")):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "echo_bot_line"}), 200

    if len(lower_text) < 1:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "too_short"}), 200

    # prevent duplicate user messages
    if not not_duplicate_user(lower_text):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "duplicate_user"}), 200

    sset("LAST_SPOKE", "user")

    # Crisis path
    if detect_crisis(lower_text):
        reply = concise(
            "I’m really glad you told me. If you’re in immediate danger, call 911. "
            "In the U.S., you can call or text 988 for the Suicide & Crisis Lifeline. "
            "Would you like resources now?"
        )
        append_capped_list("history", ("user", user_text), 16)
        append_capped_list("history", ("assistant", reply), 16)
        dbg = "crisis"
    else:
        # Generate reply normally (no server double-speak guard)
        reply, dbg = llm_reply(user_text)

    # (Removed server 'dedup_bot' drop — always deliver the reply)
    audio, tts_err = tts_b64(reply)
    sset("LAST_REPLY", reply)
    sset("LAST_SPOKE", "assistant")
    return jsonify({"reply": reply, "audio": audio, "tts_error": tts_err, "dbg": dbg}), 200

# ========= API: assistant speech ACK =========
@app.post("/api/ack_assistant")
def api_ack_assistant():
    sset("LAST_SPOKE", "assistant")
    return jsonify({"ok": True})

# ========= API: reset memory =========
@app.post("/api/reset")
def api_reset():
    sset("history", [])
    sset("RECENT_ASSIST", [])
    sset("RECENT_STYLE", [])
    sset("INTRO_SENT", False)
    sset("LAST_SPOKE", None)
    sset("LAST_REPLY", "")
    return jsonify({"ok": True})

# ========= Health/ping =========
@app.get("/api/ping")
def api_ping():
    return jsonify({
        "ok": True,
        "ts": time.time(),
        "intro_sent": sget("INTRO_SENT", False),
        "last_spoke": sget("LAST_SPOKE", None)
    })

# ========= UI (client) =========
@app.get("/")
def index():
    html = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Tahlia – Voice Companion</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:pink; --border:#1b1f24; }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--fg); margin:0; font-family:system-ui, sans-serif; display:flex; height:100vh; width:100vw; overflow:hidden; }

  /* Left column */
  #left { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:24px; min-width:0; }
  #ball { width:120px; height:120px; border-radius:50%; background:var(--accent); transition:transform 0.08s ease; box-shadow:0 10px 30px rgba(0,0,0,.3); }
  #startBtn { padding:14px 20px; border:none; border-radius:12px; background:#161a1f; color:#fff; font-weight:600; font-size:16px; cursor:pointer; box-shadow:0 2px 10px rgba(0,0,0,.25); }
  #logsBtn { padding:10px 14px; border:1px solid var(--border); border-radius:10px; background:#0e1116; color:#cfd8e3; font-weight:600; font-size:13px; cursor:pointer; }

  /* Transcript modal */
  #modalBackdrop { position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; }
  #modal { width:min(720px, 92vw); height:min(70vh, 86vh); background:var(--panel); border:1px solid var(--border); border-radius:12px; display:flex; flex-direction:column; overflow:hidden; }
  #modalHeader { display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border); background:#0e1116; }
  #modalTitle { font-size:14px; color:#9aa4af; }
  #modalClose { border:none; background:transparent; color:#cfd8e3; font-size:18px; cursor:pointer; padding:4px 8px; }
  #transcript { flex:1; overflow:auto; padding:10px; font-size:14px; line-height:1.35; }
  .line { padding:6px 8px; border-bottom:1px dashed #20242a; white-space:pre-wrap; word-break:break-word; }
  .user { color:#c9f0ff; } .bot{ color:#ffd7f0; } .sys{ color:#ffd7d7; } .meta{ font-size:12px; color:#9aa4af; }

  /* Hidden default audio element */
  #player { display:none; }
</style>
</head>
<body>
  <div id="left">
    <div id="ball"></div>
    <button id="startBtn">Start Conversation</button>
    <button id="logsBtn">Transcript & Logs</button>
    <audio id="player" autoplay></audio>
  </div>

  <!-- Modal -->
  <div id="modalBackdrop">
    <div id="modal">
      <div id="modalHeader">
        <div id="modalTitle">Transcript & Logs</div>
        <button id="modalClose" title="Close">✕</button>
      </div>
      <div id="transcript"></div>
    </div>
  </div>

<script>
/* ---------- DOM ---------- */
const player = document.getElementById("player");
const startBtn = document.getElementById("startBtn");
const logsBtn = document.getElementById("logsBtn");
const ball = document.getElementById("ball");
const modalBackdrop = document.getElementById("modalBackdrop");
const modalClose = document.getElementById("modalClose");
const transcriptEl = document.getElementById("transcript");

/* ---------- Logs ---------- */
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

/* ---------- Modal controls ---------- */
function openModal(){ modalBackdrop.style.display = "flex"; }
function closeModal(){ modalBackdrop.style.display = "none"; }
logsBtn.onclick = openModal;
modalClose.onclick = closeModal;
modalBackdrop.addEventListener("click", (e) => { if (e.target === modalBackdrop) closeModal(); });

/* ---------- State ---------- */
let rec = null;
let session = false;
let lastBotReply = "";
let introPlayed = false;

/* ASR state machine */
let asrState = "idle"; // "idle"|"starting"|"running"|"stopping"
let recIsStarting = false;
let lastASRStartTs = 0;
let asrWatchdog = null;

/* Interrupt controls (final-only interrupt + echo filter) */
let assistantSpeaking = false;
let wantsInterrupt = false;
let botSpeakingSince = 0;
const INTERRUPT_GRACE_MS = 700;
const MIN_FINAL_LEN = 1;
let lastSendTs = 0;
const ECHO_WINDOW_MS = 1800;

/* ---------- Voice-reactive ball (assistant audio only) ---------- */
let audioCtx = null, mediaSrc = null, analyser = null, dataArray = null;
let volEMA = 0; // smoothed volume 0..1

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
  const rms = Math.sqrt(sum / dataArray.length); // ~0..1
  const target = assistantSpeaking ? rms : 0;
  volEMA = volEMA * 0.85 + target * 0.15;
  const scale = 1 + Math.min(0.9, volEMA * 2.2);
  ball.style.transform = "scale(" + scale.toFixed(3) + ")";
  requestAnimationFrame(animateBall);
}

/* ---------- Utils ---------- */
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

/* ---------- ASR control ---------- */
async function startASRSafe(delay = 0) {
  if (!rec || !session) return;
  if (asrState === "running" || asrState === "starting") return;
  asrState = "starting"; recIsStarting = true;
  try {
    if (delay) await new Promise(r => setTimeout(r, delay));
    try { rec.start(); } catch(_) {}
    console.debug("ASR start requested");
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
    const now = Date.now();

    for (let i = evt.resultIndex; i < evt.results.length; i++) {
      const res = evt.results[i];
      const txt = (res[0]?.transcript || "").trim();

      if (!res.isFinal && txt) {
        const elapsed = now - botSpeakingSince;
        if (assistantSpeaking && botSpeakingSince && elapsed > INTERRUPT_GRACE_MS) wantsInterrupt = true;
      }

      if (res.isFinal) finalText += txt + " ";
    }

    finalText = (finalText || "").trim();
    if (!finalText) return;

    if (likelyEcho(finalText, lastBotReply)) { logMeta("Echo filtered"); return; }
    if (assistantSpeaking && !wantsInterrupt) return;
    if (assistantSpeaking && wantsInterrupt) { stopBotAudio(); wantsInterrupt = false; }

    if (finalText.length < MIN_FINAL_LEN) return;
    const lw = finalText.toLowerCase();
    if (lastBotReply && lw === lastBotReply.toLowerCase()) return;
    const now2 = Date.now();
    if (now2 - lastSendTs < 220) return;  // faster debounce
    lastSendTs = now2;

    logUser(finalText);
    sendToBot(finalText);
  };

  // Watchdog: every 6s
  if (asrWatchdog) clearInterval(asrWatchdog);
  asrWatchdog = setInterval(() => {
    if (!session || !rec) return;
    const tooLongSinceStart = Date.now() - lastASRStartTs > 12000;
    if ((asrState !== "running" && asrState !== "starting") || tooLongSinceStart) {
      logMeta("ASR watchdog kick"); startASRSafe();
    }
  }, 6000);

  return rec;
}

/* ---------- Networking ---------- */
async function safeJson(resp){
  const text = await resp.text();
  try { return JSON.parse(text); } catch { return { error_text: text }; }
}

function stopBotAudio(){
  try { player.pause(); player.src = ""; player.currentTime = 0; } catch(_) {}
  assistantSpeaking = false;
  botSpeakingSince = 0;
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
      const r = await fetch("/api/reply", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ text }) });
      const d = await safeJson(r);
      if (d.dbg) logMeta("Server: " + d.dbg); if (d.tts_error) logErr("TTS: " + d.tts_error);
    } catch(e){ logErr("Stop send failed: " + e); }
    return;
  }

  if (lastBotReply && lw === lastBotReply.toLowerCase()) return;

  try {
    const r = await fetch("/api/reply", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ text })
    });
    const d = await safeJson(r);

    if (d.dbg) logMeta("Server: " + d.dbg);
    if (d.error_text) logErr("Server raw: " + d.error_text);
    if (d.tts_error) logErr("TTS: " + d.tts_error);

    if (d.reply && d.reply === lastBotReply) return;
    if (d.reply) { lastBotReply = d.reply; logBot(d.reply); }

    if (d.audio) {
      try {
        player.pause(); player.src = d.audio;
        assistantSpeaking = true; botSpeakingSince = Date.now();
        setupAudioAnalyzer();
        if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
        await player.play();
        fetch("/api/ack_assistant", { method:"POST" }).catch(()=>{});
      } catch(e){ logErr("Audio play failed: " + e); }
    }
  } catch(e){
    logErr("Fetch /api/reply failed: " + e);
  }
}

/* ---------- UI events ---------- */
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

startBtn.onclick = async () => {
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
        // Always-safe reset + intro
        const resp = await fetch("/api/begin", { method:"POST" });
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
            fetch("/api/ack_assistant", { method:"POST" }).catch(()=>{});
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
};
</script>
</body></html>
"""
    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

# ========= Run (Render-compatible) =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=True)
