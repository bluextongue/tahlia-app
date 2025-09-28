# app.py
# Tahlia: Flask + OpenAI + ElevenLabs
# - Natural therapist vibe (gpt-4o-mini by default; change OPENAI_MODEL to try o4 / o4-mini)
# - Single-intro, no double-speak (server + client ack)
# - Always-on ASR with final-only interrupt + echo filter
# - Faster feel: lower interrupt grace (700ms), quicker ASR send (220ms), smaller TTS
# - Allows deeper responses when appropriate (up to 5 sentences)
# - Transcript/logs: modal panel (button opens, "X" closes; starts closed)
# - Voice-reactive ball: scales with assistant audio volume (and stays audible)
# - Default port 5050

import os
import base64
import re
import time
from collections import deque

from flask import Flask, request, jsonify, make_response

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= YOUR API KEYS (LOCAL ONLY) =========
ELEVEN_API_KEY = "3e7c3a7c14cec12c34324bd0d25a063ae44b3f4c09b1d25ac1dbcd5a606652d8"
ELEVEN_VOICE_ID = "XeomjLZoU5rr4yNIg16w"

OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o-mini"  # try "o4-mini" or "o4" for smarter

ASSISTANT_NAME = "Tahlia"

# ========= HTTP session with retries =========
session = requests.Session()
retry = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=0.35,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods={"GET", "POST"},
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Slightly larger timeouts for hosted environments
DEFAULT_TIMEOUT = (10, 120)

# ========= Flask + short memory =========
app = Flask(__name__)
history = deque(maxlen=16)

# ---- session state flags (server) ----
INTRO_SENT = False
LAST_SPOKE = None           # "user" | "assistant" | None
LAST_REPLY = ""             # last assistant text (to block dupes)
RECENT_ASSIST = deque(maxlen=6)

# ========= Prompts and rules =========
SYSTEM_PROMPT = (
    "You are " + ASSISTANT_NAME + ", a warm, natural, therapist-like conversational partner. "
    "Respond like a thoughtful human: brief reflection first, then one specific, practical tip the user can try now. "
    "Favor concrete help over broad platitudes. Offer a targeted micro-step, example, or tiny script the user could use. "
    "Keep 2-4 sentences by default; go up to 5 when the user shares detail or asks for depth. "
    "Ask at most one short, purposeful question—and only after giving something useful. "
    "Vary your wording; avoid repeating lines or canned phrases. "
    "Steer clear of vague prompts like 'how are you coping' unless the user invites it. "
    "Useful focus areas to select from (pick just one per turn unless asked): "
    "- emotion naming or a 60-second grounding (e.g., 5-4-3-2-1), "
    "- a tiny behavior experiment (next 24h), "
    "- cognitive reframe (spot one thought and try an alternative), "
    "- sleep or body basics (one tweak, not a list), "
    "- boundary or ask-for-help micro-script, "
    "- urge surfing (notice-name-ride), "
    "- planning a smallest-next-step. "
    "Avoid the phrases 'I'm here with you' and 'Let's take it one step at a time.' "
    "Crisis: if the user indicates imminent self-harm, advise calling 911 or contacting/texting 988 (U.S. crisis line) immediately."
)

CRISIS_TRIGGERS = (
    "kill myself", "suicide", "hurt myself", "harm myself", "overdose",
    "end my life", "take my life", "self harm", "self-harm"
)
STOP_WORDS = ("stop", "quit", "end", "goodbye", "end call", "terminate")

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
    if not t:
        return t
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:max_sents]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"\s+\S*$", "", t).rstrip(",;:.!? ")
    return t

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a | b) or 1)

def tokset(s: str) -> set:
    return set(re.findall(r"[a-z']+", (s or "").lower()))

def too_similar(new: str, refs: list[str], threshold: float = 0.90) -> bool:
    A = tokset(new)
    for r in refs:
        if not r:
            continue
        if jaccard(A, tokset(r)) >= threshold:
            return True
    return False

BANNED_PREFIXES = ("i'm here with you", "let's take it one step at a time")

def not_duplicate_user(text: str) -> bool:
    if not history:
        return True
    last = history[-1]
    return not (last[0] == "user" and norm(last[1]) == norm(text))

def not_duplicate_bot(text: str) -> bool:
    return norm(LAST_REPLY) != norm(text)

def ends_with_question(s: str) -> bool:
    return bool(re.search(r"\?\s*$", (s or "")))

def rephrase_local(text: str) -> str:
    """
    Lightweight local variation to avoid exact duplicates without another LLM call.
    Keeps the original guidance but changes framing so it won't be identical.
    """
    if not text:
        return text

    # Trim trailing question to avoid stacking questions
    t = re.sub(r"\?\s*$", ".", text).strip()

    prefixes = [
        "Another small angle you could try now: ",
        "If that doesn't fit, try this tiny step: ",
        "One more low-friction tweak to try: ",
        "A quick variation you can test: ",
    ]
    idx = abs(hash(text)) % len(prefixes)
    out = prefixes[idx] + t
    return concise(out)

# ========= OpenAI Chat =========
def openai_chat(messages, model=OPENAI_MODEL, temperature=0.72, max_tokens=320):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer " + OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

def llm_reply(user_text: str) -> tuple[str, str]:
    # Ensure reply is not a near-duplicate and not equal to LAST_REPLY
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in list(history)[-12:]:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_text})

    def sample(messages, temp=0.6, max_tokens=360):
        try:
            out = openai_chat(messages, temperature=temp, max_tokens=max_tokens)
            if not out:
                return ("Here's one simple thing to try: name what feels most present right now in one word, "
                        "then take a slow 4-count inhale and 6-count exhale. What feels even 5% lighter?")
            return out
        except Exception:
            return ("Here's something concrete to try right now: put a 2-minute timer and write the one problem "
                    "in a single sentence, then underline the part you can influence today. What's the tiniest next step?")

    dbg = "ok"
    reply = concise(sample(msgs))

    # Avoid stacking question endings
    recent_qs = sum(1 for r in list(RECENT_ASSIST)[-2:] if ends_with_question(r))
    if ends_with_question(reply) and recent_qs >= 1:
        msgs.append({"role": "system", "content": "Regenerate the reply without ending in a question. Provide one specific, practical tip the user can try now."})
        reply = concise(sample(msgs))
        dbg = "regen_no_question"

    # Diversity: banned starts or too similar to recent
    low = norm(reply)
    if low.startswith(BANNED_PREFIXES) or too_similar(reply, list(RECENT_ASSIST), threshold=0.90):
        msgs.append({"role": "system", "content": "Regenerate with different wording; be specific to the user's last message. Allow up to 5 sentences if helpful."})
        reply = concise(sample(msgs))
        dbg = "regen_diversity"

    # Avoid duplicating exactly the last bot line
    if norm(reply) == norm(LAST_REPLY):
        reply = concise(rephrase_local(reply))
        dbg = "regen_avoid_last"

    history.append(("user", user_text))
    history.append(("assistant", reply))
    RECENT_ASSIST.append(reply)
    return reply, dbg

# ========= ElevenLabs TTS =========
def tts_b64(text: str):
    if not ELEVEN_API_KEY:
        return "", "Missing ELEVENLABS_API_KEY"
    safe_text = (text or "").strip()
    if not safe_text:
        return "", ""
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
                if i < 2:
                    time.sleep(0.15 * (i + 1))
                    continue
                return "", f"TTS HTTP {r.status_code}: {r.text[:200]}"
            b64 = base64.b64encode(r.content).decode("utf-8")
            return "data:audio/mpeg;base64," + b64, ""
        except Exception as e:
            if i == 2:
                return "", f"TTS exception: {e}"
            time.sleep(0.2 * (i + 1))
    return "", "TTS unknown error"

# ========= API: introduction =========
@app.post("/api/intro")
def api_intro():
    global INTRO_SENT, LAST_SPOKE, LAST_REPLY, RECENT_ASSIST
    if INTRO_SENT:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "double_intro_blocked"}), 200
    intro = "Hey, I'm " + ASSISTANT_NAME + ". Your mental health assistant."
    history.clear(); RECENT_ASSIST.clear()
    INTRO_SENT = True
    LAST_SPOKE = None
    LAST_REPLY = intro
    RECENT_ASSIST.append(intro)
    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro"}), 200

# ========= API: chat reply =========
@app.post("/api/reply")
def api_reply():
    global INTRO_SENT, LAST_SPOKE, LAST_REPLY
    data = request.get_json(force=True, silent=False) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "empty_text"}), 400

    lower_text = norm(user_text)

    # Stop command
    if contains_stop_command(lower_text):
        history.clear()
        INTRO_SENT = False
        LAST_SPOKE = None
        LAST_REPLY = ""
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "stop_word"}), 200

    # Ignore if client accidentally posts the bot's last reply
    if lower_text == norm(LAST_REPLY):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "echo_bot_line"}), 200

    if len(lower_text) < 1:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "too_short"}), 200

    # Mark that the last confirmed speaker is the user
    LAST_SPOKE = "user"

    # prevent duplicate user messages
    if not not_duplicate_user(lower_text):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "duplicate_user"}), 200

    # Crisis path
    if detect_crisis(lower_text):
        reply = concise(
            "I'm really glad you told me. If you're in immediate danger, call 911. "
            "In the U.S., you can call or text 988 for the Suicide & Crisis Lifeline. "
            "Would you like resources now?"
        )
        history.append(("user", user_text)); history.append(("assistant", reply))
        dbg = "crisis"
    else:
        reply, dbg = llm_reply(user_text)

    # If reply duplicates the last bot line, do a local variation (no extra LLM call)
    if norm(reply) == norm(LAST_REPLY):
        varied = rephrase_local(reply)
        if norm(varied) != norm(LAST_REPLY):
            reply = varied
            dbg = "dedup_local_variation"
        else:
            dbg = "dedup_local_passthrough"

    audio, tts_err = tts_b64(reply)
    LAST_REPLY = reply
    return jsonify({"reply": reply, "audio": audio, "tts_error": tts_err, "dbg": dbg}), 200

# ========= API: assistant speech ACK =========
@app.post("/api/ack_assistant")
def api_ack_assistant():
    global LAST_SPOKE
    LAST_SPOKE = "assistant"
    return jsonify({"ok": True})

# ========= API: reset memory =========
@app.post("/api/reset")
def api_reset():
    global INTRO_SENT, LAST_SPOKE, LAST_REPLY, RECENT_ASSIST
    history.clear(); RECENT_ASSIST.clear()
    INTRO_SENT = False
    LAST_SPOKE = None
    LAST_REPLY = ""
    return jsonify({"ok": True})

# ========= Health/ping =========
@app.get("/api/ping")
def api_ping():
    return jsonify({"ok": True, "ts": time.time(), "intro_sent": INTRO_SENT, "last_spoke": LAST_SPOKE})

# ========= UI (client) =========
@app.get("/")
def index():
    html = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Tahlia - Voice Companion</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:#5b5bd6; --border:#1b1f24; }
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
  .user { color:#c9f0ff; } .bot{ color:#d7ffe0; } .sys{ color:#ffd7d7; } .meta{ font-size:12px; color:#9aa4af; }

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
        <button id="modalClose" title="Close">X</button>
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

    // tap the signal for analysis...
    mediaSrc.connect(analyser);
    // ...and ALSO route the actual audio to the speakers (do NOT mute)
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
    asrState = "idle"; logMeta("ASR ended -> restarting"); startASRSafe(250);
  };
  rec.onerror = (e) => {
    if (!session) return;
    asrState = "idle"; logErr("ASR error: " + (e?.error || "unknown") + " -> restarting"); setTimeout(() => startASRSafe(200), 300);
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
    if (now2 - lastSendTs < 220) return;  # faster debounce
    lastSendTs = now2;

    logUser(finalText);
    sendToBot(finalText);
  };

  # Watchdog: every 6s, only restart (and log) if needed; larger threshold to reduce noise
  if (asrWatchdog) clearInterval(asrWatchdog);
  asrWatchdog = setInterval(() => {
    if (!session || !rec) return;
    const tooLongSinceStart = Date.now() - lastASRStartTs > 30000; # was 12000
    const needsRestart = (asrState !== "running" && asrState !== "starting") || tooLongSinceStart;
    if (needsRestart) {
      logMeta("ASR watchdog kick");
      startASRSafe();
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
        const resp = await fetch("/api/intro", { method:"POST" });
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
