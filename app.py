# app.py
# Tahlia — clean, per-user Flask app (OpenAI + optional ElevenLabs TTS)
# - Per-user session state (no global flags)
# - One-time intro per browser session (quietly ignored if called again)
# - Simple “warm therapist” prompt grounded in user's last message
# - Minimal duplicate-guard (no overzealous dedupe)
# - Render-friendly (uses PORT env), debug off in production
# - Browser UI with Web Speech API + pink reactive ball + transcript

import os, re, time, base64, json
from flask import Flask, request, jsonify, make_response, session as fsession
import requests

# ---------------- Configuration ----------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ELEVEN_API_KEY  = os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "X03mvPuTfprif8QBAVeJ")
ASSISTANT_NAME  = os.environ.get("ASSISTANT_NAME", "Tahlia")

app = Flask(__name__)
# NOTE: In production set this to a secure random string in your Render env.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "please-change-me")

# HTTP session
http = requests.Session()
http.headers.update({"User-Agent": "tahlia/clean-voice-app"})

# ---------------- Small helpers ----------------
def sget(key, default):
    if key not in fsession:
        fsession[key] = default
    return fsession[key]

def sset(key, val):
    fsession[key] = val

def append_capped(key, item, cap=16):
    lst = sget(key, [])
    lst.append(item)
    if len(lst) > cap:
        lst = lst[-cap:]
    sset(key, lst)
    return lst

def concise(text, max_chars=500, max_sents=3):
    t = (text or "").strip()
    if not t:
        return t
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:max_sents]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"\s+\S*$", "", t).rstrip(",;:.!? ")
    return t

# ---------------- Model ----------------
SYSTEM_PROMPT = """
You are %%ASSISTANT_NAME%%, a warm, attentive, therapist-like conversational partner.
Respond directly to what the user just said—reflect briefly and keep it grounded in their words.
Keep replies 1–3 short sentences (4–5 if they share detail). Avoid canned scripts.
If the user expresses imminent self-harm: advise calling 911 (U.S.) or texting/calling 988 immediately.
""".strip()

def openai_chat(msgs):
    if not OPENAI_API_KEY:
        return "I'm not configured with an OpenAI API key."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": msgs, "temperature": 0.5, "max_tokens": 300}
    r = http.post(url, headers=headers, json=payload, timeout=(10, 60))
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or ""

def tts_b64(text):
    """Return (data_uri, err) using ElevenLabs, or ('','') if not configured or text empty."""
    if not ELEVEN_API_KEY:
        return "", ""
    safe = (text or "").strip()
    if not safe:
        return "", ""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {
        "text": safe[:650],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.85, "style": 0.25, "use_speaker_boost": True},
        "output_format": "mp3_22050_64"
    }
    r = http.post(url, headers=headers, json=payload, timeout=(10, 60))
    if r.status_code != 200:
        return "", f"TTS HTTP {r.status_code}"
    b64 = base64.b64encode(r.content).decode("utf-8")
    return "data:audio/mpeg;base64," + b64, ""

# ---------------- API: introduction ----------------
@app.post("/api/intro")
def api_intro():
    # Per-user one-time intro; if already sent in this browser session, no-op
    if sget("INTRO_SENT", False):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "intro_already_sent"}), 200

    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health assistant."

    # Initialize per-user state
    sset("INTRO_SENT", True)
    sset("history", [])
    sset("LAST_REPLY", "")

    # Add intro to history so the next turn is grounded
    append_capped("history", {"role": "assistant", "content": intro}, 16)

    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro"}), 200

# ---------------- API: chat reply ----------------
@app.post("/api/reply")
def api_reply():
    data = request.get_json(force=True, silent=False) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "empty"}), 200

    hist = sget("history", [])
    sys_prompt = SYSTEM_PROMPT.replace("%%ASSISTANT_NAME%%", ASSISTANT_NAME)
    msgs = [{"role": "system", "content": sys_prompt}] + hist[-12:] + [{"role": "user", "content": user_text}]

    try:
        reply = openai_chat(msgs)
    except Exception:
        reply = "Sorry—something hiccuped on my side. What would you like me to focus on?"

    reply = concise(reply, max_chars=500, max_sents=3)

    append_capped("history", {"role": "user", "content": user_text}, 16)
    append_capped("history", {"role": "assistant", "content": reply}, 16)
    sset("LAST_REPLY", reply)

    audio, tts_err = tts_b64(reply)
    return jsonify({"reply": reply, "audio": audio, "tts_error": tts_err, "dbg": "ok"}), 200

# ---------------- API: reset ----------------
@app.post("/api/reset")
def api_reset():
    sset("history", [])
    sset("INTRO_SENT", False)
    sset("LAST_REPLY", "")
    return jsonify({"ok": True})

# ---------------- Health ----------------
@app.get("/api/ping")
def api_ping():
    return jsonify({"ok": True, "ts": time.time(), "intro_sent": sget("INTRO_SENT", False)})

# ---------------- UI ----------------
@app.get("/")
def index():
    # NOTE: This is a plain triple-quoted string (NOT an f-string), so JS braces won’t break Python.
    html = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>%%ASSISTANT_NAME%% – Voice Companion</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root { --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:hotpink; --border:#1b1f24; }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--fg); margin:0; font-family:system-ui, sans-serif; display:flex; height:100vh; width:100vw; overflow:hidden; }
  #left { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:24px; min-width:0; padding:16px; }
  #ball { width:120px; height:120px; border-radius:50%; background:var(--accent); transition:transform 0.08s ease; box-shadow:0 10px 30px rgba(0,0,0,.3); }
  #startBtn { padding:14px 20px; border:none; border-radius:12px; background:#161a1f; color:#fff; font-weight:600; font-size:16px; cursor:pointer; box-shadow:0 2px 10px rgba(0,0,0,.25); }
  #logsBtn { padding:10px 14px; border:1px solid var(--border); border-radius:10px; background:#0e1116; color:#cfd8e3; font-weight:600; font-size:13px; cursor:pointer; }
  #modalBackdrop { position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; }
  #modal { width:min(740px, 94vw); height:min(76vh, 86vh); background:var(--panel); border:1px solid var(--border); border-radius:12px; display:flex; flex-direction:column; overflow:hidden; }
  #modalHeader { display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border); background:#0e1116; }
  #modalTitle { font-size:14px; color:#9aa4af; }
  #modalClose { border:none; background:transparent; color:#cfd8e3; font-size:18px; cursor:pointer; padding:4px 8px; }
  #transcript { flex:1; overflow:auto; padding:10px; font-size:14px; line-height:1.35; }
  .line { padding:6px 8px; border-bottom:1px dashed #20242a; white-space:pre-wrap; word-break:break-word; }
  .user { color:#c9f0ff; } .bot{ color:#d7ffe0; } .sys{ color:#ffd7d7; } .meta{ font-size:12px; color:#9aa4af; }
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
/* --- Assistant name from server (no templating braces in Python) --- */
const ASSISTANT_NAME = %%ASSISTANT_JSON_NAME%%;

/* --- DOM --- */
const player = document.getElementById("player");
const startBtn = document.getElementById("startBtn");
const logsBtn = document.getElementById("logsBtn");
const ball = document.getElementById("ball");
const modalBackdrop = document.getElementById("modalBackdrop");
const modalClose = document.getElementById("modalClose");
const transcriptEl = document.getElementById("transcript");

/* --- Logs --- */
function logLine(kind, text){
  const d = document.createElement("div");
  d.className = "line " + (kind || "meta");
  d.textContent = text;
  transcriptEl.appendChild(d);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}
const logUser = (t) => logLine("user", "You: " + t);
const logBot  = (t) => logLine("bot",  ASSISTANT_NAME + ": " + t);
const logErr  = (t) => logLine("sys",  "Error: " + t);
const logMeta = (t) => logLine("meta", t);

/* --- Modal controls --- */
const openModal = () => { modalBackdrop.style.display = "flex"; };
const closeModal = () => { modalBackdrop.style.display = "none"; };
logsBtn.onclick = openModal;
modalClose.onclick = closeModal;
modalBackdrop.addEventListener("click", (e) => { if (e.target === modalBackdrop) closeModal(); });

/* --- State --- */
let rec = null;
let session = false;
let introDone = false;
let lastBotReply = "";

/* --- Voice-reactive ball (assistant audio only) --- */
let audioCtx = null, mediaSrc = null, analyser = null, dataArray = null, volEMA = 0;
function setupAnalyzer(){
  if (audioCtx) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    mediaSrc = audioCtx.createMediaElementSource(player);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    mediaSrc.connect(analyser);
    mediaSrc.connect(audioCtx.destination);
    animateBall();
  } catch(e){ logErr("Audio analyzer init failed: " + e); }
}
function animateBall(){
  if (!analyser || !dataArray) { requestAnimationFrame(animateBall); return; }
  analyser.getByteTimeDomainData(dataArray);
  let sum = 0;
  for (let i=0;i<dataArray.length;i++){
    const v = (dataArray[i]-128)/128;
    sum += v*v;
  }
  const rms = Math.sqrt(sum / dataArray.length);
  const target = player && !player.paused ? rms : 0;
  volEMA = volEMA * 0.85 + target * 0.15;
  const scale = 1 + Math.min(0.9, volEMA * 2.2);
  ball.style.transform = "scale(" + scale.toFixed(3) + ")";
  requestAnimationFrame(animateBall);
}

/* --- ASR (simple, resilient) --- */
function ensureASR(){
  if (rec) return rec;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("SpeechRecognition not supported. Use Chrome."); return null; }
  rec = new SR();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "en-US";

  rec.onstart = () => logMeta("ASR started");
  rec.onerror = (e) => {
    // no-speech happens often on quiet start; treat softly
    if (e?.error === "no-speech") { logMeta("ASR: no-speech"); return; }
    logErr("ASR error: " + (e?.error || "unknown"));
  };
  rec.onend = () => {
    if (session) {
      logMeta("ASR restarting");
      try { rec.start(); } catch(_) {}
    }
  };
  rec.onresult = (evt) => {
    let finalText = "";
    for (let i = evt.resultIndex; i < evt.results.length; i++){
      const res = evt.results[i];
      const txt = (res[0]?.transcript || "").trim();
      if (res.isFinal && txt) finalText += txt + " ";
    }
    finalText = (finalText || "").trim();
    if (!finalText) return;
    if (lastBotReply && finalText.toLowerCase() === lastBotReply.toLowerCase()) return;
    logUser(finalText);
    sendToBot(finalText);
  };
  return rec;
}

/* --- Networking --- */
async function safeJson(resp){
  const text = await resp.text();
  try { return JSON.parse(text); } catch { return { error_text: text }; }
}

async function sendToBot(text){
  try{
    const r = await fetch("/api/reply", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text })
    });
    const d = await safeJson(r);
    if (d.dbg) logMeta("Server: " + d.dbg);
    if (d.error_text) logErr("Server raw: " + d.error_text);
    if (d.tts_error) logErr("TTS: " + d.tts_error);

    if (d.reply) { lastBotReply = d.reply; logBot(d.reply); }
    if (d.audio) {
      try {
        player.pause(); player.src = d.audio;
        setupAnalyzer();
        if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
        await player.play();
      } catch(e){ logErr("Audio play failed: " + e); }
    }
  } catch(e){
    logErr("Fetch /api/reply failed: " + e);
  }
}

/* --- UI --- */
player.onpause = () => { ball.style.transform = "scale(1)"; };
player.onended = () => { ball.style.transform = "scale(1)"; };

startBtn.onclick = async () => {
  if (!session) {
    // Ask mic permission early to reduce ASR errors
    try {
      await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation:true, noiseSuppression:true, autoGainControl:true }
      });
    } catch(e) {
      alert("Microphone permission required"); return;
    }

    const r = ensureASR(); if (!r) return;
    session = True; // flag our local state
    startBtn.textContent = "■ End Conversation";
    try { rec.start(); } catch(_) {}

    // One-time intro per page session
    if (!introDone) {
      const resp = await fetch("/api/intro", { method:"POST" });
      const d = await safeJson(resp);
      introDone = true;
      if (d.dbg) logMeta("Server: " + d.dbg);
      if (d.reply) { lastBotReply = d.reply; logBot(d.reply); }
      if (d.tts_error) logErr("TTS: " + d.tts_error);
      if (d.audio) {
        try {
          player.pause(); player.src = d.audio;
          setupAnalyzer(); if (audioCtx?.state === "suspended") { try { audioCtx.resume(); } catch(_){} }
          await player.play();
        } catch(e){ logErr("Audio play failed: " + e); }
      }
    }
  } else {
    session = false; startBtn.textContent = "Start Conversation";
    try { rec && rec.stop(); } catch(_) {}
    try { player.pause(); player.src = ""; } catch(_) {}
  }
};
</script>
</body></html>
"""
    # Safely inject assistant name (and JSON-escaped version) without using f-strings in the big block
    html = html.replace("%%ASSISTANT_NAME%%", ASSISTANT_NAME)
    html = html.replace("%%ASSISTANT_JSON_NAME%%", json.dumps(ASSISTANT_NAME))
    return make_response(html, 200)

# ---------------- Run (Render-compatible) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
