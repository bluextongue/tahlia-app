# app.py
# Tahlia — clean, per-user Flask app (OpenAI + optional ElevenLabs TTS)
# - Per-user session state (no global flags)
# - One-time intro per browser session (quietly ignored if called again)
# - Simple “warm therapist” prompt that responds to what the user just said
# - Minimal duplicate-guard
# - Render-friendly (uses PORT env), debug off
# - Simple browser UI with Web Speech API + pink reactive ball

import os, re, time, base64
from flask import Flask, request, jsonify, make_response, session as fsession
import requests

# ---------- Config (env overrides are honored on Render) ----------
ELEVEN_API_KEY  = os.environ.get("ELEVEN_API_KEY",  "3e7c3a7c14cec12c34324bd0d25a063ae44b3f4c09b1d25ac1dbcd5a606652d8")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "X03mvPuTfprif8QBAVeJ")

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "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")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL",    "gpt-4o-mini")
ASSISTANT_NAME  = os.environ.get("ASSISTANT_NAME",  "Tahlia")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")  # allow proxy if needed

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "please-change-me")

# HTTP session
http = requests.Session()
http.headers.update({"User-Agent": "tahlia/clean-voice-app/1.0"})

# ---------- Small helpers ----------
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

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

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

# ---------- Model ----------
SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a warm, attentive, therapist-like conversational partner.
Respond to what the user just said—reflect briefly, stay grounded in their words.
Be natural and conversational. Keep 1–3 short sentences (4–5 if they share details).
Avoid canned lines and generic therapy scripts.
If the user expresses imminent self-harm: advise calling emergency services (911 in the U.S.) or texting/calling 988 immediately.
""".strip()

def openai_chat(history_msgs, temperature=0.5, max_tokens=300):
    """
    Returns (reply_text, dbg_string). Never raises. Includes HTTP code/body snippet in dbg on failure.
    """
    if not OPENAI_API_KEY:
        return "", "openai_missing_api_key"

    url = f"{OPENAI_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": history_msgs,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        r = http.post(url, headers=headers, json=payload, timeout=(12, 60))
    except Exception as e:
        return "", f"openai_request_exc:{type(e).__name__}"

    if r.status_code != 200:
        # surface a short snippet so you see 401/429/etc. in your Live Tail “Server:” line
        snippet = (r.text or "")[:200].replace("\n", " ")
        return "", f"openai_http_{r.status_code}:{snippet}"

    try:
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            return "", "openai_empty_choice"
        return text, "ok"
    except Exception as e:
        return "", f"openai_parse_exc:{type(e).__name__}"

def tts_b64(text: str):
    """Return (data: URI base64 MP3, tts_err). Empty string pair if no key or no text."""
    if not ELEVEN_API_KEY:
        return "", ""
    safe = (text or "").strip()
    if not safe:
        return "", ""
    safe = safe[:650]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {
        "text": safe,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.85,
            "style": 0.25,
            "use_speaker_boost": True
        },
        "output_format": "mp3_22050_64"
    }
    try:
        r = http.post(url, headers=headers, json=payload, timeout=(10, 60))
        if r.status_code != 200:
            return "", f"TTS HTTP {r.status_code}"
        b64 = base64.b64encode(r.content).decode("utf-8")
        return "data:audio/mpeg;base64," + b64, ""
    except Exception as e:
        return "", f"TTS exception: {type(e).__name__}"

# ---------- API: introduction ----------
@app.post("/api/intro")
def api_intro():
    # Per-user one-time intro; if already sent, return a no-op (don’t error)
    if sget("INTRO_SENT", False):
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "intro_already_sent"}), 200

    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health assistant."
    sset("INTRO_SENT", True)
    sset("history", [])
    sset("LAST_REPLY", "")

    # Add intro to history (so the next turn is grounded but won’t be echoed back)
    append_capped("history", {"role": "assistant", "content": intro}, 16)

    audio, tts_err = tts_b64(intro)
    return jsonify({"reply": intro, "audio": audio, "tts_error": tts_err, "dbg": "intro"}), 200

# ---------- API: chat reply ----------
@app.post("/api/reply")
def api_reply():
    data = request.get_json(force=True, silent=False) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "tts_error": "", "dbg": "empty"}), 200

    # Build per-user chat messages (system + short rolling history + latest user)
    hist = sget("history", [])
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + hist[-12:] + [{"role": "user", "content": user_text}]

    # Call model
    reply, model_dbg = openai_chat(msgs)
    if not reply:
        # Show *why* it failed in Live Tail; still return a polite line to the user
        polite = "Sorry—something hiccuped on my side. What did you want me to focus on first?"
        append_capped("history", {"role": "user", "content": user_text}, 16)
        append_capped("history", {"role": "assistant", "content": polite}, 16)
        sset("LAST_REPLY", polite)
        audio, tts_err = tts_b64(polite)
        return jsonify({"reply": polite, "audio": audio, "tts_error": tts_err, "dbg": model_dbg}), 200

    reply = concise(reply, max_chars=500, max_sents=3)

    # Update per-user history
    append_capped("history", {"role": "user", "content": user_text}, 16)
    append_capped("history", {"role": "assistant", "content": reply}, 16)
    sset("LAST_REPLY", reply)

    # TTS (optional)
    audio, tts_err = tts_b64(reply)
    return jsonify({"reply": reply, "audio": audio, "tts_error": tts_err, "dbg": model_dbg}), 200

# ---------- API: reset ----------
@app.post("/api/reset")
def api_reset():
    sset("history", [])
    sset("INTRO_SENT", False)
    sset("LAST_REPLY", "")
    return jsonify({"ok": True})

# ---------- Health ----------
@app.get("/api/ping")
def api_ping():
    return jsonify({"ok": True, "ts": time.time(), "intro_sent": sget("INTRO_SENT", False)})

# ---------- UI ----------
@app.get("/")
def index():
    html = f"""
<!doctype html><html><head><meta charset="utf-8"/>
<title>{ASSISTANT_NAME} – Voice Companion</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {{ --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:hotpink; --border:#1b1f24; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--fg); margin:0; font-family:system-ui, sans-serif; display:flex; height:100vh; width:100vw; overflow:hidden; }}
  #left {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:24px; min-width:0; }}
  #ball {{ width:120px; height:120px; border-radius:50%; background:var(--accent); transition:transform 0.08s ease; box-shadow:0 10px 30px rgba(0,0,0,.3); }}
  #startBtn {{ padding:14px 20px; border:none; border-radius:12px; background:#161a1f; color:#fff; font-weight:600; font-size:16px; cursor:pointer; box-shadow:0 2px 10px rgba(0,0,0,.25); }}
  #logsBtn {{ padding:10px 14px; border:1px solid var(--border); border-radius:10px; background:#0e1116; color:#cfd8e3; font-weight:600; font-size:13px; cursor:pointer; }}
  #modalBackdrop {{ position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; }}
  #modal {{ width:min(720px, 92vw); height:min(70vh, 86vh); background:var(--panel); border:1px solid var(--border); border-radius:12px; display:flex; flex-direction:column; overflow:hidden; }}
  #modalHeader {{ display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--border); background:#0e1116; }}
  #modalTitle {{ font-size:14px; color:#9aa4af; }}
  #modalClose {{ border:none; background:transparent; color:#cfd8e3; font-size:18px; cursor:pointer; padding:4px 8px; }}
  #transcript {{ flex:1; overflow:auto; padding:10px; font-size:14px; line-height:1.35; }}
  .line {{ padding:6px 8px; border-bottom:1px dashed #20242a; white-space:pre-wrap; word-break:break-word; }}
  .user {{ color:#c9f0ff; }} .bot{{ color:#d7ffe0; }} .sys{{ color:#ffd7d7; }} .meta{{ font-size:12px; color:#9aa4af; }}
  #player {{ display:none; }}
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
const player = document.getElementById("player");
const startBtn = document.getElementById("startBtn");
const logsBtn = document.getElementById("logsBtn");
const ball = document.getElementById("ball");
const modalBackdrop = document.getElementById("modalBackdrop");
const modalClose = document.getElementById("modalClose");
const transcriptEl = document.getElementById("transcript");

/* ---------- Logs ---------- */
function logLine(kind, text) {{
  const d = document.createElement("div");
  d.className = "line " + (kind || "meta");
  d.textContent = text;
  transcriptEl.appendChild(d);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}}
const logUser = (t) => logLine("user", "You: " + t);
const logBot  = (t) => logLine("bot", "{ASSISTANT_NAME}: " + t);
const logErr  = (t) => logLine("sys", "Error: " + t);
const logMeta = (t) => logLine("meta", t);

/* ---------- Modal ---------- */
function openModal() {{ modalBackdrop.style.display = "flex"; }}
function closeModal() {{ modalBackdrop.style.display = "none"; }}
logsBtn.onclick = openModal; modalClose.onclick = closeModal;
modalBackdrop.addEventListener("click", (e) => {{ if (e.target === modalBackdrop) closeModal(); }});

/* ---------- State ---------- */
let rec = null, session = false, introDone = false, lastBotReply = "";
let assistantSpeaking = false, botSpeakingSince = 0;

/* ---------- Reactive ball (assistant audio only) ---------- */
let audioCtx = null, mediaSrc = null, analyser = null, dataArray = null, volEMA = 0;
function setupAnalyzer() {{
  if (audioCtx) return;
  try {{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    mediaSrc = audioCtx.createMediaElementSource(player);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    mediaSrc.connect(analyser); mediaSrc.connect(audioCtx.destination);
    animateBall();
  }} catch(e) {{ logErr("Audio analyzer init failed: " + e); }}
}}
function animateBall() {{
  if (!analyser || !dataArray) {{ requestAnimationFrame(animateBall); return; }}
  analyser.getByteTimeDomainData(dataArray);
  let sum = 0; for (let i = 0; i < dataArray.length; i++) {{ const v = (dataArray[i] - 128) / 128; sum += v*v; }}
  const rms = Math.sqrt(sum / dataArray.length);
  const target = assistantSpeaking ? rms : 0;
  volEMA = volEMA * 0.85 + target * 0.15;
  const scale = 1 + Math.min(0.9, volEMA * 2.2);
  ball.style.transform = "scale(" + scale.toFixed(3) + ")";
  requestAnimationFrame(animateBall);
}}

/* ---------- ASR (very simple) ---------- */
function ensureASR() {{
  if (rec) return rec;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {{ alert("SpeechRecognition not supported. Use Chrome."); return null; }}
  rec = new SR(); rec.continuous = true; rec.interimResults = true; rec.lang = "en-US";

  rec.onresult = (evt) => {{
    let finalText = "";
    for (let i = evt.resultIndex; i < evt.results.length; i++) {{
      const res = evt.results[i], txt = (res[0]?.transcript || "").trim();
      if (res.isFinal && txt) finalText += txt + " ";
    }}
    finalText = (finalText || "").trim();
    if (!finalText) return;
    if (lastBotReply && finalText.toLowerCase() === lastBotReply.toLowerCase()) return;
    logUser(finalText);
    sendToBot(finalText);
  }};

  rec.onstart = () => logMeta("ASR started");
  rec.onerror = (e) => logErr("ASR error: " + (e?.error || "unknown"));
  rec.onend   = () => {{ if (session) {{ setTimeout(() => {{ try {{ rec.start(); }} catch(_ ) {{}} }}, 250); logMeta("ASR restart"); }} }};
  return rec;
}}

/* ---------- Networking ---------- */
async function safeJson(resp) {{
  const t = await resp.text();
  try {{ return JSON.parse(t); }} catch {{ return {{ error_text: t }}; }}
}}

function stopBotAudio() {{
  try {{ player.pause(); player.src = ""; player.currentTime = 0; }} catch(_ ) {{}}
  assistantSpeaking = false; botSpeakingSince = 0;
}}

async function sendToBot(text) {{
  try {{
    const r = await fetch("/api/reply", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ text }})
    }});
    const d = await safeJson(r);
    if (d.dbg) logMeta("Server: " + d.dbg);
    if (d.error_text) logErr("Server raw: " + d.error_text);
    if (d.tts_error) logErr("TTS: " + d.tts_error);
    if (d.reply) {{ lastBotReply = d.reply; logBot(d.reply); }}
    if (d.audio) {{
      try {{
        player.pause(); player.src = d.audio; assistantSpeaking = true; botSpeakingSince = Date.now();
        setupAnalyzer(); if (audioCtx?.state === "suspended") {{ try {{ audioCtx.resume(); }} catch(_ ) {{}} }}
        await player.play();
      }} catch(e) {{ logErr("Audio play failed: " + e); }}
    }}
  }} catch(e) {{
    logErr("Fetch /api/reply failed: " + e);
  }}
}}

/* ---------- UI ---------- */
player.onpause = () => {{ assistantSpeaking = false; botSpeakingSince = 0; ball.style.transform = "scale(1)"; }};
player.onended = () => {{ assistantSpeaking = false; botSpeakingSince = 0; ball.style.transform = "scale(1)"; }};

startBtn.onclick = async () => {{
  if (!session) {{
    try {{
      await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }} }});
    }} catch(e) {{ alert("Microphone permission required"); return; }}

    const r = ensureASR(); if (!r) return;
    session = true; startBtn.textContent = "■ End Conversation";
    try {{ rec.start(); }} catch(_ ) {{}}
    // One-time intro per page load
    if (!introDone) {{
      const resp = await fetch("/api/intro", {{ method: "POST" }});
      const d = await safeJson(resp); introDone = true;
      if (d.dbg) logMeta("Server: " + d.dbg);
      if (d.reply) {{ lastBotReply = d.reply; logBot(d.reply); }}
      if (d.tts_error) logErr("TTS: " + d.tts_error);
      if (d.audio) {{
        try {{
          player.pause(); player.src = d.audio; assistantSpeaking = true; botSpeakingSince = Date.now();
          setupAnalyzer(); if (audioCtx?.state === "suspended") {{ try {{ audioCtx.resume(); }} catch(_ ) {{}} }}
          await player.play();
        }} catch(e) {{ logErr("Audio play failed: " + e); }}
      }}
    }}
  }} else {{
    session = false; startBtn.textContent = "Start Conversation";
    try {{ rec && rec.stop(); }} catch(_ ) {{}}
    stopBotAudio();
  }}
}};
</script>
</body></html>
"""
    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

# ---------- Run (Render-compatible) ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
