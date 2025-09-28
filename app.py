# app.py
# Minimal Flask Chat + Voice (OpenAI + ElevenLabs) for Render
# - Hardwired keys (as requested)
# - Simple UI: Start/Stop, transcript, audio playback
# - Chrome SpeechRecognition (final results only)
# - Render-friendly (binds 0.0.0.0, PORT default 5050)
# - /api/test_openai to validate connectivity on Render

import os
import base64
import re
import time
from flask import Flask, request, jsonify, make_response
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= YOUR API KEYS (LOCAL ONLY) =========
ELEVEN_API_KEY = "3e7c3a7c14cec12c34324bd0d25a063ae44b3f4c09b1d25ac1dbcd5a606652d8"
ELEVEN_VOICE_ID = "XeomjLZoU5rr4yNIg16w"

OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o-mini"

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

# Larger timeouts for hosted envs
DEFAULT_TIMEOUT = (10, 120)

# ========= Flask app =========
app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are " + ASSISTANT_NAME + ", a warm, practical, therapist-like conversational partner. "
    "Give a brief reflection first, then one specific tip the user can try now. "
    "Prefer 2-4 sentences. Be concrete, no filler."
)

def concise(text, max_chars=520, max_sents=5):
    t = (text or "").strip()
    if not t:
        return t
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:max_sents]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"\s+\S*$", "", t).rstrip(",;:.!? ")
    return t

def openai_chat(user_text: str) -> str:
    # Call Chat Completions
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer " + OPENAI_API_KEY, "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.6, "max_tokens": 320}
    r = session.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    if r.status_code >= 400:
        # Include a short snippet so it's visible in /api/reply debug
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    reply = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return concise(reply)

def tts_b64(text: str):
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

# ========= API: OpenAI connectivity quick test =========
@app.get("/api/test_openai")
def api_test_openai():
    try:
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": "Bearer " + OPENAI_API_KEY}
        r = session.get(url, headers=headers, timeout=(10, 30))
        ok = 200 <= r.status_code < 300
        return jsonify({
            "ok": ok,
            "status": r.status_code,
            "body_head": (r.text or "")[:200]
        }), (200 if ok else 500)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500

# ========= API: chat =========
@app.post("/api/reply")
def api_reply():
    data = request.get_json(force=True, silent=False) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"reply": "", "audio": "", "err": "empty_text", "dbg": {"why": "no_user_text"}}), 400

    dbg = {"openai": "ok"}
    try:
        reply = openai_chat(user_text)
        if not reply:
            reply = "Try a tiny step: inhale for 4, exhale for 6, three rounds. Then jot one smallest next action."
            dbg["openai"] = "empty_reply_fallback"
    except Exception as e:
        reply = (
            "Quick grounding: inhale 4 / exhale 6 for 4 rounds. "
            "Then name one tiny next step that reduces friction by 5%."
        )
        dbg["openai"] = f"error: {str(e)[:200]}"

    audio, tts_err = tts_b64(reply)
    if tts_err:
        dbg["tts"] = tts_err
    return jsonify({"reply": reply, "audio": audio, "err": tts_err or "", "dbg": dbg}), 200

# ========= UI =========
@app.get("/")
def index():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Tahlia - Minimal Chat & Voice</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { --bg:#0b0c10; --fg:#eaf2f8; --panel:#111417; --accent:#5b5bd6; --border:#1b1f24; }
    * { box-sizing:border-box; }
    body { background:var(--bg); color:var(--fg); margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    .wrap { max-width:800px; margin:0 auto; padding:24px; }
    h1 { margin:0 0 12px; font-size:20px; }
    .row { display:flex; gap:12px; align-items:center; margin:12px 0; }
    button { padding:12px 16px; border:none; border-radius:10px; background:#1a1f24; color:#fff; font-weight:600; cursor:pointer; }
    button[disabled] { opacity:.6; cursor:not-allowed; }
    #status { font-size:13px; color:#9aa4af; }
    #log { margin-top:16px; padding:12px; background:var(--panel); border:1px solid var(--border); border-radius:10px; min-height:200px; white-space:pre-wrap; }
    #player { display:none; }
    .info { font-size:13px; color:#9aa4af; margin-top:10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Tahlia - Minimal Chat & Voice</h1>
    <div class="row">
      <button id="startBtn">Start</button>
      <button id="stopBtn" disabled>Stop</button>
      <span id="status">idle</span>
    </div>
    <div class="info">
      Tip: On Render, open <code>/api/test_openai</code> to verify connectivity.
    </div>
    <div id="log"></div>
    <audio id="player" autoplay></audio>
  </div>

<script>
(function(){
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const statusEl = document.getElementById("status");
  const logEl = document.getElementById("log");
  const player = document.getElementById("player");

  let rec = null;
  let sessionOn = false;

  function log(line){
    logEl.textContent += (line + "\\n");
    logEl.scrollTop = logEl.scrollHeight;
  }
  function setStatus(t){ statusEl.textContent = t; }

  function ensureASR(){
    if (rec) return rec;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      alert("SpeechRecognition not supported. Use Chrome.");
      return null;
    }
    rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";

    rec.onstart = () => { setStatus("listening"); };
    rec.onend = () => {
      if (!sessionOn) { setStatus("idle"); return; }
      setStatus("restarting...");
      try { rec.start(); } catch(_) {}
    };
    rec.onerror = (e) => {
      log("ASR error: " + (e && e.error ? e.error : "unknown"));
      if (!sessionOn) return;
      setTimeout(() => { try { rec.start(); } catch(_) {} }, 400);
    };
    rec.onresult = (evt) => {
      let finalText = "";
      for (let i = evt.resultIndex; i < evt.results.length; i++) {
        const res = evt.results[i];
        const txt = (res[0] && res[0].transcript ? res[0].transcript.trim() : "");
        if (res.isFinal && txt) finalText += txt + " ";
      }
      finalText = (finalText || "").trim();
      if (!finalText) return;
      log("You: " + finalText);
      sendToBot(finalText);
    };
    return rec;
  }

  async function sendToBot(text){
    try {
      const r = await fetch("/api/reply", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ text })
      });
      const d = await r.json();
      if (d.dbg) { try { log("DBG: " + JSON.stringify(d.dbg)); } catch(_) {} }
      if (d.err) log("Note: " + d.err);
      if (d.reply) log("Tahlia: " + d.reply);
      if (d.audio) {
        try {
          player.pause();
          player.src = d.audio;
          await player.play();
        } catch (e) {
          log("Audio play failed: " + e);
        }
      }
    } catch(e){
      log("Request failed: " + e);
    }
  }

  startBtn.onclick = async () => {
    if (sessionOn) return;
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch(e) { alert("Microphone permission required"); return; }

    const r = ensureASR();
    if (!r) return;

    sessionOn = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("starting...");
    try { r.start(); } catch(_) {}
  };

  stopBtn.onclick = () => {
    sessionOn = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("idle");
    try { rec && rec.stop(); } catch(_) {}
  };
})();
</script>
</body>
</html>
"""
    resp = make_response(html, 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
