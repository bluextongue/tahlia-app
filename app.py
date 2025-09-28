# app.py
import os, re, time, base64
from flask import Flask, request, jsonify, make_response, session as fsession
import requests

# --- Config ---
ELEVEN_API_KEY  = os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "X03mvPuTfprif8QBAVeJ")

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ASSISTANT_NAME  = os.environ.get("ASSISTANT_NAME", "Tahlia")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "please-change-me")

http = requests.Session()
http.headers.update({"User-Agent": "tahlia/clean-voice-app"})

# --- Helpers ---
def sget(key, default):
    if key not in fsession: fsession[key] = default
    return fsession[key]

def sset(key, val): fsession[key] = val

def append_capped(key, item, cap=16):
    lst = sget(key, [])
    lst.append(item)
    if len(lst) > cap: lst = lst[-cap:]
    sset(key, lst)
    return lst

def concise(text, max_chars=500, max_sents=3):
    t = (text or "").strip()
    if not t: return t
    sents = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sents[:max_sents]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = re.sub(r"\s+\S*$", "", t).rstrip(",;:.!? ")
    return t

# --- Model ---
SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a warm, attentive, therapist-like conversational partner.
Respond directly to what the user just said—reflect briefly and keep it grounded in their words.
Keep replies 1–3 short sentences. Avoid canned scripts.
If the user expresses imminent self-harm: advise calling 911 (U.S.) or 988 immediately.
"""

def openai_chat(msgs):
    if not OPENAI_API_KEY: return "Missing OpenAI key."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": msgs, "temperature": 0.5, "max_tokens": 300}
    r = http.post(url, headers=headers, json=payload, timeout=(10, 60))
    r.raise_for_status()
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

def tts_b64(text):
    if not ELEVEN_API_KEY: return "", ""
    safe = (text or "").strip()
    if not safe: return "", ""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": safe[:650], "model_id": "eleven_multilingual_v2"}
    r = http.post(url, headers=headers, json=payload, timeout=(10,60))
    if r.status_code != 200: return "", f"TTS {r.status_code}"
    b64 = base64.b64encode(r.content).decode()
    return "data:audio/mpeg;base64,"+b64, ""

# --- Routes ---
@app.post("/api/intro")
def api_intro():
    if sget("INTRO_SENT", False):
        return jsonify({"reply":"", "audio":"", "tts_error":"", "dbg":"ok"})
    intro = f"Hey, I'm {ASSISTANT_NAME}. Your mental health assistant."
    sset("INTRO_SENT", True)
    sset("history", [])
    append_capped("history", {"role":"assistant","content":intro})
    audio, tts_err = tts_b64(intro)
    return jsonify({"reply":intro,"audio":audio,"tts_error":tts_err,"dbg":"intro"})

@app.post("/api/reply")
def api_reply():
    data = request.get_json() or {}
    user_text = (data.get("text") or "").strip()
    if not user_text: return jsonify({"reply":"","audio":"","tts_error":"","dbg":"empty"})
    hist = sget("history", [])
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + hist[-12:] + [{"role":"user","content":user_text}]
    try: reply = openai_chat(msgs)
    except: reply = "Sorry—something hiccuped. What should we focus on first?"
    reply = concise(reply)
    append_capped("history", {"role":"user","content":user_text})
    append_capped("history", {"role":"assistant","content":reply})
    audio, tts_err = tts_b64(reply)
    return jsonify({"reply":reply,"audio":audio,"tts_error":tts_err,"dbg":"ok"})

@app.post("/api/reset")
def api_reset():
    sset("history", []); sset("INTRO_SENT", False)
    return jsonify({"ok":True})

@app.get("/api/ping")
def api_ping():
    return jsonify({"ok":True,"ts":time.time()})

@app.get("/")
def index():
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>{ASSISTANT_NAME}</title>
<style>
:root {{ --accent: hotpink; }}
body {{ background:#0b0c10; color:#fff; font-family:sans-serif; display:flex; height:100vh; align-items:center; justify-content:center; flex-direction:column; }}
#ball {{ width:120px; height:120px; border-radius:50%; background:var(--accent); transition:transform .1s; }}
</style></head>
<body>
<div id="ball"></div>
<button id="startBtn">Start Conversation</button>
<audio id="player" autoplay></audio>
<script>
const player=document.getElementById("player"), ball=document.getElementById("ball"), btn=document.getElementById("startBtn");
function logLine(kind,text){{console.log(kind+":",text)}}
btn.onclick=async()=>{{const r=await fetch("/api/intro",{method:"POST"});const d=await r.json();if(d.reply)logLine("bot",d.reply)}}
</script>
</body></html>"""
    return make_response(html,200)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5050))
    app.run(host="0.0.0.0",port=port,debug=False)
