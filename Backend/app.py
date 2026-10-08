import base64
import os
import re

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

# ---------------- Config ----------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MURF_STREAM_URL = "https://global.api.murf.ai/v1/speech/stream"
MAX_TTS_CHARS = 2500  # keep each Murf request safely small

if not GEMINI_API_KEY or not MURF_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or MURF_API_KEY in Backend/.env")

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- Prompts ----------------
PROMPTS = {
    "Summary": """
You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.
Write plain spoken text only: no markdown, no bullet points, no asterisks, no headings.

Respond ONLY in {language}.
""",
    "Detailed": """
You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.
Write plain spoken text only: no markdown, no bullet points, no asterisks, no headings.

Respond ONLY in {language}.
""",
}


# ---------------- Helpers ----------------
def clean_text(text: str) -> str:
    """Remove markdown symbols so the voice doesn't read them out."""
    text = re.sub(r"[*#_`>]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_text(text: str, limit: int = MAX_TTS_CHARS):
    """Split text into chunks (at sentence ends) that fit Murf's limit."""
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= limit:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            # a single very long sentence: hard-split it
            while len(s) > limit:
                chunks.append(s[:limit])
                s = s[limit:]
            current = s
    if current:
        chunks.append(current)
    return chunks


def generate_description(place: str, answer_type: str, language: str) -> str:
    prompt = PROMPTS[answer_type].format(place=place, language=language)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return clean_text(response.text or "")


def murf_tts_chunk(text: str, voice_id: str, locale: str) -> bytes:
    headers = {"api-key": MURF_API_KEY, "Content-Type": "application/json"}
    payload = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO",
    }
    resp = requests.post(
        MURF_STREAM_URL, headers=headers, json=payload, stream=True, timeout=90
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Murf error {resp.status_code}: {resp.text[:300]}")

    audio = b"".join(chunk for chunk in resp.iter_content(chunk_size=8192) if chunk)
    if not audio:
        raise RuntimeError("Murf returned empty audio")
    return audio


def generate_speech(text: str, voice_id: str, locale: str) -> bytes:
    audio = b""
    for part in split_text(text):
        audio += murf_tts_chunk(part, voice_id, locale)
    return audio


# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Travel Guide backend running"})


@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    data = request.get_json(silent=True) or {}

    place = data.get("place")
    answer_type = data.get("answerType")
    language = data.get("language")
    voice_id = data.get("voiceId")
    locale = data.get("locale")

    if not all([place, answer_type, language, voice_id, locale]):
        return jsonify({"error": "Missing required fields"}), 400
    if answer_type not in PROMPTS:
        return jsonify({"error": f"Invalid answerType: {answer_type}"}), 400

    # Step 1: text from Gemini
    try:
        description = generate_description(place, answer_type, language)
        if not description:
            raise RuntimeError("Gemini returned empty text")
    except Exception as e:
        print(f"[Gemini error] {e}")
        return jsonify({"error": "Text generation failed", "details": str(e)}), 500

    # Step 2: audio from Murf (if this fails, still return the text)
    encoded_audio = ""
    audio_error = None
    try:
        audio_bytes = generate_speech(description, voice_id, locale)
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        audio_error = str(e)
        print(f"[Murf error] {e}")

    return jsonify(
        {
            "description": description,
            "audioBase64": encoded_audio,
            "audioError": audio_error,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)