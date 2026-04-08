import os
import json
import tempfile
from flask import Flask, request, jsonify
from groq import Groq
import anthropic

app = Flask(__name__)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def clean_and_parse(text):
    """Strip markdown code blocks and parse JSON from Claude response."""
    raw = text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    if not raw.startswith("{"):
        raw = "{" + raw
    return json.loads(raw)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/test", methods=["GET"])
def test_summarise():
    transcript = "I need to call John tomorrow about the budget, he mentioned we might be over by 20% and I want to discuss options before the Friday meeting."
    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are a personal assistant that processes voice notes.

Given this voice note transcript, respond with ONLY valid JSON — no markdown, no code blocks, no explanation.

Transcript:
{transcript}

JSON format:
{{
  "title": "concise title under 60 characters",
  "summary": "2-3 sentences that synthesize and condense the core ideas, decisions, or action items. Do NOT repeat or quote the transcript — interpret and distill it."
}}"""
            }
        ]
    )
    try:
        result = clean_and_parse(message.content[0].text)
        result["transcript"] = transcript
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"raw_response": message.content[0].text, "error": str(e)})

@app.route("/process", methods=["POST"])
def process_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    filename = audio_file.filename or "recording.m4a"

    # Save audio to a temp file
    suffix = os.path.splitext(filename)[-1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Step 1: Transcribe with Groq Whisper (free)
        # Always use .m4a extension so Groq recognises the format
        safe_filename = "recording.m4a"
        with open(tmp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(safe_filename, f, "audio/m4a"),
                model="whisper-large-v3",
                language="en",
            )
        transcript = transcription.text

        # Step 2: Generate title and summary with Claude Haiku
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a personal assistant that processes voice notes.

Given this voice note transcript, respond with ONLY valid JSON — no markdown, no code blocks, no explanation.

Transcript:
{transcript}

JSON format:
{{
  "title": "concise title under 60 characters",
  "summary": "2-3 sentences that synthesize and condense the core ideas, decisions, or action items. Do NOT repeat or quote the transcript — interpret and distill it."
}}"""
                }
            ]
        )

        result = clean_and_parse(message.content[0].text)
        result["transcript"] = transcript
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            "title": "Voice Note",
            "summary": transcript[:300],
            "transcript": transcript
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
