import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

TEST_TRANSCRIPT = """
I need to call John tomorrow about the budget, he mentioned we might be over
by 20% and I want to discuss options before the Friday meeting. Also remind me
to send the Q2 report to Sarah by end of day Thursday.
"""

def summarise(transcript):
    message = client.messages.create(
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
  "summary": "2-3 sentences that synthesize and condense the core ideas, decisions, or action items. Do NOT repeat or quote the transcript — interpret and distill it.",
  "labels": ["label1", "label2"]
}}

For labels pick 1-3 that best fit from: work, personal, urgent, idea, task, meeting, follow-up, shopping, health, finance, learning"""
            }
        ]
    )
    return json.loads(message.content[0].text)


if __name__ == "__main__":
    print("Input transcript:")
    print(TEST_TRANSCRIPT.strip())
    print("\nProcessing...\n")

    result = summarise(TEST_TRANSCRIPT)

    print("Title:    ", result["title"])
    print("Summary:  ", result["summary"])
    print("Labels:   ", result["labels"])
