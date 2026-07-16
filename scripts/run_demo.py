from pathlib import Path

from src.inference import run_meeting_analysis
from src.parser import parse_meeting_text
from src.schemas import Participant

SAMPLE_PATH = Path(__file__).resolve().parents[1]/"data"/"sample_transcript.txt"

raw = SAMPLE_PATH.read_text(encoding="utf-8")

meeting = parse_meeting_text(
    raw,
    participants=[
        Participant(
            name="Alice McAliceFace", 
            role="design",
            department="product"
        ),
        Participant(
            name="Bob McBobFace",
            role="finance",
            department="finance"
        ),
    ]
)

print(f"Title: {meeting.title}")
print(f"Lines parsed: {len(meeting.lines)}")

print("Calling HuggingFace model...")
result = run_meeting_analysis(meeting)
print("Analysis complete!")

print("=== Attributed Lines ===")
for line in result.attributed_lines:
    print(f"[{line.confidence:.2f}] { line.speaker_name }: { line.text }")
    if line.reasoning:
      print(f"Why: { line.reasoning }")

print("=== Summary Bullets ===")
for bullet in result.summary_bullets:
    print(f"- { bullet }")

print("=== Action Items ===")
for item in result.action_items:
    print(f"- { item }")
