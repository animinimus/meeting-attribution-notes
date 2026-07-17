from pathlib import Path
from src.eval_runner import load_eval_case, score_case

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT/"data"/"eval"/"alice_bob_q3.json"

case = load_eval_case(EVAL_PATH)
report = score_case(case, ROOT)

print(f"Eval: {report['id']}")
print(f"Passed: {report['passed']}")
print(f"Line count ok: {report['line_count_ok']}")
print(f"Speaker matches: {report['speaker_matches']}")
print(f"Expected speakers: {report['expected_speakers']}")
print(f"Actual speakers: {report['actual_speakers']}")
print(f"Summary bullets: {report['summary_bullets']}")
print(f"Action items: {report['action_items']}")