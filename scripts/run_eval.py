import argparse
from pathlib import Path

from src.eval_runner import load_eval_case, score_case

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = ROOT/"data"/"eval"/"alice_bob_q3.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run meeting attribution eval case.")
    parser.add_argument(
        "--eval-path",
        type = Path,
        default = DEFAULT_EVAL_PATH,
        help = "Path to eval JSON file.",
    )
    parser.add_argument(
        "--backend",
        choices = ["hf", "local"],
        default = "hf",
        help = "Inference backend: Hugging Face API or local LoRA adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type = Path,
        default = None,
        help = "Path to LoRA adapter folder (local backend only).",
    )
    args = parser.parse_args()
 
    case = load_eval_case(args.eval_path)
    report = score_case(
        case,
        ROOT,
        backend=args.backend,
        adapter_path=args.adapter_path,
    )

    print(f"Backend: { args.backend }")
    print(f"Eval: { report['id'] }")
    print(f"Passed: { report['passed'] }")
    print(f"Line count ok: { report['line_count_ok'] }")
    print(f"Speaker matches: { report['speaker_matches'] }")
    print(f"Expected speakers: { report['expected_speakers'] }")
    print(f"Actual speakers: { report['actual_speakers'] }")
    print(f"Summary bullets: { report['summary_bullets'] }")
    print(f"Action items: { report['action_items'] }")


if __name__ == "__main__":
    main()
