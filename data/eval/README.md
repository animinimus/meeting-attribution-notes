# Eval data

| File | Purpose |
|------|---------|
| `alice_bob_q3.json` | End-to-end regression on `data/sample_transcript.txt` |
| `speaker_attribution_held_out.jsonl` | Three held-out speaker-label cases (gold from `meeting_examples.jsonl` indices 6–8) |

Regenerate the held-out file:

```bash
python3 scripts/compare_base_vs_lora.py --write-held-out
```