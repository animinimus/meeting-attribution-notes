from src.json_parser import parse_meeting_output

def test_parse_meeting_output_from_json_with_extras():
    raw = """
    Sure! Here is your JSON:
    ```json
    {
        "attributed_lines": [
            {
                "text": "Hey Bob",
                "speaker_name": "Alice",
                "confidence": 0.9,
                "reasoning": "Addresses Bob"
            }
        ],
        "summary_bullets": ["Budget discussed"],
        "action_items": ["Bob sent spreadsheet"]
    }
    ```
    """

    output = parse_meeting_output(raw)
    assert output.attributed_lines[0].speaker_name == "Alice"
    assert output.summary_bullets[0] == "Budget discussed"
