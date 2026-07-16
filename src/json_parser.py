import json
import re

from src.schemas import MeetingOutput

def extract_json_object(raw_text: str) -> dict:
    """
    Extract a JSON object from the raw text, ignoring any markdown or other formatting.

    e.g.,

    ```json

    { content }

    ```
    """
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text) # remove opening ``` or ```json
        text = re.sub(r"\s*```$", "", text) # remove closing ```

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback to substring the { and } for cases with meta text before the response, e.g., "Here's your JSON..."
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 and end == -1 or end <= start:
        raise ValueError("No valid JSON object found in the text.")

    return json.loads(text[start : end + 1])

def parse_meeting_output(raw_text: str) -> MeetingOutput:
    """
    Parse model text response into MeetingOutput schema.
    """
    data = extract_json_object(raw_text)
    return MeetingOutput.model_validate(data)
