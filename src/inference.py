import os 

from dotenv import load_dotenv 
from huggingface_hub import InferenceClient

from src.json_parser import parse_meeting_output
from src.prompt import build_messages
from src.schemas import MeetingInput, MeetingOutput

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"

def get_client() -> InferenceClient:
    """
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
    return InferenceClient(token=token)

def run_meeting_analysis(
    meeting: MeetingInput,
    model: str = DEFAULT_MODEL,
) -> MeetingOutput:
    """
    Retrieve the model's output based on the meeting input and prompt.
    """
    client = get_client()
    messages = build_messages(meeting)

    response = client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=1200,
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content


    # print(f"Raw text: {raw_text}")
    return parse_meeting_output(raw_text)