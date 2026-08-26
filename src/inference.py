import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.json_parser import parse_meeting_output
from src.prompt import build_messages
from src.schemas import MeetingInput, MeetingOutput

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "models" / "meeting-attribution-lora"

Backend = Literal["hf", "local"]

_local_model = None
_local_tokenizer = None
_local_adapter_path: Optional[Path] = None


def get_client() -> InferenceClient:
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
    return InferenceClient(token=token)


def _resolve_adapter_path(adapter_path: Optional[str | Path]) -> Path:
    if adapter_path is not None:
        path = Path(adapter_path)
    else:
        env_path = os.getenv("LORA_ADAPTER_PATH")
        path = Path(env_path) if env_path else DEFAULT_ADAPTER_DIR

    if not path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {path}. "
            "Unzip meeting-attribution-lora.zip into models/meeting-attribution-lora/ "
            "or set LORA_ADAPTER_PATH."
        )
    return path


def _get_local_model_and_tokenizer(adapter_path: Path):
    global _local_model, _local_tokenizer, _local_adapter_path

    if _local_model is not None and _local_tokenizer is not None:
        if _local_adapter_path == adapter_path:
            return _local_model, _local_tokenizer

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    _local_model = model
    _local_tokenizer = tokenizer
    _local_adapter_path = adapter_path
    return model, tokenizer


def _run_local_inference(
    messages: list[dict],
    adapter_path: Path,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    model, tokenizer = _get_local_model_and_tokenizer(adapter_path)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs.update(do_sample=True, temperature=temperature)
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def run_meeting_analysis(
    meeting: MeetingInput,
    model: str = DEFAULT_MODEL,
    backend: Backend = "hf",
    adapter_path: Optional[str | Path] = None,
    max_new_tokens: int = 1200,
    temperature: float = 0.2,
) -> MeetingOutput:
    """
    Run meeting analysis via Hugging Face Inference API (default) or a local LoRA adapter.
    """
    messages = build_messages(meeting)

    if backend == "hf":
        client = get_client()
        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        raw_text = response.choices[0].message.content
    elif backend == "local":
        path = _resolve_adapter_path(adapter_path)
        raw_text = _run_local_inference(
            messages,
            adapter_path=path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'hf' or 'local'.")

    return parse_meeting_output(raw_text)
