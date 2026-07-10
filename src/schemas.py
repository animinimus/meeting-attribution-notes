from pydantic import BaseModel, Field
from typing import List, Optional


class Participant(BaseModel):
    """A class for a participant in a meeting."""
    name: str
    role: Optional[str] = None       # e.g. "head of sales"
    department: Optional[str] = None # e.g. "design"


class TranscriptLine(BaseModel):
    """One line from the raw transcript before we know who said it."""
    raw_label: Optional[str] = None  # e.g. "Speaker 1", or None
    text: str


class AttributedLine(BaseModel):
    """After the model runs: who probably said this line."""
    text: str
    speaker_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class MeetingInput(BaseModel):
    """Everything the user provides."""
    title: str
    participants: List[Participant]
    lines: List[TranscriptLine]


class MeetingOutput(BaseModel):
    """The output of the meeting."""
    attributed_lines: List[AttributedLine]
    summary_bullets: List[str]
    action_items: List[str]