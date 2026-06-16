"""Core data structures shared across the voice agent pipeline.

Keeping these as small, typed dataclasses makes the data flowing between the
ASR -> NLU -> dialogue -> NLG -> TTS stages explicit and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Intent(str, Enum):
    """The set of things a patient might mean during a confirmation call."""

    CONFIRM = "confirm"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    ASK_TIME = "ask_time"
    ASK_LOCATION = "ask_location"
    REQUEST_HUMAN = "request_human"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class Appointment:
    """A scheduled visit the agent is calling about."""

    provider: str
    specialty: str
    date: str
    time: str
    location: str

    def pretty(self) -> str:
        return (
            f"{self.specialty} appointment with {self.provider} "
            f"on {self.date} at {self.time}, at {self.location}"
        )


@dataclass
class Patient:
    """The person the agent is speaking with."""

    name: str
    appointment: Appointment

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


@dataclass
class NLUResult:
    """The structured meaning extracted from one patient utterance."""

    intent: Intent
    confidence: float
    matched: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    is_emergency: bool = False


@dataclass
class AgentResponse:
    """What the agent says back, plus control flags for the call loop."""

    text: str
    end_call: bool = False
    handoff: bool = False
    handoff_reason: Optional[str] = None
