"""The orchestrator that runs a full call by wiring the stages together.

    ASR.listen() -> NLUEngine.understand() -> DialogueManager.handle()
                 -> AgentResponse -> TTS.say()

The loop is deliberately thin: every interesting decision lives in the
dialogue manager, and every interesting bit of language lives in NLG. That
separation is what makes the agent testable without audio and swappable
between text and voice with no logic changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .asr import ASR, TextASR
from .dialogue import DialogueManager
from .models import NLUResult, Patient
from .nlu import NLUEngine
from .tts import TTS, ConsoleTTS


@dataclass
class CallTranscript:
    """A record of the call, handy for logging, evaluation, or QA."""

    turns: List[Tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    handoff: bool = False
    handoff_reason: str | None = None

    def add(self, speaker: str, text: str) -> None:
        self.turns.append((speaker, text))


class VoiceAgent:
    def __init__(
        self,
        patient: Patient,
        asr: ASR | None = None,
        tts: TTS | None = None,
        nlu: NLUEngine | None = None,
        max_turns: int = 12,
    ) -> None:
        self.patient = patient
        self.asr = asr or TextASR()
        self.tts = tts or ConsoleTTS()
        self.nlu = nlu or NLUEngine()
        self.dialogue = DialogueManager(patient)
        self.max_turns = max_turns

    def run(self) -> CallTranscript:
        transcript = CallTranscript()

        opening = self.dialogue.start()
        self.tts.say(opening.text)
        transcript.add("agent", opening.text)

        for _ in range(self.max_turns):
            if self.dialogue.finished:
                break

            utterance = self.asr.listen()
            transcript.add("patient", utterance)

            nlu_result: NLUResult = self.nlu.understand(utterance)
            response = self.dialogue.handle(nlu_result)

            self.tts.say(response.text)
            transcript.add("agent", response.text)

            if response.handoff:
                transcript.handoff = True
                transcript.handoff_reason = response.handoff_reason
            if response.end_call:
                break

        return transcript
