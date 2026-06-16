"""Text-to-speech (TTS) backends behind one small interface.

`ConsoleTTS` just prints what the agent would say, so the whole pipeline runs
anywhere with no audio hardware. `Pyttsx3TTS` speaks aloud offline. As with
ASR, the heavy/optional import is lazy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TTS(ABC):
    @abstractmethod
    def say(self, text: str) -> None:
        """Render the agent's reply to the patient."""


class ConsoleTTS(TTS):
    def __init__(self, label: str = "Agent: ") -> None:
        self.label = label

    def say(self, text: str) -> None:
        print(f"{self.label}{text}")


class Pyttsx3TTS(TTS):
    """Speaks aloud using the offline pyttsx3 engine.

    Requires: pip install pyttsx3
    """

    def __init__(self, rate: int = 175) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Spoken output needs pyttsx3. Install it with:\n"
                "    pip install pyttsx3"
            ) from exc
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def say(self, text: str) -> None:  # pragma: no cover - requires audio out
        print(f"Agent: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
