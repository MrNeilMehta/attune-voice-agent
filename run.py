"""Run the voice agent from the command line.

Examples
--------
    python run.py                 # text mode (type your replies)
    python run.py --demo          # auto-play a scripted conversation
    python run.py --mode voice    # mic + speech (needs voice extras)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_agent import Appointment, NLUEngine, Patient, VoiceAgent  # noqa: E402
from voice_agent.asr import ASR, TextASR  # noqa: E402
from voice_agent.tts import ConsoleTTS  # noqa: E402


def load_patient(path: Path) -> Patient:
    data = json.loads(path.read_text())
    return Patient(name=data["name"], appointment=Appointment(**data["appointment"]))


class ScriptedASR(ASR):
    """Feeds a fixed list of replies. Used by --demo to show a full call."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def listen(self) -> str:
        line = next(self._lines, "goodbye")
        print(f"You: {line}")
        return line


def build_voice_agent(patient: Patient) -> VoiceAgent:
    from voice_agent.asr import WhisperASR
    from voice_agent.tts import Pyttsx3TTS

    return VoiceAgent(patient, asr=WhisperASR(), tts=Pyttsx3TTS())


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare voice agent demo")
    parser.add_argument(
        "--mode", choices=["text", "voice"], default="text", help="interaction mode"
    )
    parser.add_argument(
        "--patient",
        type=Path,
        default=Path(__file__).parent / "data" / "sample_patient.json",
        help="path to patient JSON",
    )
    parser.add_argument(
        "--demo", action="store_true", help="play a scripted reschedule conversation"
    )
    args = parser.parse_args()

    patient = load_patient(args.patient)

    if args.demo:
        agent = VoiceAgent(
            patient,
            asr=ScriptedASR(
                ["What time is it again?", "Actually I need to reschedule", "Friday morning"]
            ),
            tts=ConsoleTTS(),
            nlu=NLUEngine(),
        )
    elif args.mode == "voice":
        agent = build_voice_agent(patient)
    else:
        agent = VoiceAgent(patient, asr=TextASR(), tts=ConsoleTTS())

    transcript = agent.run()

    if transcript.handoff:
        print(f"\n[call handed off to a human — reason: {transcript.handoff_reason}]")


if __name__ == "__main__":
    main()
