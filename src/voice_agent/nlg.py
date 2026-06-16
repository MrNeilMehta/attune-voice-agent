"""Natural-language generation: turn a decision into warm, human phrasing.

Responses are template-based with light randomization so the agent does not
sound robotic. Each method takes the patient/appointment context it needs and
returns a string. Keeping NLG separate from the dialogue logic means you can
tune tone (or localize) without touching the state machine.
"""
from __future__ import annotations

import random

from .models import Patient


class ResponseGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _pick(self, options: list[str]) -> str:
        return self._rng.choice(options)

    def greeting(self, patient: Patient) -> str:
        appt = patient.appointment
        opener = self._pick(
            [
                f"Hi {patient.first_name}, this is the care team calling with a "
                "quick reminder.",
                f"Hello {patient.first_name}, I'm reaching out on behalf of your "
                "care team.",
            ]
        )
        return (
            f"{opener} You have a {appt.pretty()}. "
            "Can you let me know if that time still works for you?"
        )

    def confirmed(self, patient: Patient) -> str:
        return self._pick(
            [
                f"Wonderful, thank you {patient.first_name}. You're all set, and "
                "we look forward to seeing you.",
                f"Great, I've marked you as confirmed. Take care, "
                f"{patient.first_name}!",
            ]
        )

    def ask_reschedule(self) -> str:
        return self._pick(
            [
                "No problem at all. What day or time would work better for you?",
                "That's completely fine. When would be a better time to come in?",
            ]
        )

    def reschedule_captured(self, preference: str) -> str:
        return (
            f"Thank you. I've noted that you'd prefer {preference}. A scheduler "
            "will follow up shortly to lock in a new time that works."
        )

    def cancelled(self) -> str:
        return self._pick(
            [
                "Okay, I've noted that you won't be able to make it. Someone from "
                "the office will reach out to help you find another time.",
                "Understood, I'll cancel this one for now. The team will follow up "
                "so you don't miss the care you need.",
            ]
        )

    def answer_time(self, patient: Patient) -> str:
        appt = patient.appointment
        return (
            f"Of course. It's on {appt.date} at {appt.time}. "
            "Does that time work for you?"
        )

    def answer_location(self, patient: Patient) -> str:
        appt = patient.appointment
        return f"It's at {appt.location}. Will you be able to make it?"

    def handoff(self, reason: str) -> str:
        if reason == "emergency":
            return (
                "It sounds like this may be urgent. If this is a medical "
                "emergency, please hang up and call 911 right now. Otherwise, "
                "I'm connecting you with a member of our clinical team."
            )
        return self._pick(
            [
                "Of course, let me connect you with someone from the team who can "
                "help.",
                "No problem, I'll hand you over to a team member right now.",
            ]
        )

    def reprompt(self) -> str:
        return self._pick(
            [
                "Sorry, I didn't quite catch that. Could you let me know if the "
                "appointment time works, or if you'd like to change it?",
                "I want to make sure I get this right. Are you able to keep the "
                "appointment, or would you prefer to reschedule?",
            ]
        )

    def goodbye(self, patient: Patient) -> str:
        return f"Thanks for your time, {patient.first_name}. Take care!"
