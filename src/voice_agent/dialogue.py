"""Dialogue management: the state machine that decides what to do next.

This is the heart of the agent and the part most worth understanding deeply.
It consumes :class:`NLUResult` objects and emits :class:`AgentResponse` objects,
tracking just enough state to run a coherent confirmation call:

    START -> CONFIRMING -> (RESCHEDULING) -> ENDED / HANDOFF

Design notes worth mentioning in an interview:
* The manager is fully decoupled from audio. It only deals in text + intents,
  which is why the same logic is exercised by the unit tests and the live
  voice loop.
* It fails safe: emergencies, explicit human requests, and repeated confusion
  all route to a human rather than guessing.
"""
from __future__ import annotations

from enum import Enum, auto

from .models import AgentResponse, Intent, NLUResult, Patient
from .nlg import ResponseGenerator

MAX_UNKNOWN_BEFORE_HANDOFF = 2


class State(Enum):
    START = auto()
    CONFIRMING = auto()
    RESCHEDULING = auto()
    ENDED = auto()
    HANDOFF = auto()


class DialogueManager:
    def __init__(self, patient: Patient, nlg: ResponseGenerator | None = None) -> None:
        self.patient = patient
        self.nlg = nlg or ResponseGenerator()
        self.state = State.START
        self._unknown_count = 0

    @property
    def finished(self) -> bool:
        return self.state in (State.ENDED, State.HANDOFF)

    def start(self) -> AgentResponse:
        """Produce the opening line and move into the confirming state."""
        self.state = State.CONFIRMING
        return AgentResponse(self.nlg.greeting(self.patient))

    def handle(self, nlu: NLUResult) -> AgentResponse:
        """Advance the conversation given the latest understood utterance."""
        if nlu.is_emergency:
            return self._to_handoff("emergency")

        if nlu.intent == Intent.REQUEST_HUMAN:
            return self._to_handoff("requested")

        if self.state == State.RESCHEDULING:
            return self._handle_rescheduling(nlu)

        # Default: we're in CONFIRMING.
        return self._handle_confirming(nlu)

    # -- state handlers -------------------------------------------------

    def _handle_confirming(self, nlu: NLUResult) -> AgentResponse:
        if nlu.intent == Intent.CONFIRM:
            self.state = State.ENDED
            return AgentResponse(self.nlg.confirmed(self.patient), end_call=True)

        if nlu.intent == Intent.CANCEL:
            self.state = State.ENDED
            return AgentResponse(self.nlg.cancelled(), end_call=True)

        if nlu.intent == Intent.RESCHEDULE:
            self.state = State.RESCHEDULING
            return AgentResponse(self.nlg.ask_reschedule())

        if nlu.intent == Intent.ASK_TIME:
            return AgentResponse(self.nlg.answer_time(self.patient))

        if nlu.intent == Intent.ASK_LOCATION:
            return AgentResponse(self.nlg.answer_location(self.patient))

        if nlu.intent == Intent.GOODBYE:
            self.state = State.ENDED
            return AgentResponse(self.nlg.goodbye(self.patient), end_call=True)

        return self._handle_unknown()

    def _handle_rescheduling(self, nlu: NLUResult) -> AgentResponse:
        # In this state we treat the raw utterance as the patient's preferred
        # time. A production system would run a date/time parser here; for the
        # demo we capture the phrase and hand off to a human scheduler.
        preference = nlu.entities.get("raw_text", "the time you mentioned")
        self.state = State.ENDED
        return AgentResponse(self.nlg.reschedule_captured(preference), end_call=True)

    def _handle_unknown(self) -> AgentResponse:
        self._unknown_count += 1
        if self._unknown_count >= MAX_UNKNOWN_BEFORE_HANDOFF:
            return self._to_handoff("confused")
        return AgentResponse(self.nlg.reprompt())

    def _to_handoff(self, reason: str) -> AgentResponse:
        self.state = State.HANDOFF
        return AgentResponse(
            self.nlg.handoff(reason), end_call=True, handoff=True, handoff_reason=reason
        )
