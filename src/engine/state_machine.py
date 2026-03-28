from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from src.domain.enums import AlertLevel, DriverState

if TYPE_CHECKING:
    from src.config.settings import StateMachineSettings
logger = structlog.get_logger()


class BaseState(ABC):
    """Abstract base class for driver states."""

    @property
    @abstractmethod
    def name(self) -> DriverState:
        """Returns the enum representing this state."""

    @property
    @abstractmethod
    def alert_level(self) -> AlertLevel:
        """Returns the alert level for this state."""

    @abstractmethod
    def handle(self, context: DriverStateMachine, score: float) -> None:
        """Evaluate the score and request transitions if necessary."""


class AlertState(BaseState):
    """State when the driver is attentive and safe."""

    @property
    def name(self) -> DriverState:
        return DriverState.ALERT

    @property
    def alert_level(self) -> AlertLevel:
        return AlertLevel.NONE

    def handle(self, context: DriverStateMachine, score: float) -> None:
        if score >= context.settings.distracted_threshold:
            context.request_transition(DistractedState(), context.settings.distracted_sustain_sec)
        elif score >= context.settings.warning_threshold:
            context.request_transition(WarningState(), context.settings.warning_sustain_sec)
        else:
            context._cancel_transition()


class WarningState(BaseState):
    """State when the driver exhibits mild distraction signs."""

    @property
    def name(self) -> DriverState:
        return DriverState.WARNING

    @property
    def alert_level(self) -> AlertLevel:
        return AlertLevel.MEDIUM

    def handle(self, context: DriverStateMachine, score: float) -> None:
        if score >= context.settings.distracted_threshold:
            context.request_transition(DistractedState(), context.settings.distracted_sustain_sec)
        elif score < context.settings.safe_threshold:
            context.request_transition(AlertState(), context.settings.recovery_sustain_sec)
        else:
            context._cancel_transition()


class DistractedState(BaseState):
    """State when the driver is critically distracted or distracted."""

    @property
    def name(self) -> DriverState:
        return DriverState.DISTRACTED

    @property
    def alert_level(self) -> AlertLevel:
        return AlertLevel.HIGH

    def handle(self, context: DriverStateMachine, score: float) -> None:
        if score < context.settings.safe_threshold:
            context.request_transition(AlertState(), context.settings.recovery_sustain_sec)
        else:
            context._cancel_transition()


class DriverStateMachine:
    """Context class for the State Pattern managing driver alerts."""

    def __init__(self, settings: StateMachineSettings) -> None:
        self.settings = settings
        self._current_state: BaseState = AlertState()
        self._pending_state: BaseState | None = None
        self._transition_start: float | None = None
        self._sustain_required: float = 0.0

    @property
    def state(self) -> DriverState:
        """Get the current state enum representation."""
        return self._current_state.name

    @property
    def alert_level(self) -> AlertLevel:
        """Get the required alert level for the current state."""
        return self._current_state.alert_level

    def update(self, distraction_score: float) -> DriverState:
        """Evaluate the new score against the current state logic."""
        self._current_state.handle(self, distraction_score)
        self._process_transition()
        return self.state

    def request_transition(self, target_state: BaseState, sustain_sec: float) -> None:
        """Register a requested state change with its required sustain duration."""
        if target_state.name == self._current_state.name:
            self._cancel_transition()
            return
        if self._pending_state is None or target_state.name != self._pending_state.name:
            self._pending_state = target_state
            self._transition_start = time.monotonic()
            self._sustain_required = sustain_sec

    def _process_transition(self) -> None:
        """Execute a state change if the temporal hysteresis conditions are met."""
        if self._pending_state is None or self._transition_start is None:
            return
        elapsed = time.monotonic() - self._transition_start
        if elapsed >= self._sustain_required:
            old_name = self._current_state.name.name
            self._current_state = self._pending_state
            logger.info(
                "state_transition",
                old_state=old_name,
                new_state=self._current_state.name.name,
                sustain_sec=round(elapsed, 1),
            )
            self._cancel_transition()

    def _cancel_transition(self) -> None:
        """Clear any pending transitions."""
        self._pending_state = None
        self._transition_start = None
        self._sustain_required = 0.0

    def reset(self) -> None:
        """Forcibly reset the state machine to Alert safely."""
        self._current_state = AlertState()
        self._cancel_transition()
