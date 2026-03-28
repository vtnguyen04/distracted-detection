from __future__ import annotations

import time

from src.config.settings import StateMachineSettings
from src.domain.enums import AlertLevel, DriverState
from src.engine.state_machine import DriverStateMachine


class TestDriverStateMachine:
    """Tests for state transitions with hysteresis."""

    def test_initial_state_is_alert(
        self,
        state_machine_settings: StateMachineSettings,
    ) -> None:
        """State machine should start in ALERT state."""
        sm = DriverStateMachine(state_machine_settings)
        assert sm.state == DriverState.ALERT
        assert sm.alert_level == AlertLevel.NONE

    def test_stays_alert_below_warning_threshold(
        self,
        state_machine_settings: StateMachineSettings,
    ) -> None:
        """Low scores should keep state at ALERT."""
        sm = DriverStateMachine(state_machine_settings)
        for _ in range(10):
            state = sm.update(0.2)
        assert state == DriverState.ALERT

    def test_transitions_to_warning_after_sustain(
        self,
    ) -> None:
        """Score above warning threshold sustained for required duration = WARNING."""
        settings = StateMachineSettings(
            warning_threshold=0.5,
            warning_sustain_sec=0.01,
        )
        sm = DriverStateMachine(settings)
        sm.update(0.6)
        time.sleep(0.02)
        state = sm.update(0.6)
        assert state == DriverState.WARNING

    def test_transitions_to_distracted_after_sustain(
        self,
    ) -> None:
        """Score above distracted threshold sustained = DISTRACTED."""
        settings = StateMachineSettings(
            warning_threshold=0.5,
            distracted_threshold=0.7,
            warning_sustain_sec=0.01,
            distracted_sustain_sec=0.01,
        )
        sm = DriverStateMachine(settings)
        sm.update(0.6)
        time.sleep(0.02)
        sm.update(0.6)
        sm.update(0.8)
        time.sleep(0.02)
        state = sm.update(0.8)
        assert state == DriverState.DISTRACTED

    def test_recovery_to_alert(
        self,
    ) -> None:
        """Low score sustained after DISTRACTED = back to ALERT."""
        settings = StateMachineSettings(
            warning_threshold=0.5,
            distracted_threshold=0.7,
            safe_threshold=0.3,
            warning_sustain_sec=0.01,
            distracted_sustain_sec=0.01,
            recovery_sustain_sec=0.01,
        )
        sm = DriverStateMachine(settings)
        sm.update(0.8)
        time.sleep(0.02)
        sm.update(0.8)
        sm.update(0.8)
        time.sleep(0.02)
        sm.update(0.8)
        sm.update(0.1)
        time.sleep(0.02)
        state = sm.update(0.1)
        assert state == DriverState.ALERT

    def test_no_false_transition_without_sustain(
        self,
    ) -> None:
        """Momentary high score should NOT trigger transition."""
        settings = StateMachineSettings(
            warning_threshold=0.5,
            warning_sustain_sec=10.0,
        )
        sm = DriverStateMachine(settings)
        sm.update(0.9)
        state = sm.update(0.9)
        assert state == DriverState.ALERT

    def test_reset(self) -> None:
        """Reset should return to ALERT state."""
        settings = StateMachineSettings(
            warning_sustain_sec=0.01,
        )
        sm = DriverStateMachine(settings)
        sm.update(0.6)
        time.sleep(0.02)
        sm.update(0.6)
        sm.reset()
        assert sm.state == DriverState.ALERT
