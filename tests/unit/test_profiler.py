import time

from src.utils.profiler import FpsTracker, InferenceProfiler, profile_inference


def test_fps_tracker() -> None:
    """Test FPS calculation."""
    tracker = FpsTracker(window_size=3)
    fps = tracker.tick()
    assert fps == 0.0
    time.sleep(0.01)
    fps = tracker.tick()
    assert fps > 0.0
    assert tracker.current_fps == fps


def test_inference_profiler() -> None:
    """Test profiling latency."""
    profiler = InferenceProfiler(enabled=True)
    t0 = profiler.start_timer()
    time.sleep(0.01)
    elapsed = profiler.record("test_comp", t0)
    assert elapsed >= 10.0
    summary = profiler.get_summary()
    assert "test_comp" in summary
    assert summary["test_comp"]["count"] == 1
    assert summary["test_comp"]["avg_ms"] == elapsed


def test_profiler_disabled() -> None:
    """Test profiler does nothing when disabled."""
    profiler = InferenceProfiler(enabled=False)
    t0 = profiler.start_timer()
    elapsed = profiler.record("test", t0)
    assert elapsed == 0.0
    assert not profiler.get_summary()


def test_profile_inference_decorator() -> None:
    """Test decorator injects timing into profiler."""

    class DummyComponent:
        def __init__(self):
            self._profiler = InferenceProfiler(enabled=True)

        @profile_inference("dummy")
        def run(self):
            time.sleep(0.01)
            return True

    comp = DummyComponent()
    result = comp.run()
    assert result is True
    summary = comp._profiler.get_summary()
    assert "dummy" in summary
    assert summary["dummy"]["count"] == 1
    assert summary["dummy"]["avg_ms"] >= 10.0
