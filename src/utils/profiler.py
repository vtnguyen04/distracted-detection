from __future__ import annotations

import functools
import json
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
logger = structlog.get_logger()


class FpsTracker:
    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._last_time = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        fps = self._compute_fps()
        self._last_time = now
        return fps

    def _compute_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def current_fps(self) -> float:
        return self._compute_fps()


class InferenceProfiler:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self._frame_count = 0

    def start_timer(self) -> float:
        return time.perf_counter()

    def record(self, component: str, start_time: float) -> float:
        if not self._enabled:
            return 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._timings[component].append(elapsed_ms)
        return elapsed_ms

    def tick_frame(self) -> None:
        self._frame_count += 1

    def get_summary(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for component, times in self._timings.items():
            if not times:
                continue
            sorted_times = sorted(times)
            n = len(sorted_times)
            summary[component] = {
                "avg_ms": sum(times) / n,
                "min_ms": sorted_times[0],
                "max_ms": sorted_times[-1],
                "p95_ms": sorted_times[int(n * 0.95)] if n >= 2 else sorted_times[-1],
                "count": n,
            }
        return summary

    def log_summary(self, interval_frames: int = 100) -> None:
        if not self._enabled:
            return
        if self._frame_count > 0 and self._frame_count % interval_frames == 0:
            summary = self.get_summary()
            for component, stats in summary.items():
                logger.info(
                    "profiler_summary",
                    component=component,
                    avg_ms=round(stats["avg_ms"], 2),
                    p95_ms=round(stats["p95_ms"], 2),
                    max_ms=round(stats["max_ms"], 2),
                    frames=self._frame_count,
                )

    def export_json(self, output_path: str | Path) -> None:
        data = {"total_frames": self._frame_count, "components": self.get_summary()}
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("profiler_exported", path=str(output_path))

    def reset(self) -> None:
        self._timings.clear()
        self._frame_count = 0


def profile_inference(component_name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if args and hasattr(args[0], "_profiler"):
                profiler = args[0]._profiler
                if profiler and isinstance(profiler, InferenceProfiler):
                    profiler._timings[component_name].append(elapsed_ms)
            return result

        return wrapper

    return decorator
