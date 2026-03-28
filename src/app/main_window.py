import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from src.app.themes.automotive import AUTOMOTIVE_THEME
from src.app.widgets.line_graph_widget import LineGraphWidget
from src.app.widgets.video_widget import VideoWidget
from src.config.constants import SIGNAL_SLOTS_PER_DETECTOR


class TelemetryWorker(QThread):
    """Background thread to poll SharedMemory without blocking Qt UI."""

    frame_ready = Signal(np.ndarray)
    metrics_ready = Signal(dict)

    def __init__(self, running, shm_manager) -> None:
        super().__init__()
        self._running = running
        self._shm = shm_manager

    def run(self) -> None:
        while self._running.value:
            if not self._shm.state.get_event("show_frame").wait(timeout=0.1):
                continue
            self._shm.state.get_event("show_frame").clear()
            try:
                annotated = self._shm.result_reader.read()
                if annotated is not None:
                    self.frame_ready.emit(annotated.copy())
            except Exception as e:
                print("Frame Error:", e)
            try:
                alert_level = self._shm.state.get("alert_level")
                try:
                    distraction_score = self._shm.state.get("distraction_score")
                except Exception:
                    distraction_score = 0.0
                ear_val = float(self._shm._sig_array[0 * SIGNAL_SLOTS_PER_DETECTOR])
                mar_val = float(self._shm._sig_array[1 * SIGNAL_SLOTS_PER_DETECTOR])
                perclos_val = float(self._shm._sig_array[3 * SIGNAL_SLOTS_PER_DETECTOR])
                blink_val = float(self._shm._sig_array[4 * SIGNAL_SLOTS_PER_DETECTOR])
                fps = float(self._shm.state.get("fps"))
                pitch = float(self._shm._sig_array[-9])
                yaw = float(self._shm._sig_array[-8])
                roll = float(self._shm._sig_array[-7])
                state_name = "ALERT"
                if alert_level == 2:
                    state_name = "WARNING"
                elif alert_level >= 3:
                    state_name = "DISTRACTED"
                self.metrics_ready.emit(
                    {
                        "ear": ear_val,
                        "mar": mar_val,
                        "perclos": perclos_val,
                        "state": state_name,
                        "score": distraction_score,
                        "fps": max(0, fps),
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "blink_rate": blink_val,
                    }
                )
            except Exception as e:
                print("Telemetry Error:", e, traceback.format_exc())


# ─── Helper: Create a glass panel frame ───
def _glass_panel() -> QFrame:
    panel = QFrame()
    panel.setObjectName("GlassPanel")
    return panel


# ─── Helper: Create a section title ───
def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionTitle")
    return lbl


# ─── Helper: Create a metric pill ───
def _metric_pill(icon_char: str, title: str, initial_val: str) -> tuple[QFrame, QLabel, QLabel]:
    pill = QFrame()
    pill.setObjectName("MetricPill")
    pill.setFixedHeight(42)
    layout = QHBoxLayout(pill)
    layout.setContentsMargins(14, 0, 14, 0)
    layout.setSpacing(8)

    icon = QLabel(icon_char)
    icon.setStyleSheet("color: #00FFcc; font-size: 9px; background: transparent; border: none;")
    icon.setFixedWidth(12)

    label = QLabel(title)
    label.setStyleSheet(
        "color: #6B7A8D; font-size: 10px; font-weight: 700; letter-spacing: 2px; background: transparent; border: none;"
    )

    val = QLabel(initial_val)
    val.setStyleSheet("color: #E0E6ED; font-size: 13px; font-weight: 800; background: transparent; border: none;")
    val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    layout.addWidget(icon)
    layout.addWidget(label)
    layout.addStretch()
    layout.addWidget(val)

    return pill, label, val


class MainWindow(QMainWindow):
    """Premium Automotive Dashboard — Cyber HUD v2."""

    def __init__(self, running, shm_manager) -> None:
        super().__init__()
        self.setWindowTitle("Driver Awareness System")
        self.setStyleSheet(AUTOMOTIVE_THEME)
        self.setMinimumSize(1280, 760)
        self._running = running
        self._worker = TelemetryWorker(running, shm_manager)
        self._current_state = "ALERT"

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ═══════════════════════════════════════════
        # LEFT PANEL (320px fixed)
        # ═══════════════════════════════════════════
        left_panel = _glass_panel()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)

        # ── State Banner ──
        left_layout.addWidget(_section_title("DRIVER STATUS"))
        self.state_banner = QLabel("● ALERT")
        self.state_banner.setObjectName("Banner_ALERT")
        self.state_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.state_banner.setFixedHeight(48)
        left_layout.addWidget(self.state_banner)

        left_layout.addSpacing(8)

        # ── Awareness Score ──
        left_layout.addWidget(_section_title("AWARENESS SCORE"))

        score_row = QHBoxLayout()
        score_row.setSpacing(2)
        self.score_value = QLabel("100")
        self.score_value.setObjectName("ScoreValue")
        self.score_value.setStyleSheet("color: #00FF88; font-size: 42px; font-weight: 900;")
        self.score_value.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        score_unit = QLabel("%")
        score_unit.setObjectName("ScoreUnit")
        score_unit.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        score_row.addStretch()
        score_row.addWidget(self.score_value)
        score_row.addWidget(score_unit)
        score_row.addStretch()
        left_layout.addLayout(score_row)

        # ── Progress bar ──
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(100)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedHeight(8)
        left_layout.addWidget(self.score_bar)

        left_layout.addSpacing(4)

        # ── Signal Graph ──
        left_layout.addWidget(_section_title("SIGNAL HISTORY"))
        self.signal_graph = LineGraphWidget()
        self.signal_graph.setFixedHeight(70)
        left_layout.addWidget(self.signal_graph)

        left_layout.addSpacing(8)

        # ── Metrics ──
        left_layout.addWidget(_section_title("BIOMETRICS"))

        self.pill_ear, _, self.lbl_ear = _metric_pill("◉", "EYE OPENNESS", "—")
        self.pill_mar, _, self.lbl_mar = _metric_pill("◉", "MOUTH OPN", "—")
        self.pill_blink, _, self.lbl_blink = _metric_pill("◉", "BLINK RATE", "—")
        self.pill_perclos, _, self.lbl_perclos = _metric_pill("◉", "PERCLOS", "—")

        left_layout.addWidget(self.pill_ear)
        left_layout.addWidget(self.pill_mar)
        left_layout.addWidget(self.pill_blink)
        left_layout.addWidget(self.pill_perclos)

        left_layout.addSpacing(4)
        left_layout.addWidget(_section_title("SYSTEM"))

        self.pill_fps, _, self.lbl_fps = _metric_pill("⚡", "FPS", "—")
        self.pill_pose, _, self.lbl_pose = _metric_pill("◎", "HEAD POSE", "—")

        left_layout.addWidget(self.pill_fps)
        left_layout.addWidget(self.pill_pose)

        left_layout.addStretch()
        root.addWidget(left_panel)

        # ═══════════════════════════════════════════
        # RIGHT PANEL (video feed)
        # ═══════════════════════════════════════════
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.video_widget = VideoWidget()
        right_layout.addWidget(self.video_widget, stretch=1)

        root.addWidget(right_panel, stretch=1)

        # ── Connect signals ──
        self._worker.frame_ready.connect(self.video_widget.update_frame)
        self._worker.metrics_ready.connect(self._update_metrics)
        self._worker.start()

    @Slot(dict)
    def _update_metrics(self, data: dict) -> None:
        score = data.get("score", 0.0)
        aware_pct = max(0, min(100, int((1.0 - score) * 100)))
        state = data.get("state", "ALERT")

        # ── Update state banner ──
        if state != self._current_state:
            self._current_state = state
            state_icons = {"ALERT": "●", "WARNING": "▲", "DISTRACTED": "■"}
            icon = state_icons.get(state, "●")
            self.state_banner.setText(f"{icon}  {state}")
            self.state_banner.setObjectName(f"Banner_{state}")
            self.state_banner.style().unpolish(self.state_banner)
            self.state_banner.style().polish(self.state_banner)

        # ── Update score display ──
        state_colors = {
            "ALERT": "#00FF88",
            "WARNING": "#FFAA00",
            "DISTRACTED": "#FF3250",
        }
        color = state_colors.get(state, "#00FF88")
        self.score_value.setText(str(aware_pct))
        self.score_value.setStyleSheet(f"color: {color}; font-size: 42px; font-weight: 900;")

        # ── Update progress bar color ──
        bar_colors = {
            "ALERT": "#00FF88, #00CCAA",
            "WARNING": "#FFAA00, #FF8800",
            "DISTRACTED": "#FF3250, #FF1030",
        }
        bar_gradient = bar_colors.get(state, "#00FF88, #00CCAA")
        self.score_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(18, 24, 38, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                border-radius: 3px;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {bar_gradient.split(",")[0].strip()},
                    stop:1 {bar_gradient.split(",")[1].strip()}
                );
            }}
        """)
        self.score_bar.setValue(aware_pct)

        # ── Signal graph ──
        self.signal_graph.set_value(aware_pct)

        # ── Metric pills ──
        ear = data.get("ear", 0.0)
        mar = data.get("mar", 0.0)
        perclos = data.get("perclos", 0.0)
        blink = data.get("blink_rate", 0.0)

        self._set_metric(self.pill_ear, self.lbl_ear, f"{ear:.2f}", ear < 0.18)
        self._set_metric(self.pill_mar, self.lbl_mar, f"{mar:.2f}", mar >= 1.0)
        self._set_metric(self.pill_perclos, self.lbl_perclos, f"{perclos:.2f}", perclos >= 0.30)
        self._set_metric(self.pill_blink, self.lbl_blink, f"{blink:.0f} ms", blink > 600)

        self.lbl_fps.setText(f"{data.get('fps', 0):.0f} FPS")
        pitch = data.get("pitch", 0)
        yaw = data.get("yaw", 0)
        roll = data.get("roll", 0)
        self.lbl_pose.setText(f"P:{pitch:.0f}  Y:{yaw:.0f}  R:{roll:.0f}")

    def _set_metric(self, pill: QFrame, val_lbl: QLabel, text: str, is_danger: bool) -> None:
        val_lbl.setText(text)
        if is_danger:
            pill.setObjectName("MetricPill_Danger")
            val_lbl.setStyleSheet(
                "color: #FF3250; font-size: 13px; font-weight: 800; background: transparent; border: none;"
            )
        else:
            pill.setObjectName("MetricPill")
            val_lbl.setStyleSheet(
                "color: #E0E6ED; font-size: 13px; font-weight: 800; background: transparent; border: none;"
            )
        pill.style().unpolish(pill)
        pill.style().polish(pill)

    def closeEvent(self, event) -> None:
        self._running.value = 0
        self._worker.quit()
        self._worker.wait()
        import os
        import signal

        os.kill(os.getpid(), signal.SIGINT)
        os._exit(0)
