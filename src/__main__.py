from __future__ import annotations

import argparse

import structlog

from src.config.logging import setup_logging
from src.config.settings import load_settings

logger = structlog.get_logger()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distracted Detection — Real-time distraction detection system")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file (default: uses built-in defaults + env vars)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings(config_path=args.config)
    setup_logging(settings.pipeline.log_level)
    logger.info(
        "app_starting",
        backend=settings.pipeline.model_backend,
        detectors=settings.pipeline.enabled_detectors,
        camera_source=settings.camera.source or settings.camera.index,
    )
    import sys

    from src.engine.process_manager import ProcessManager

    manager = ProcessManager(settings)
    manager.start()
    try:
        if not settings.pipeline.show_ui:
            import time

            logger.info("headless_mode", info="Running without UI. Press Ctrl+C to stop.")
            while manager._running.value:
                time.sleep(0.5)
            return

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication

        from src.app.main_window import MainWindow

        app = QApplication(sys.argv)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 17, 21))
        palette.setColor(QPalette.WindowText, Qt.white)
        app.setPalette(palette)
        window = MainWindow(running=manager._running, shm_manager=manager._shm)
        window.show()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("app_interrupted")
    finally:
        logger.info("app_shutdown")
        manager.stop()


if __name__ == "__main__":
    main()
