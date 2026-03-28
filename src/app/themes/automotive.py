AUTOMOTIVE_THEME = """
/* ── Global Reset ── */
* {
    margin: 0;
    padding: 0;
}
QWidget {
    background-color: #0A0D12;
    color: #E0E6ED;
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
}
QMainWindow {
    background-color: #0A0D12;
}

/* ── Glass Panels ── */
QFrame#GlassPanel {
    background-color: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(18, 24, 38, 0.95),
        stop:1 rgba(12, 16, 26, 0.90)
    );
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
}

/* ── Section Titles ── */
QLabel#SectionTitle {
    color: #6B7A8D;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding-left: 4px;
}

/* ── State Banners ── */
QLabel#Banner_ALERT {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(0, 255, 136, 0.08),
        stop:1 rgba(0, 200, 100, 0.03)
    );
    color: #00FF88;
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-left: 3px solid #00FF88;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 800;
    padding: 12px 20px;
    letter-spacing: 4px;
}
QLabel#Banner_WARNING {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(255, 170, 0, 0.10),
        stop:1 rgba(255, 140, 0, 0.03)
    );
    color: #FFAA00;
    border: 1px solid rgba(255, 170, 0, 0.3);
    border-left: 3px solid #FFAA00;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 800;
    padding: 12px 20px;
    letter-spacing: 4px;
}
QLabel#Banner_DISTRACTED {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(255, 50, 80, 0.12),
        stop:1 rgba(255, 30, 60, 0.04)
    );
    color: #FF3250;
    border: 1px solid rgba(255, 50, 80, 0.4);
    border-left: 3px solid #FF3250;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 800;
    padding: 12px 20px;
    letter-spacing: 4px;
}

/* ── Metric Pills ── */
QFrame#MetricPill {
    background-color: rgba(18, 24, 38, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 14px;
}
QFrame#MetricPill:hover {
    background-color: rgba(30, 42, 65, 0.7);
    border: 1px solid rgba(0, 255, 204, 0.15);
}
QFrame#MetricPill_Danger {
    background-color: rgba(255, 50, 80, 0.08);
    border: 1px solid rgba(255, 50, 80, 0.2);
    border-radius: 14px;
}

/* ── Score Display ── */
QLabel#ScoreValue {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
}
QLabel#ScoreUnit {
    font-size: 16px;
    font-weight: 600;
    color: #6B7A8D;
}

/* ── Cyber ProgressBar ── */
QProgressBar {
    background-color: rgba(18, 24, 38, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    border-radius: 5px;
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #00FFcc, stop:0.5 #00CCAA, stop:1 #00FF88
    );
}
"""
