from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

COLORS = {
    "window": "#090c12",
    "rail": "#0d1118",
    "surface": "#121722",
    "surface_raised": "#171d29",
    "input": "#080b10",
    "line": "#252d3b",
    "line_soft": "#1d2430",
    "text": "#edf1f8",
    "muted": "#8c96aa",
    "faint": "#606b7f",
    "blue": "#5b8cff",
    "blue_hover": "#70a0ff",
    "purple": "#a879ff",
    "green": "#2fdaa7",
    "red": "#ff647c",
}


STYLESHEET = f"""
* {{
    font-family: "Inter", "Noto Sans", "DejaVu Sans";
    font-size: 13px;
    color: {COLORS["text"]};
}}
QMainWindow, QWidget#root {{ background: {COLORS["window"]}; }}
QWidget#rail {{
    background: {COLORS["rail"]};
    border-right: 1px solid {COLORS["line_soft"]};
}}
QLabel#appName {{ font-size: 16px; font-weight: 700; }}
QLabel#appTag {{ color: {COLORS["faint"]}; font-size: 9px; letter-spacing: 2px; }}
QLabel#pageTitle {{ font-size: 23px; font-weight: 700; }}
QLabel#pageSubtitle, QLabel[muted="true"] {{ color: {COLORS["muted"]}; }}
QLabel#sectionLabel {{
    color: #aeb7c8;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
}}
QLabel#fieldLabel {{ color: {COLORS["muted"]}; font-size: 12px; }}
QLabel#hint {{ color: {COLORS["faint"]}; font-size: 11px; }}
QLabel#success {{ color: {COLORS["green"]}; }}
QLabel#error {{ color: {COLORS["red"]}; }}
QFrame#card {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["line"]};
    border-radius: 13px;
}}
QFrame#previewFrame {{
    background: #05070b;
    border: 1px solid {COLORS["line"]};
    border-radius: 14px;
}}
QPushButton {{
    min-height: 34px;
    padding: 0 13px;
    background: {COLORS["surface_raised"]};
    border: 1px solid {COLORS["line"]};
    border-radius: 8px;
}}
QPushButton:hover {{ background: #1d2533; border-color: #354055; }}
QPushButton:pressed {{ background: #111620; }}
QPushButton:disabled {{ color: {COLORS["faint"]}; background: #10141c; }}
QPushButton[primary="true"] {{
    color: white;
    font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS["blue"]}, stop:1 #735cff);
    border-color: #7799ff;
}}
QPushButton[primary="true"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS["blue_hover"]}, stop:1 #8b72ff);
}}
QPushButton#navButton {{
    color: {COLORS["muted"]};
    text-align: left;
    border: 0;
    border-radius: 9px;
    background: transparent;
    padding-left: 14px;
}}
QPushButton#navButton:hover {{ color: {COLORS["text"]}; background: #131a26; }}
QPushButton#navButton:checked {{
    color: #dce6ff;
    background: #18243b;
    border-left: 2px solid {COLORS["blue"]};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 34px;
    padding: 0 10px;
    background: {COLORS["input"]};
    border: 1px solid {COLORS["line"]};
    border-radius: 8px;
    selection-background-color: {COLORS["blue"]};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS["blue"]};
}}
QComboBox::drop-down {{ width: 26px; border: 0; }}
QComboBox QAbstractItemView {{
    background: {COLORS["surface_raised"]};
    border: 1px solid {COLORS["line"]};
    selection-background-color: #263758;
    outline: 0;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 1px solid #3a465b;
    border-radius: 5px;
    background: {COLORS["input"]};
}}
QCheckBox::indicator:checked {{
    background: {COLORS["blue"]};
    border-color: #8ab0ff;
    image: none;
}}
QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: #202735; }}
QSlider::sub-page:horizontal {{
    border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS["blue"]}, stop:1 {COLORS["purple"]});
}}
QSlider::handle:horizontal {{
    width: 14px; margin: -6px 0;
    border-radius: 7px; background: #f4f7ff;
}}
QProgressBar {{
    height: 7px;
    border: 0;
    border-radius: 3px;
    background: #1c2330;
    text-align: center;
}}
QProgressBar::chunk {{
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS["blue"]}, stop:1 {COLORS["purple"]});
}}
QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ width: 9px; margin: 2px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #343d4d; min-height: 30px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{
    color: {COLORS["text"]};
    background: {COLORS["surface_raised"]};
    border: 1px solid {COLORS["line"]};
    padding: 6px;
}}
"""


def apply_theme(application: QApplication) -> None:
    application.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["window"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["input"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["surface"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["blue"]))
    application.setPalette(palette)
    application.setFont(QFont("Inter", 10))
    application.setStyleSheet(STYLESHEET)
