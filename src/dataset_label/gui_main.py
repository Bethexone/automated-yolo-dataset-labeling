from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QFrame, QHBoxLayout, QLabel, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (CardWidget, CheckBox, ComboBox, FluentIcon as FIF,
    InfoBar, InfoBarPosition, LineEdit,
    PrimaryPushButton, PushButton, SubtitleLabel, TableWidget, TitleLabel)

from .config import load_provider_configs
from .vision import OpenAICompatibleDetector
from .workflow import _render_preview, discover_images, label_directory_to


class Workspace(QWidget):
    def __init__(self):
        super().__init__(); self.dataset_dir = ""; self.export_dir = ""; self.preview_path = None
        self.profiles = load_provider_configs(Path("config.local.toml"))
        root = QVBoxLayout(self); root.setContentsMargins(36, 30, 36, 30); root.setSpacing(18)
        root.addWidget(TitleLabel("New annotation run")); root.addWidget(SubtitleLabel("Configure your source, classes, and export destination."))
        columns = QHBoxLayout(); columns.setSpacing(18); root.addLayout(columns, 1)
        config = CardWidget(); cl = QVBoxLayout(config); cl.setContentsMargins(22, 22, 22, 22); cl.setSpacing(14); columns.addWidget(config, 3)
        cl.addWidget(SubtitleLabel("Run configuration"))
        self.api = ComboBox(); self.api.addItems(self.profiles.keys()); cl.addWidget(QLabel("Active API configuration")); cl.addWidget(self.api)
        self.dataset = LineEdit(); self.dataset.setReadOnly(True); self.dataset.setPlaceholderText("Select a flat dataset folder")
        self.export = LineEdit(); self.export.setReadOnly(True); self.export.setPlaceholderText("Select final YOLO export folder")
        for label, field, callback in (("Dataset folder", self.dataset, self.pick_dataset), ("Export folder", self.export, self.pick_export)):
            cl.addWidget(QLabel(label)); row=QHBoxLayout(); row.addWidget(field); b=PushButton("Browse"); b.clicked.connect(callback); row.addWidget(b); cl.addLayout(row)
        self.preview = CheckBox("Generate preview images"); self.preview.setChecked(True); cl.addWidget(self.preview)
        cl.addWidget(SubtitleLabel("Classes")); self.classes = TableWidget(self); self.classes.setColumnCount(3); self.classes.setHorizontalHeaderLabels(["ID", "Class name", "Action"]); self.classes.verticalHeader().setVisible(False); self.classes.setColumnWidth(0, 70); self.classes.setColumnWidth(1, 220); self.classes.setColumnWidth(2, 110); self.classes.setMinimumHeight(190); cl.addWidget(self.classes)
        add = PushButton("+ Add class"); add.clicked.connect(lambda: self.add_class()); cl.addWidget(add, alignment=Qt.AlignRight)
        self.add_class("person"); self.add_class("safety helmet")
        test = CardWidget(); tl = QVBoxLayout(test); tl.setContentsMargins(22,22,22,22); tl.setSpacing(14); columns.addWidget(test,2)
        tl.addWidget(SubtitleLabel("Single-image test")); tl.addWidget(QLabel("First image in the dataset. Double-click to enlarge."))
        self.image = QLabel("Test result preview"); self.image.setAlignment(Qt.AlignCenter); self.image.setMinimumSize(360,300); self.image.setStyleSheet("background:#eef2f7;border-radius:8px;"); self.image.mouseDoubleClickEvent=lambda event:self.zoom(); tl.addWidget(self.image,1)
        btest=PushButton("Test first image"); btest.clicked.connect(self.test); tl.addWidget(btest)
        run=PrimaryPushButton("Start labeling", self); run.clicked.connect(self.run); root.addWidget(run, alignment=Qt.AlignRight)

    def add_class(self, name=""):
        row=self.classes.rowCount(); self.classes.insertRow(row); self.classes.setItem(row,0,QTableWidgetItem(str(row))); self.classes.setItem(row,1,QTableWidgetItem(name)); b=PushButton("Remove"); b.clicked.connect(lambda: self.remove_class(b)); self.classes.setCellWidget(row,2,b)
    def remove_class(self, button):
        row = next(
            (index for index in range(self.classes.rowCount()) if self.classes.cellWidget(index, 2) is button),
            -1,
        )
        if row < 0:
            return
        self.classes.removeRow(row)
        for i in range(self.classes.rowCount()): self.classes.item(i,0).setText(str(i))
    def pick_dataset(self):
        self.dataset_dir=QFileDialog.getExistingDirectory(self,"Dataset folder"); self.dataset.setText(self.dataset_dir)
    def pick_export(self):
        self.export_dir=QFileDialog.getExistingDirectory(self,"Export folder"); self.export.setText(self.export_dir)
    def names(self): return [self.classes.item(i,1).text().strip() for i in range(self.classes.rowCount())]
    def fail(self, text): InfoBar.error("Operation failed", text, parent=self, position=InfoBarPosition.TOP, duration=5000)
    def test(self):
        try:
            if not self.dataset_dir: raise ValueError("Select a dataset folder first.")
            image=discover_images(Path(self.dataset_dir))[0]; det=OpenAICompatibleDetector(self.profiles[self.api.currentText()]); found=det.detect(image,self.names()); self.preview_path=Path.cwd()/".test-preview.png"; _render_preview(image,found,self.names(),self.preview_path); self.image.setPixmap(QPixmap(str(self.preview_path)).scaled(self.image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        except Exception as e: self.fail(str(e))
    def zoom(self):
        if not self.preview_path:return
        d=QDialog(self); l=QVBoxLayout(d); x=QLabel(); x.setPixmap(QPixmap(str(self.preview_path))); l.addWidget(x); d.exec()
    def run(self):
        try:
            if not self.dataset_dir or not self.export_dir or not self.names() or any(not n for n in self.names()): raise ValueError("Select folders and enter non-empty classes.")
            det=OpenAICompatibleDetector(self.profiles[self.api.currentText()]); label_directory_to(Path(self.dataset_dir),Path(self.export_dir),self.names(),det,self.preview.isChecked()); InfoBar.success("Completed", "YOLO dataset export completed.", parent=self, position=InfoBarPosition.TOP)
        except Exception as e: self.fail(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Labeler")
        self.resize(1220, 800)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame(self)
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 26, 18, 22)
        sidebar_layout.setSpacing(10)
        brand = TitleLabel("YOLO\nLabeler")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addSpacing(24)

        current = PrimaryPushButton(FIF.HOME, "Annotation")
        current.setFixedHeight(38)
        sidebar_layout.addWidget(current)
        sidebar_layout.addStretch(1)
        sidebar_layout.addWidget(QLabel("VERSION  ·  v0.1.0"))

        self.workspace = Workspace()
        root.addWidget(sidebar)
        root.addWidget(self.workspace, 1)
        self.setObjectName("main-window")
        self.setStyleSheet("""
            #main-window { background: #f7f8fa; color: #202124; }
            #sidebar { background: #ffffff; border-right: 1px solid #e6e8eb; }
            #brand { font-size: 24px; font-weight: 700; color: #1f2937; }
            QLabel { background: transparent; color: #202124; }
            #sidebar QLabel { color: #7b8494; font-size: 12px; letter-spacing: 0.6px; }
        """)

def main() -> int:
    app=QApplication(sys.argv); window=MainWindow(); window.show(); return app.exec()
