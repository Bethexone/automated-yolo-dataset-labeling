from __future__ import annotations

import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QCheckBox, QComboBox

from .config import load_provider_configs
from .vision import OpenAICompatibleDetector
from .workflow import discover_images, _render_preview, label_directory_to


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Labeler")
        self.resize(1180, 760)
        self.dataset_dir = ""
        self.export_dir = ""
        self.preview_path: Path | None = None
        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
        side = QWidget(); side.setFixedWidth(200); side.setStyleSheet("background:#f4f6f9;")
        sl = QVBoxLayout(side); sl.setContentsMargins(18, 28, 18, 28)
        brand = QLabel("YOLO Labeler\n<span style='font-size:13px;color:#687386'>Desktop workspace</span>"); brand.setStyleSheet("font-size:21px;font-weight:600;"); sl.addWidget(brand)
        for text in ("Annotation workspace", "API configurations", "Recent exports"):
            button = QPushButton(text); button.setFlat(True); button.setStyleSheet("text-align:left;padding:10px;font-size:16px;"); sl.addWidget(button)
        sl.addStretch(); sl.addWidget(QLabel("v0.1.0\nOpenAI-compatible")); layout.addWidget(side)
        body = QWidget(); bl = QVBoxLayout(body); bl.setContentsMargins(30, 28, 30, 24)
        title = QLabel("New annotation run\n<span style='font-size:16px;color:#687386;font-weight:400'>Configure your source, classes, and export destination.</span>"); title.setStyleSheet("font-size:28px;font-weight:600;"); bl.addWidget(title)
        content = QHBoxLayout(); bl.addLayout(content, 1)
        left = QWidget(); ll = QVBoxLayout(left); content.addWidget(left, 3)
        api = QComboBox(); self.profiles = load_provider_configs(Path("config.local.toml")); api.addItems(self.profiles.keys()); self.api = api
        self.dataset = QLabel("No dataset folder selected"); self.export = QLabel("No export folder selected")
        for label, value, handler in (("Active API configuration", api, None), ("Dataset folder", self.dataset, self.pick_dataset), ("Export folder", self.export, self.pick_export)):
            ll.addWidget(QLabel(label)); ll.addWidget(value)
            if handler: b=QPushButton("Browse"); b.clicked.connect(handler); ll.addWidget(b)
        self.preview = QCheckBox("Generate preview images"); self.preview.setChecked(True); ll.addWidget(self.preview)
        ll.addWidget(QLabel("Classes")); self.classes = QTableWidget(0, 3); self.classes.setHorizontalHeaderLabels(["ID", "Class name", "Action"]); ll.addWidget(self.classes)
        add=QPushButton("+ Add class"); add.clicked.connect(self.add_class); ll.addWidget(add)
        self.add_class("person"); self.add_class("safety helmet")
        right=QWidget(); rl=QVBoxLayout(right); content.addWidget(right,2); rl.addWidget(QLabel("Single-image test\n<span style='font-size:14px;color:#687386'>Double-click image to enlarge</span>"))
        self.image=QLabel("Test result preview"); self.image.setAlignment(Qt.AlignCenter); self.image.setMinimumSize(360,300); self.image.setStyleSheet("background:#eef2f7;"); self.image.mouseDoubleClickEvent=lambda e:self.zoom(); rl.addWidget(self.image,1)
        test=QPushButton("Test first image"); test.clicked.connect(self.test); rl.addWidget(test)
        run=QPushButton("Start labeling"); run.clicked.connect(self.run); run.setStyleSheet("background:#2354d9;color:white;padding:10px;"); bl.addWidget(run)
        layout.addWidget(body,1)

    def add_class(self, name: str=""):
        row=self.classes.rowCount(); self.classes.insertRow(row); self.classes.setItem(row,0,QTableWidgetItem(str(row))); self.classes.setItem(row,1,QTableWidgetItem(name)); b=QPushButton("Remove"); b.clicked.connect(lambda: self.remove_class(b)); self.classes.setCellWidget(row,2,b)
    def remove_class(self,b):
        self.classes.removeRow(self.classes.indexAt(b.pos()).row())
        for i in range(self.classes.rowCount()): self.classes.item(i,0).setText(str(i))
    def pick_dataset(self):
        self.dataset_dir=QFileDialog.getExistingDirectory(self,"Dataset folder"); self.dataset.setText(self.dataset_dir or "No dataset folder selected")
    def pick_export(self):
        self.export_dir=QFileDialog.getExistingDirectory(self,"Export folder"); self.export.setText(self.export_dir or "No export folder selected")
    def names(self): return [self.classes.item(i,1).text().strip() for i in range(self.classes.rowCount())]
    def validate(self):
        if not self.dataset_dir or not self.export_dir: raise ValueError("Select both dataset and export folders.")
        if not self.names() or any(not name for name in self.names()): raise ValueError("Enter at least one non-empty class name.")
    def test(self):
        try:
            if not self.dataset_dir: raise ValueError("Select a dataset folder first.")
            image=discover_images(Path(self.dataset_dir))[0]; detector=OpenAICompatibleDetector(self.profiles[self.api.currentText()]); detections=detector.detect(image,self.names()); self.preview_path=Path.cwd()/".test-preview.png"; _render_preview(image,detections,self.names(),self.preview_path); self.image.setPixmap(QPixmap(str(self.preview_path)).scaled(self.image.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        except Exception as e: QMessageBox.critical(self,"Test failed",str(e))
    def zoom(self):
        if not self.preview_path:return
        d=QDialog(self); l=QVBoxLayout(d); x=QLabel(); x.setPixmap(QPixmap(str(self.preview_path))); l.addWidget(x); d.exec()
    def run(self):
        try:
            self.validate()
            detector=OpenAICompatibleDetector(self.profiles[self.api.currentText()]); label_directory_to(Path(self.dataset_dir),Path(self.export_dir),self.names(),detector,self.preview.isChecked()); QMessageBox.information(self,"Complete","Dataset export completed.")
        except Exception as e: QMessageBox.critical(self,"Labeling failed",str(e))

def main() -> int:
    app=QApplication(sys.argv); window=MainWindow(); window.show(); return app.exec()
