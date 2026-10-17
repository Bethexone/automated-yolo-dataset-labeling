# PySide6 Desktop UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a modern PySide6 desktop application for configuring, testing, and running YOLO dataset labeling.

**Architecture:** Keep the detection and export code UI-independent. Add an explicit export-root workflow API, a configuration-store layer, and a PySide6/qfluentwidgets presentation layer that calls those APIs without blocking the window.

**Tech Stack:** Python 3.12, PySide6, qfluentwidgets, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-pyside6-desktop-ui-design.md`

## Global Constraints

- Use the selected export directory directly as the YOLO dataset root.
- Never overwrite an existing export root; stop on the first error.
- Keep local API keys in ignored `config.local.toml`.
- Preserve CLI default output `<input>/yolo_output`.
- Generate `preview/` only when the preview checkbox is checked.

## Review Focus

- Existing export root: reject before copying any image.
- Empty or duplicate class names: disable run and show validation text.
- Dataset with no supported images: test and batch show an actionable error.
- API failure: restore controls and show the provider error without partial final metadata.
- Test image double-click: open full-size view and close with Escape.

---

### Task 1: Export-root workflow API

**Files:** Modify `src/dataset_label/workflow.py`, `tests/test_core.py`.

**Interfaces:** Produce `label_directory_to(input_dir: Path, output_dir: Path, class_names: Sequence[str], detector: Detector, create_previews: bool) -> dict[str, int]`.

- [ ] Write failing tests for a selected output root, preview creation, and an existing-root error.
- [ ] Run targeted pytest and confirm failure.
- [ ] Implement `label_directory_to` and make existing `label_directory` delegate to it with `input_dir / "yolo_output"`.
- [ ] Run full pytest and confirm pass.
- [ ] Commit `feat: support selected export root`.

### Task 2: Named local API configuration store

**Files:** Modify `src/dataset_label/config.py`, `tests/test_core.py`, `config.example.toml`.

**Interfaces:** Produce `load_provider_configs(path: Path) -> dict[str, ProviderConfig]` and `save_provider_configs(path: Path, configs: Mapping[str, ProviderConfig]) -> None`.

- [ ] Write failing tests for one Qwen profile, profile selection, and TOML round-trip.
- [ ] Run targeted pytest and confirm failure.
- [ ] Implement named profile parsing/writing while preserving one-profile compatibility.
- [ ] Run full pytest and confirm pass.
- [ ] Commit `feat: add named provider configurations`.

### Task 3: GUI model and validation

**Files:** Create `src/dataset_label/gui_model.py`, `tests/test_gui_model.py`.

**Interfaces:** Produce `RunSettings`, `validate_run_settings(settings: RunSettings) -> list[str]`, and `renumber_classes(names: Sequence[str]) -> list[tuple[int, str]]`.

- [ ] Write failing tests for blank/duplicate classes and sequential IDs after removal.
- [ ] Run targeted pytest and confirm failure.
- [ ] Implement settings validation and class ordering.
- [ ] Run full pytest and confirm pass.
- [ ] Commit `feat: add GUI run model`.

### Task 4: Modern PySide6 workspace

**Files:** Create `src/dataset_label/gui.py`, `src/dataset_label/gui_main.py`, `tests/test_gui.py`; modify `pyproject.toml`.

**Interfaces:** Produce `MainWindow(QWidget)` and a `dataset-label-gui` script entry point.

- [ ] Write failing Qt tests for add/remove class renumbering, preview checkbox, and run-button validation.
- [ ] Run targeted pytest and confirm failure.
- [ ] Implement the 200px sidebar, consistent workspace typography, API selector, folder pickers, classes table, and qfluentwidgets controls.
- [ ] Run GUI and full test suites and confirm pass.
- [ ] Commit `feat: add PySide6 labeling workspace`.

### Task 5: Test and batch controller

**Files:** Modify `src/dataset_label/gui.py`, `tests/test_gui.py`.

**Interfaces:** Produce `run_test()` and `run_batch()` window actions using the shared workflow APIs.

- [ ] Write failing Qt tests for first-image test rendering, double-click zoom dialog, and batch preview propagation.
- [ ] Run targeted pytest and confirm failure.
- [ ] Implement background workers, result image display, Escape-close zoom dialog, disabled controls while running, and error reporting.
- [ ] Run full pytest and manual GUI smoke test.
- [ ] Commit `feat: connect desktop labeling workflow`.

### Task 6: Documentation and release verification

**Files:** Modify `README.md`, `config.example.toml`.

- [ ] Document installation, `dataset-label-gui`, profiles, export-root behavior, and preview option.
- [ ] Run `python -m pytest -v` with `C:\software\miniconda3\envs\dataset_label\python.exe`.
- [ ] Verify Git status contains no dataset images, annotations, previews, or `config.local.toml`.
- [ ] Commit `docs: document desktop UI`.
