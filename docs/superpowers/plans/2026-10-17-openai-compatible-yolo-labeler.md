# OpenAI-Compatible YOLO Auto Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python 3.12 command-line tool that labels a flat image directory through an OpenAI-compatible vision API and emits an unsplit YOLO bounding-box dataset.

**Architecture:** A small `argparse` CLI orchestrates configuration loading, deterministic input discovery, a generic OpenAI-compatible client, strict JSON/coordinate validation, and YOLO writing. The provider transport is isolated behind a callable protocol so tests can mock it without network access.

**Tech Stack:** Python 3.12 at `C:\software\miniconda3\envs\dataset_label\python.exe`, `openai`, standard-library `argparse` and `tomllib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-17-openai-compatible-yolo-labeler-design.md`

## Global Constraints

- Create and use only the `dataset_label` Conda environment with Python 3.12 at `C:\software\miniconda3\envs\dataset_label\python.exe`.
- Support only flat input folders and JPG, JPEG, PNG, and WebP images.
- Accept only OpenAI-compatible Chat Completions vision providers configured by `base_url`, `api_key`, and `model` in `config.local.toml`.
- Never track `config.local.toml` or a real API key.
- Create `<input-dir>/yolo_output` with no train/validation split; refuse to overwrite it.
- Process files serially in deterministic filename order and fail immediately on any error.
- Convert Qwen3-VL `[0,999]` relative boxes directly to YOLO `[0,1]` values.
- First invocation uses classes `人员` then `安全帽`, yielding IDs 0 and 1.

## Review Focus

- An empty but valid detection array must still produce an empty matching label file; Task 4 tests this.
- A JSON object with an unrequested category must abort rather than silently omit it; Task 3 tests this.
- A box on an exact coordinate boundary of 0 or 999 must be accepted; Task 3 tests this.
- A pre-existing `yolo_output` directory must abort before any API client is created or file is copied; Task 4 tests this.
- Mixed-case supported extensions must be discovered while unsupported files and subdirectories are ignored; Task 2 tests this.

---

## File Structure

- `pyproject.toml` — project metadata, runtime dependency, test dependency, and `dataset-label` console script.
- `.gitignore` — exclude local configuration, Conda files, bytecode, coverage, and test caches.
- `config.example.toml` — tracked provider configuration shape without a credential.
- `src/dataset_label/config.py` — provider configuration model and strict TOML loading.
- `src/dataset_label/models.py` — typed detection and normalized YOLO row value objects.
- `src/dataset_label/vision.py` — OpenAI-compatible client wrapper plus strict API JSON parsing.
- `src/dataset_label/yolo.py` — box conversion and YOLO/data.yaml serialization.
- `src/dataset_label/workflow.py` — input discovery, output creation, serial orchestration, and reporting.
- `src/dataset_label/cli.py` — command-line parsing and exit-code handling.
- `src/dataset_label/__init__.py` — package marker.
- `tests/conftest.py` — temporary image and mocked detector fixtures.
- `tests/test_config.py` — configuration tests.
- `tests/test_vision.py` — API-response parsing and validation tests.
- `tests/test_yolo.py` — coordinate conversion and serialization tests.
- `tests/test_workflow.py` — end-to-end filesystem workflow tests with a mocked detector.
- `README.md` — environment creation, configuration, invocation, output, and failure behavior.

### Task 1: Create the Python project and local-secret boundary

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.example.toml`
- Create: `src/dataset_label/__init__.py`
- Create: `src/dataset_label/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`

**Interfaces:**
- Produces: a runnable `dataset-label` console command dispatched to `dataset_label.cli:main`.
- Produces: local `config.local.toml` exclusion used by all later tasks.

- [ ] **Step 1: Create the specified Conda environment and install project dependencies**

Run with the supplied Conda installation so its interpreter is exactly `C:\software\miniconda3\envs\dataset_label\python.exe`; install `openai` as runtime dependency and `pytest` as test dependency.

- [ ] **Step 2: Write the failing CLI help test**

```python
def test_main_prints_help_and_returns_zero(capsys):
    assert main(["--help"]) == 0
    assert "dataset-label" in capsys.readouterr().out
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_cli.py::test_main_prints_help_and_returns_zero -v`

Expected: FAIL because the package and `main` do not exist.

- [ ] **Step 4: Implement `main(argv: Sequence[str] | None = None) -> int` in `src/dataset_label/cli.py`**

Use `argparse`; define positional `input_dir` and required `--classes` with one-or-more names. Keep `--help` as a success path when `main` is called directly.

- [ ] **Step 5: Add packaging and safe configuration artifacts**

Define the `dataset-label` console script in `pyproject.toml`, add the `openai` dependency, exclude `config.local.toml` in `.gitignore`, and create `config.example.toml` using the exact `[provider]` fields from the spec. Document environment creation and copying the example to a local configuration file without inserting a real key.

- [ ] **Step 6: Run the CLI test to verify it passes**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_cli.py -v`

Expected: PASS.

### Task 2: Add configuration and flat image discovery

**Files:**
- Create: `src/dataset_label/config.py`
- Create: `src/dataset_label/workflow.py`
- Create: `tests/test_config.py`
- Create: `tests/test_workflow.py`

**Interfaces:**
- Produces: `ProviderConfig(base_url: str, api_key: str, model: str)`.
- Produces: `load_provider_config(path: Path) -> ProviderConfig`.
- Produces: `discover_images(input_dir: Path) -> list[Path]`.
- Consumes: no product interfaces from later tasks.

- [ ] **Step 1: Write failing configuration and discovery tests**

```python
def test_load_provider_config_requires_all_provider_values(tmp_path):
    config_path = tmp_path / "config.local.toml"
    config_path.write_text("[provider]\nbase_url = 'https://x'\n")
    with pytest.raises(ValueError, match="api_key"):
        load_provider_config(config_path)

def test_discover_images_is_flat_sorted_and_case_insensitive(tmp_path):
    (tmp_path / "b.JPG").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "note.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inside.jpg").touch()
    assert [item.name for item in discover_images(tmp_path)] == ["a.png", "b.JPG"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_config.py tests/test_workflow.py -v`

Expected: FAIL because the loaders and discovery function do not exist.

- [ ] **Step 3: Implement strict TOML loading and flat discovery**

Implement `load_provider_config` with `tomllib`; reject missing files, absent `[provider]`, blank values, and non-string values with `ValueError`. Implement `discover_images` to require a directory, examine only direct files, use suffixes `.jpg`, `.jpeg`, `.png`, `.webp` case-insensitively, and return filename-sorted paths.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_config.py tests/test_workflow.py -v`

Expected: PASS.

### Task 3: Parse detections and serialize YOLO labels

**Files:**
- Create: `src/dataset_label/models.py`
- Create: `src/dataset_label/vision.py`
- Create: `src/dataset_label/yolo.py`
- Create: `tests/test_vision.py`
- Create: `tests/test_yolo.py`

**Interfaces:**
- Produces: `Detection(label: str, box: tuple[float, float, float, float])`.
- Produces: `parse_detections(content: str, allowed_labels: Sequence[str]) -> list[Detection]`.
- Produces: `to_yolo_row(detection: Detection, class_id: int) -> str`.
- Produces: `render_data_yaml(output_dir: Path, class_names: Sequence[str]) -> str`.
- Consumes: class-order sequence supplied by `cli.py` and `workflow.py`.

- [ ] **Step 1: Write failing response-parser and coordinate tests**

```python
def test_parse_detections_rejects_unrequested_category():
    with pytest.raises(ValueError, match="not requested"):
        parse_detections('[{"label":"汽车","bbox":[0,0,999,999]}]', ["人员", "安全帽"])

def test_to_yolo_row_converts_qwen_relative_box():
    row = to_yolo_row(Detection("人员", (100, 200, 600, 800)), 0)
    assert row == "0 0.350350 0.500501 0.500501 0.600601"

def test_boundary_coordinates_are_valid():
    assert parse_detections('[{"label":"人员","bbox":[0,0,999,999]}]', ["人员"])
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_vision.py tests/test_yolo.py -v`

Expected: FAIL because parsing and conversion functions do not exist.

- [ ] **Step 3: Implement `parse_detections` and YOLO serialization**

Require a top-level JSON array of objects with exactly usable `label` and four-number `bbox` values. Reject invalid JSON, missing fields, unknown labels, coordinate values outside `[0,999]`, and any box where `x2 <= x1` or `y2 <= y1`. Convert rows with six decimal places. Render `data.yaml` with `path`, `train: images`, `val:` omitted, and the ordered names mapping.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_vision.py tests/test_yolo.py -v`

Expected: PASS.

### Task 4: Implement OpenAI-compatible calls and the fail-fast output workflow

**Files:**
- Modify: `src/dataset_label/vision.py`
- Modify: `src/dataset_label/workflow.py`
- Modify: `src/dataset_label/cli.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_workflow.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ProviderConfig`, `discover_images`, `parse_detections`, `to_yolo_row`, and `render_data_yaml`.
- Produces: `OpenAICompatibleDetector(config: ProviderConfig)` with `detect(image_path: Path, labels: Sequence[str]) -> list[Detection]`.
- Produces: `label_directory(input_dir: Path, class_names: Sequence[str], detector: Detector) -> LabelingSummary`.

- [ ] **Step 1: Write failing workflow tests with a fake detector**

```python
def test_label_directory_writes_images_labels_and_data_yaml(tmp_path, fake_detector):
    source = make_image(tmp_path / "source" / "one.png")
    summary = label_directory(source.parent, ["人员", "安全帽"], fake_detector)
    output = source.parent / "yolo_output"
    assert (output / "images" / "one.png").is_file()
    assert (output / "labels" / "one.txt").read_text() == "0 0.350350 0.500501 0.500501 0.600601\n"
    assert "0: 人员" in (output / "data.yaml").read_text()

def test_existing_output_aborts_before_detector_is_called(tmp_path, fake_detector):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "yolo_output").mkdir()
    with pytest.raises(FileExistsError):
        label_directory(input_dir, ["人员"], fake_detector)
    assert fake_detector.calls == []
```

- [ ] **Step 2: Run the targeted workflow tests to verify they fail**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_workflow.py -v`

Expected: FAIL because the detector and directory workflow do not exist.

- [ ] **Step 3: Implement `OpenAICompatibleDetector.detect`**

Instantiate `openai.OpenAI` from `ProviderConfig`, encode the image as a MIME-appropriate Base64 Data URL, and call Chat Completions with an explicit JSON-only detection prompt. Pass the returned content to `parse_detections`; propagate all transport and parse failures unchanged.

- [ ] **Step 4: Implement `label_directory` and wire it to `main`**

Check output-directory absence before constructing or invoking the detector. Create `images` and `labels`, process discovered files one by one, write the label before copying its source image, then write `data.yaml` only after all images succeed. Print image and per-class totals and return exit code 0. Catch expected user errors in `main`, print the error to stderr, and return a nonzero code.

- [ ] **Step 5: Run the workflow and CLI tests to verify they pass**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest tests/test_workflow.py tests/test_cli.py -v`

Expected: PASS.

### Task 5: Run the full verification suite and document the real invocation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: completed `dataset-label` CLI.
- Produces: user-facing setup and run instructions consistent with actual behavior.

- [ ] **Step 1: Add the exact local setup, configuration, and invocation instructions**

Document copying `config.example.toml` to `config.local.toml`, filling a newly generated Key locally, and running:

```text
dataset-label "C:\\Project\\Dataset\\_label\\test1" --classes 人员 安全帽
```

State that the output is `<input-dir>\\yolo_output`, does not overwrite existing output, and stops on its first error.

- [ ] **Step 2: Run the complete test suite**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pytest -v`

Expected: PASS with no network calls.

- [ ] **Step 3: Verify package installation and CLI discovery**

Run: `C:\software\miniconda3\envs\dataset_label\python.exe -m pip install -e .` followed by `C:\software\miniconda3\envs\dataset_label\Scripts\dataset-label.exe --help`

Expected: installation succeeds and help displays the input directory and `--classes` arguments.

## Self-Review

- Spec coverage: Tasks 1–5 cover local configuration, generic OpenAI-compatible calls, flat discovery, deterministic serial processing, strict JSON and coordinate validation, YOLO conversion, unsplit output, fail-fast behavior, output collision prevention, and automated verification.
- Step scan: Each task begins with a targeted failing test, has a precise implementation boundary, and ends with a concrete verification command.
- Type consistency: `ProviderConfig` flows from `config.py` to `OpenAICompatibleDetector`; `Detection` flows from `vision.py` through `yolo.py` to `workflow.py`; class names remain ordered sequences throughout.
- Review focus: all five identified high-risk inputs are assigned to explicit Task 2–4 tests.
- Proportion: the plan specifies interfaces and expected test behavior without embedding full production implementations.

## Repository Note

The current workspace is not a Git repository, so commit steps are intentionally omitted. If Git is initialized later, each completed task should be committed independently.
