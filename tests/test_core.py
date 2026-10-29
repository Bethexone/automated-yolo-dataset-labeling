import pytest

from dataset_label.config import load_provider_config
from dataset_label.config import load_provider_configs
from dataset_label.models import Detection
from dataset_label.vision import build_detection_prompt
from dataset_label.vision import parse_detections
from dataset_label.workflow import discover_images
from dataset_label.workflow import label_directory
from dataset_label.yolo import to_yolo_row


def test_config_requires_api_key(tmp_path):
    path = tmp_path / "config.local.toml"
    path.write_text("[provider]\nbase_url='https://x'\nmodel='m'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_provider_config(path)


def test_config_loads_named_provider_profiles(tmp_path):
    path = tmp_path / "config.local.toml"
    path.write_text('[providers.qwen]\nbase_url="https://x"\napi_key="k"\nmodel="m"\n', encoding="utf-8")
    assert load_provider_configs(path)["qwen"].model == "m"


def test_discovery_is_flat_sorted_and_case_insensitive(tmp_path):
    (tmp_path / "b.JPG").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inside.jpg").touch()
    assert [p.name for p in discover_images(tmp_path)] == ["a.png", "b.JPG"]


def test_parser_rejects_unknown_category():
    with pytest.raises(ValueError, match="not requested"):
        parse_detections('{"detections":[{"label":"汽车","bbox":[0,0,999,999]}]}', ["人员"])


def test_parser_reads_detections_from_json_object():
    result = parse_detections('{"detections":[{"label":"person","bbox":[0,0,999,999]}]}', ["person"])
    assert result == [Detection("person", (0.0, 0.0, 999.0, 999.0))]


def test_yolo_row_converts_relative_coordinates():
    assert to_yolo_row(Detection("人员", (100, 200, 600, 800)), 0) == "0 0.350000 0.500000 0.500000 0.600000"


def test_parser_accepts_qwen_coordinate_boundary_1000():
    result = parse_detections('{"detections":[{"label":"person","bbox":[977,297,1000,336]}]}', ["person"])
    assert result[0].box == (977.0, 297.0, 1000.0, 336.0)


def test_detection_prompt_requires_the_detections_key_even_when_empty():
    prompt = build_detection_prompt(["person", "safety helmet"])

    assert '"detections"' in prompt
    assert 'exactly' in prompt
    assert 'empty list' in prompt


def test_workflow_writes_complete_unsplit_dataset(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "one.png").write_bytes(b"image")

    class Detector:
        def detect(self, image_path, labels):
            return [Detection("人员", (100, 200, 600, 800))]

    summary = label_directory(source, ["人员", "安全帽"], Detector())
    output = source / "yolo_output"
    assert summary == {"images": 1, "人员": 1, "安全帽": 0}
    assert (output / "images" / "one.png").read_bytes() == b"image"
    assert (output / "labels" / "one.txt").read_text() == "0 0.350000 0.500000 0.500000 0.600000\n"
    assert "0: 人员" in (output / "data.yaml").read_text(encoding="utf-8")
    assert (output / "classes.txt").read_text(encoding="utf-8") == "人员\n安全帽\n"


def test_workflow_writes_to_selected_export_root(tmp_path):
    from dataset_label.workflow import label_directory_to

    source = tmp_path / "input"
    source.mkdir()
    (source / "one.png").write_bytes(b"image")
    output = tmp_path / "chosen-output"

    class Detector:
        def detect(self, image_path, labels):
            return []

    label_directory_to(source, output, ["person"], Detector(), create_previews=False)
    assert (output / "images" / "one.png").is_file()
    assert (output / "classes.txt").read_text(encoding="utf-8") == "person\n"


def test_test_mode_saves_first_image_visualization(tmp_path):
    from PIL import Image
    from dataset_label.workflow import test_image

    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (100, 100), "white").save(source / "one.png")

    class Detector:
        def detect(self, image_path, labels):
            return [Detection("person", (100, 200, 600, 800))]

    assert test_image(source, ["person"], Detector()) == ["0 0.350000 0.500000 0.500000 0.600000"]
    assert (source / "yolo_output" / "preview" / "one.png").is_file()
    assert not (source / "yolo_output" / "data.yaml").exists()


def test_workflow_saves_visualizations_when_preview_is_enabled(tmp_path):
    from PIL import Image

    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (100, 100), "white").save(source / "one.png")

    class Detector:
        def detect(self, image_path, labels):
            return [Detection("person", (100, 200, 600, 800))]

    label_directory(source, ["person"], Detector(), create_previews=True)
    assert (source / "yolo_output" / "preview" / "one.png").is_file()
