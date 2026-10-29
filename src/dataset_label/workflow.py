from pathlib import Path
import shutil
from collections.abc import Sequence
from typing import Protocol
from .models import Detection
from .yolo import render_data_yaml, to_yolo_row


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    return sorted((item for item in input_dir.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES), key=lambda item: item.name.lower())


class Detector(Protocol):
    def detect(self, image_path: Path, labels: Sequence[str]) -> list[Detection]: ...


def _render_preview(image_path: Path, detections: Sequence[Detection], class_names: Sequence[str], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    colors = ["red", "lime", "blue", "yellow", "magenta", "cyan"]
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            class_id = class_names.index(detection.label)
            color = colors[class_id % len(colors)]
            box = (x1 / 1000 * canvas.width, y1 / 1000 * canvas.height, x2 / 1000 * canvas.width, y2 / 1000 * canvas.height)
            draw.rectangle(box, outline=color, width=max(1, min(canvas.width, canvas.height) // 200))
            draw.text((box[0], max(0, box[1] - 12)), detection.label, fill=color, font=font)
        canvas.save(output_path)


def test_image(input_dir: Path, class_names: Sequence[str], detector: Detector) -> list[str]:
    images = discover_images(input_dir)
    if not images:
        raise ValueError("input directory contains no supported images")
    output = input_dir / "yolo_output"
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    preview_output = output / "preview"
    preview_output.mkdir(parents=True)
    image = images[0]
    detections = detector.detect(image, class_names)
    _render_preview(image, detections, class_names, preview_output / image.name)
    return [to_yolo_row(detection, class_names.index(detection.label)) for detection in detections]


def label_directory_to(input_dir: Path, output: Path, class_names: Sequence[str], detector: Detector, create_previews: bool = False, resume: bool = False) -> dict[str, int]:
    images = discover_images(input_dir)
    image_output, label_output = output / "images", output / "labels"
    completed = set()
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}")
        if not image_output.is_dir() or not label_output.is_dir():
            raise ValueError("resume output must contain images and labels directories")
        classes_path = output / "classes.txt"
        if classes_path.exists() and classes_path.read_text(encoding="utf-8").splitlines() != list(class_names):
            raise ValueError("resume output classes do not match requested classes")
        source_stems = {image.stem for image in images}
        for label_path in label_output.glob("*.txt"):
            if label_path.stem not in source_stems:
                raise ValueError(f"resume output contains an unknown label: {label_path.name}")
            if any(image.stem == label_path.stem and (image_output / image.name).is_file() for image in images):
                completed.add(label_path.stem)
    else:
        image_output.mkdir(parents=True)
        label_output.mkdir()
    preview_output = output / "preview"
    if create_previews:
        preview_output.mkdir(exist_ok=True)
    totals = {name: 0 for name in class_names}
    for image in images:
        if image.stem in completed:
            continue
        detections = detector.detect(image, class_names)
        lines = []
        for detection in detections:
            totals[detection.label] += 1
            lines.append(to_yolo_row(detection, class_names.index(detection.label)))
        (label_output / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        shutil.copy2(image, image_output / image.name)
        if create_previews:
            _render_preview(image, detections, class_names, preview_output / image.name)
    (output / "data.yaml").write_text(render_data_yaml(output, class_names), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return {"images": len(images), **totals}


def label_directory(input_dir: Path, class_names: Sequence[str], detector: Detector, create_previews: bool = False, resume: bool = False) -> dict[str, int]:
    return label_directory_to(input_dir, input_dir / "yolo_output", class_names, detector, create_previews, resume)
