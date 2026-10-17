from pathlib import Path
from collections.abc import Sequence
from .models import Detection


def to_yolo_row(detection: Detection, class_id: int) -> str:
    x1, y1, x2, y2 = detection.box
    return f"{class_id} {(x1+x2)/2/1000:.6f} {(y1+y2)/2/1000:.6f} {(x2-x1)/1000:.6f} {(y2-y1)/1000:.6f}"


def render_data_yaml(output_dir: Path, class_names: Sequence[str]) -> str:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    return f"path: {output_dir.as_posix()}\ntrain: images\nnames:\n{names}\n"
