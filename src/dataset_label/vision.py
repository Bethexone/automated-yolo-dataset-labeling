import json
import base64
import mimetypes
from pathlib import Path
from collections.abc import Sequence
from .models import Detection
from .config import ProviderConfig


def parse_detections(content: str, allowed_labels: Sequence[str]) -> list[Detection]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON response") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("detections"), list):
        raise ValueError("response must be a JSON object with detections")
    result = []
    for item in payload["detections"]:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ValueError("detection requires label")
        if item["label"] not in allowed_labels:
            raise ValueError(f"category {item['label']} is not requested")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in box):
            raise ValueError("detection requires four numeric bbox coordinates")
        x1, y1, x2, y2 = map(float, box)
        if not all(0 <= value <= 1000 for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid bbox coordinates for {item['label']}: {box}")
        result.append(Detection(item["label"], (x1, y1, x2, y2)))
    return result


class OpenAICompatibleDetector:
    def __init__(self, config: ProviderConfig):
        from openai import OpenAI
        self.config, self.client = config, OpenAI(api_key=config.api_key, base_url=config.base_url)

    def detect(self, image_path: Path, labels: Sequence[str]) -> list[Detection]:
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = f"Detect only these categories: {list(labels)}. Return JSON object {{\"detections\":[{{\"label\":string,\"bbox\":[x1,y1,x2,y2]}}]}}. Coordinates must be integers from 0 to 1000."
        response = self.client.chat.completions.create(model=self.config.model, response_format={"type": "json_object"}, messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}, {"type": "text", "text": prompt}]}])
        return parse_detections(response.choices[0].message.content or "", labels)
