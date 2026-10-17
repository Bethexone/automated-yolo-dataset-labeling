from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    model: str


def load_provider_config(path: Path) -> ProviderConfig:
    with path.open("rb") as file:
        provider = tomllib.load(file).get("provider", {})
    values = []
    for key in ("base_url", "api_key", "model"):
        value = provider.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"provider.{key} is required")
        values.append(value)
    return ProviderConfig(*values)
