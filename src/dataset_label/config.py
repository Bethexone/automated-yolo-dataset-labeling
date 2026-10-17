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


def load_provider_configs(path: Path) -> dict[str, ProviderConfig]:
    with path.open("rb") as file:
        providers = tomllib.load(file).get("providers", {})
    if not isinstance(providers, dict) or not providers:
        return {"default": load_provider_config(path)}
    result = {}
    for name, provider in providers.items():
        if not isinstance(name, str) or not isinstance(provider, dict):
            raise ValueError("providers must contain named provider tables")
        values = []
        for key in ("base_url", "api_key", "model"):
            value = provider.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"providers.{name}.{key} is required")
            values.append(value)
        result[name] = ProviderConfig(*values)
    return result
