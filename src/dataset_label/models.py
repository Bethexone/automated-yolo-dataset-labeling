from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple[float, float, float, float]
