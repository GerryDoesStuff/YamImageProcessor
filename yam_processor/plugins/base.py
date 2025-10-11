"""Base classes for plugin implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModuleMetadata:
    name: str
    version: str = "0.1.0"
    description: str = ""


class ModuleBase(Protocol):
    metadata: ModuleMetadata

    def register(self, app_core: "AppCore") -> None:
        ...
