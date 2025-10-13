"""Helpers for loading Qt translation catalogues at application start."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from PyQt5 import QtCore, QtWidgets


def _normalise_locale_codes(locales: Sequence[str] | None) -> List[str]:
    """Return a list of normalised locale codes in preference order."""

    if not locales:
        system_locale = QtCore.QLocale.system()
        locales = [*system_locale.uiLanguages(), system_locale.name()]

    seen: set[str] = set()
    ordered: list[str] = []
    for entry in locales:
        if not entry:
            continue
        key = entry.replace("-", "_")
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
            if "_" in key:
                language = key.split("_", 1)[0]
                if language and language not in seen:
                    ordered.append(language)
                    seen.add(language)
    return ordered


def _candidate_paths(
    directory: Path, prefix: str, locale: str
) -> Iterable[Path]:
    """Yield possible translation file paths for a locale."""

    yield directory / f"{prefix}_{locale}.qm"
    if "_" in locale:
        language = locale.split("_", 1)[0]
        yield directory / f"{prefix}_{language}.qm"


@dataclass
class TranslationConfig:
    """Configuration describing how to locate translation catalogues."""

    directories: Sequence[Path | str] = field(default_factory=list)
    locales: Sequence[str] = field(default_factory=tuple)
    file_prefix: str = "yam_processor"


class TranslationLoader(QtCore.QObject):
    """Load and register Qt translation catalogues for the application."""

    def __init__(
        self,
        app: QtWidgets.QApplication,
        config: TranslationConfig,
    ) -> None:
        super().__init__(app)
        self._app = app
        self._config = config
        self._translators: list[QtCore.QTranslator] = []

    def install(self) -> list[Path]:
        """Load translations and install the translators on the application."""

        self.remove()
        loaded: list[Path] = []
        locales = _normalise_locale_codes(self._config.locales)
        directories = [Path(path) for path in self._config.directories]

        for directory in directories:
            if not directory.exists():
                continue
            for locale in locales:
                for candidate in _candidate_paths(directory, self._config.file_prefix, locale):
                    if not candidate.exists() or candidate in loaded:
                        continue
                    translator = QtCore.QTranslator(self._app)
                    if translator.load(str(candidate)):
                        self._app.installTranslator(translator)
                        self._translators.append(translator)
                        loaded.append(candidate)
        return loaded

    def remove(self) -> None:
        """Remove previously installed translators from the application."""

        while self._translators:
            translator = self._translators.pop()
            self._app.removeTranslator(translator)


def bootstrap_translations(
    app: QtWidgets.QApplication,
    config: TranslationConfig,
) -> TranslationLoader:
    """Create and install a :class:`TranslationLoader` for the application."""

    loader = TranslationLoader(app, config)
    loader.install()
    return loader


__all__ = [
    "TranslationConfig",
    "TranslationLoader",
    "bootstrap_translations",
]
