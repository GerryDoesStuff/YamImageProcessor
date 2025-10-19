"""Controller-backed pipeline execution helpers."""
from __future__ import annotations

import logging
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from plugins.module_base import ModuleStage
from processing.pipeline_manager import PipelineImage, PipelineStep as ManagedPipelineStep
from ui.unified import UnifiedPipelineController

LOGGER = logging.getLogger(__name__)


def coerce_pipeline_image(image: PipelineImage | object) -> np.ndarray:
    """Convert ``image`` to a dense :class:`numpy.ndarray` copy."""

    if isinstance(image, np.ndarray):
        return np.array(image, copy=True)

    to_array = getattr(image, "to_array", None)
    if callable(to_array):
        try:
            array = to_array()
        except Exception:  # pragma: no cover - defensive conversion
            LOGGER.debug("Failed to materialise tiled pipeline image", exc_info=True)
        else:
            return np.array(array, copy=True)

    try:
        return np.array(np.asarray(image), copy=True)
    except Exception:  # pragma: no cover - defensive coercion
        LOGGER.debug("Failed to coerce pipeline image result", exc_info=True)
        raise


class ControllerBackedPipeline:
    """Expose ``apply`` semantics backed by ``UnifiedPipelineController``."""

    def __init__(
        self,
        controller: UnifiedPipelineController,
        stage: ModuleStage,
        *,
        source_resolver: Optional[Callable[[PipelineImage], PipelineImage]] = None,
        seed_resolver: Optional[
            Callable[[PipelineImage], Mapping[ModuleStage, PipelineImage]]
        ] = None,
    ) -> None:
        self._controller = controller
        self._stage = stage
        self._source_resolver = source_resolver
        self._seed_resolver = seed_resolver

    @property
    def steps(self) -> Tuple[ManagedPipelineStep, ...]:
        """Return cloned pipeline steps owned by the configured stage."""

        try:
            cached = self._controller.cached_stage_steps(self._stage)
        except Exception:  # pragma: no cover - defensive access
            LOGGER.debug("Failed to access cached steps for stage %s", self._stage, exc_info=True)
            return ()
        return tuple(step.clone() for step in cached)

    def apply(self, image: PipelineImage) -> np.ndarray:
        """Execute enabled steps for ``stage`` using ``controller``."""

        source_input = image
        if self._source_resolver is not None:
            try:
                source_input = self._source_resolver(image)
            except Exception:  # pragma: no cover - defensive source resolution
                LOGGER.exception("Failed to resolve controller pipeline source input")
                raise

        source_array = (
            source_input
            if isinstance(source_input, np.ndarray)
            else np.asarray(source_input)
        )

        seeds: Optional[Mapping[ModuleStage, PipelineImage]] = None
        if self._seed_resolver is not None:
            try:
                resolved = self._seed_resolver(image)
            except Exception:  # pragma: no cover - defensive seed resolution
                LOGGER.exception("Failed to resolve controller pipeline seeds")
                raise
            else:
                if resolved:
                    seeds = {
                        stage: (
                            value
                            if isinstance(value, np.ndarray)
                            else np.asarray(value)
                        )
                        for stage, value in resolved.items()
                    }
        try:
            self._controller.run_enabled_stages(
                source_array, seeded_results=seeds
            )
        except Exception:  # pragma: no cover - defensive execution
            LOGGER.exception("Controller-backed pipeline execution failed for %s", self._stage)
            raise
        result = self._controller.cached_stage_result(self._stage)
        if result is None:
            return np.array(source_array, copy=True)
        return coerce_pipeline_image(result)


__all__ = ["ControllerBackedPipeline", "coerce_pipeline_image"]
