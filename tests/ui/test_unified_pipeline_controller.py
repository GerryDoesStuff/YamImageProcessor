from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - skip when Qt bindings missing
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    pytestmark = pytest.mark.skip(reason=f"PyQt5 unavailable: {exc}")
else:
    from plugins.module_base import ModuleStage
    from processing.pipeline_manager import PipelineManager, PipelineStep
    from ui.unified import UnifiedPipelineController


if QtCore is not None:

    def _noop_processor(image, **kwargs):
        return image


    @pytest.fixture()
    def qapp() -> QtWidgets.QApplication:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return app


    @pytest.fixture()
    def pipeline_manager() -> PipelineManager:
        steps = [
            PipelineStep(
                name="preprocess",
                function=_noop_processor,
                params={"gain": 1},
                stage=ModuleStage.PREPROCESSING,
            ),
            PipelineStep(
                name="segment",
                function=_noop_processor,
                params={"threshold": 0.25},
                stage=ModuleStage.SEGMENTATION,
            ),
            PipelineStep(
                name="extract",
                function=_noop_processor,
                params={"features": ["Hu"]},
                stage=ModuleStage.ANALYSIS,
            ),
        ]
        return PipelineManager(steps)


    @dataclass
    class _Probe:
        received: List[Tuple[PipelineStep, ...]]


    class _PaneProbe(QtCore.QObject):
        def __init__(
            self,
            controller: UnifiedPipelineController,
            stage: ModuleStage,
            probe: _Probe,
        ) -> None:
            super().__init__(controller)
            self._stage = stage
            self._probe = probe
            controller.stage_cache_updated.connect(self._on_stage_cache_updated)

        def _on_stage_cache_updated(
            self, stage: ModuleStage, steps: Tuple[PipelineStep, ...]
        ) -> None:
            if stage == self._stage:
                self._probe.received.append(steps)


    class _StubAppCore:
        def __init__(self, manager: PipelineManager) -> None:
            self._manager = manager

        def get_pipeline_manager(self) -> PipelineManager:
            return self._manager


    @pytest.fixture()
    def controller(pipeline_manager: PipelineManager) -> UnifiedPipelineController:
        return UnifiedPipelineController(_StubAppCore(pipeline_manager))


    @pytest.fixture()
    def segmentation_probe(controller: UnifiedPipelineController) -> _Probe:
        probe = _Probe(received=[])
        _PaneProbe(controller, ModuleStage.SEGMENTATION, probe)
        return probe


    @pytest.fixture()
    def extraction_probe(controller: UnifiedPipelineController) -> _Probe:
        probe = _Probe(received=[])
        _PaneProbe(controller, ModuleStage.ANALYSIS, probe)
        return probe


    def test_pipeline_controller_updates_downstream_caches(
        qapp: QtWidgets.QApplication,
        controller: UnifiedPipelineController,
        segmentation_probe: _Probe,
        extraction_probe: _Probe,
        pipeline_manager: PipelineManager,
    ) -> None:
        controller.recompute_pipeline()

        assert len(segmentation_probe.received) == 1
        assert len(extraction_probe.received) == 1

        initial_segmentation_steps = segmentation_probe.received[-1]
        initial_extraction_steps = extraction_probe.received[-1]

        initial_bounds = {
            stage: controller.pipeline_stage_bounds(stage)
            for stage in ModuleStage
        }
        initial_combined_cache = controller.cached_pipeline()
        initial_pre_cache = controller.cached_stage_steps(ModuleStage.PREPROCESSING)
        assert initial_pre_cache[0].params["gain"] == 1

        def _mutate_gain(step: PipelineStep) -> None:
            step.params["gain"] = 5

        controller.mutate_stage_step(
            ModuleStage.PREPROCESSING,
            index=0,
            mutator=_mutate_gain,
        )

        assert len(segmentation_probe.received) == 2
        assert len(extraction_probe.received) == 2

        updated_segmentation_steps = segmentation_probe.received[-1]
        updated_extraction_steps = extraction_probe.received[-1]
        assert updated_segmentation_steps is not initial_segmentation_steps
        assert updated_extraction_steps is not initial_extraction_steps
        assert updated_segmentation_steps[0] is not initial_segmentation_steps[0]
        assert updated_extraction_steps[0] is not initial_extraction_steps[0]

        updated_bounds = {
            stage: controller.pipeline_stage_bounds(stage)
            for stage in ModuleStage
        }
        assert updated_bounds == initial_bounds

        updated_pre_cache = controller.cached_stage_steps(ModuleStage.PREPROCESSING)
        assert updated_pre_cache is not initial_pre_cache
        assert updated_pre_cache[0].params["gain"] == 5

        updated_combined_cache = controller.cached_pipeline()
        assert updated_combined_cache is not initial_combined_cache
        assert updated_combined_cache[0].params["gain"] == 5

        live_pre_steps = controller.stage_steps(ModuleStage.PREPROCESSING)
        assert live_pre_steps[0].params["gain"] == 5

        assert pipeline_manager.steps[0].params["gain"] == 5

    def test_run_enabled_stages_caches_results(
        qapp: QtWidgets.QApplication,
    ) -> None:
        base_image = np.array([[1, 2], [3, 4]], dtype=np.float32)

        observed_inputs: Dict[str, np.ndarray] = {}

        def _pre(image, **kwargs):
            observed_inputs["pre"] = np.array(image, copy=True)
            return image + 1

        def _seg(image, **kwargs):
            observed_inputs["seg"] = np.array(image, copy=True)
            return image * 2

        def _extract(image, **kwargs):
            observed_inputs["extract"] = np.array(image, copy=True)
            return image - 3

        manager = PipelineManager(
            [
                PipelineStep(
                    name="preprocess",
                    function=_pre,
                    stage=ModuleStage.PREPROCESSING,
                ),
                PipelineStep(
                    name="segment",
                    function=_seg,
                    stage=ModuleStage.SEGMENTATION,
                ),
                PipelineStep(
                    name="extract",
                    function=_extract,
                    stage=ModuleStage.ANALYSIS,
                ),
            ]
        )
        controller = UnifiedPipelineController(_StubAppCore(manager))

        results = controller.run_enabled_stages(base_image)

        np.testing.assert_array_equal(observed_inputs["pre"], base_image)
        np.testing.assert_array_equal(observed_inputs["seg"], base_image + 1)
        np.testing.assert_array_equal(
            observed_inputs["extract"], (base_image + 1) * 2
        )

        pre_result = controller.cached_stage_result(ModuleStage.PREPROCESSING)
        seg_result = controller.cached_stage_result(ModuleStage.SEGMENTATION)
        extract_result = controller.cached_stage_result(ModuleStage.ANALYSIS)

        assert pre_result is not None
        assert seg_result is not None
        assert extract_result is not None

        np.testing.assert_array_equal(pre_result, base_image + 1)
        np.testing.assert_array_equal(seg_result, (base_image + 1) * 2)
        np.testing.assert_array_equal(extract_result, ((base_image + 1) * 2) - 3)

        expected_dependencies = {
            ModuleStage.PREPROCESSING: (),
            ModuleStage.SEGMENTATION: (ModuleStage.PREPROCESSING,),
            ModuleStage.ANALYSIS: (
                ModuleStage.PREPROCESSING,
                ModuleStage.SEGMENTATION,
            ),
        }
        for stage, deps in expected_dependencies.items():
            assert controller.stage_dependencies(stage) == deps

        cached_results = controller.cached_stage_results()
        assert cached_results.keys() == expected_dependencies.keys()
        for stage, result in cached_results.items():
            np.testing.assert_array_equal(result, results[stage])

    def test_mutating_mid_stage_invalidates_downstream_only(
        qapp: QtWidgets.QApplication,
    ) -> None:
        base_image = np.arange(9, dtype=np.float32).reshape(3, 3)

        manager = PipelineManager(
            [
                PipelineStep(
                    name="preprocess",
                    function=lambda image, **_: image + 1,
                    stage=ModuleStage.PREPROCESSING,
                ),
                PipelineStep(
                    name="segment",
                    function=lambda image, **_: image * 2,
                    stage=ModuleStage.SEGMENTATION,
                ),
                PipelineStep(
                    name="extract",
                    function=lambda image, **_: image - 5,
                    stage=ModuleStage.ANALYSIS,
                ),
            ]
        )
        controller = UnifiedPipelineController(_StubAppCore(manager))

        controller.run_enabled_stages(base_image)
        assert controller.cached_stage_result(ModuleStage.PREPROCESSING) is not None
        assert controller.cached_stage_result(ModuleStage.SEGMENTATION) is not None
        assert controller.cached_stage_result(ModuleStage.ANALYSIS) is not None

        manager.toggle_step("segment")

        assert controller.cached_stage_result(ModuleStage.PREPROCESSING) is not None
        assert controller.cached_stage_result(ModuleStage.SEGMENTATION) is None
        assert controller.cached_stage_result(ModuleStage.ANALYSIS) is None

        seg_steps = controller.cached_stage_steps(ModuleStage.SEGMENTATION)
        assert seg_steps and not seg_steps[0].enabled

        controller.run_enabled_stages(base_image)
        assert controller.cached_stage_result(ModuleStage.SEGMENTATION) is not None
        assert controller.cached_stage_result(ModuleStage.ANALYSIS) is not None

    def test_mutating_upstream_stage_invalidates_all_results(
        qapp: QtWidgets.QApplication,
    ) -> None:
        base_image = np.ones((2, 2), dtype=np.float32)

        manager = PipelineManager(
            [
                PipelineStep(
                    name="preprocess",
                    function=lambda image, **_: image + 2,
                    stage=ModuleStage.PREPROCESSING,
                ),
                PipelineStep(
                    name="segment",
                    function=lambda image, **_: image * 3,
                    stage=ModuleStage.SEGMENTATION,
                ),
                PipelineStep(
                    name="extract",
                    function=lambda image, **_: image - 1,
                    stage=ModuleStage.ANALYSIS,
                ),
            ]
        )
        controller = UnifiedPipelineController(_StubAppCore(manager))

        controller.run_enabled_stages(base_image)
        for stage in ModuleStage:
            assert controller.cached_stage_result(stage) is not None

        manager.toggle_step("preprocess")

        for stage in ModuleStage:
            assert controller.cached_stage_result(stage) is None

else:  # pragma: no cover - executed only when Qt bindings missing

    def test_unified_pipeline_controller_skipped() -> None:
        pytest.skip("PyQt5 unavailable")
