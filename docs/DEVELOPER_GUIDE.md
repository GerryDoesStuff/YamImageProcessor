# Developer Guide

This guide captures the conventions to follow when extending Yam Image Processor
with new modules, hooking them into the application runtime, and ensuring
observability and accessibility parity across the project.

## Pipeline plugin architecture

Processing features are authored as subclasses of the base classes defined in
[`yam_processor.plugins.base`](../yam_processor/plugins/base.py). Each subclass
must declare which pipeline stage it belongs to by inheriting from the
appropriate ABC:

- `PreprocessingModule` for transforms that operate on raw images before
  segmentation.
- `SegmentationModule` for algorithms that derive masks or labelled imagery.
- `AnalysisModule` for components that turn segmented imagery into structured
  metrics.

All module classes must implement the abstract `process` method for their
stage, expose a `metadata` property that returns a
`ModuleMetadata` instance, and optionally override
`capabilities`, `parameter_schema`, or `preview` when additional configuration or
UI integration is required. Module metadata is displayed throughout the UI and
is used when serialising pipelines, so keep the values descriptive and stable.

### Performance roadmap

Developers planning GPU-accelerated features should consult the
[`docs/performance_roadmap.md`](performance_roadmap.md) document. It outlines the
target backends, the expected usage of `ModuleCapabilities.requires_gpu` and
`PipelineStep.execution.requires_gpu`, and how OpenCV or scikit-image kernels
will be wired into the pipeline manager. Aligning new modules with the roadmap
ensures upcoming executors can adopt them without refactoring.

### Example skeleton

```python
import numpy as np

from yam_processor.plugins.base import (
    ModuleMetadata,
    ModuleCapabilities,
    PreprocessingModule,
)


class MyNormalisationModule(PreprocessingModule):
    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            name="Normalise Intensities",
            version="1.0.0",
            description="Scales image intensities between 0 and 1.",
        )

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(supports_batch=True)

    def process(self, image: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
        scaled = (image - image.min()) / (image.max() - image.min() + epsilon)
        return scaled.astype(image.dtype, copy=False)
```

Whenever a module exposes configuration to end users, implement
`parameter_schema` to return the sequence of `ParameterSpec` controls used by the
configuration dialogs. Use the optional `preview` hook to provide a lightweight
visualisation path that avoids mutating the original buffer in place.

## Registering modules with `AppCore`

Plugin packages are discovered by the application's module registry, and each
package must expose a module-level `register_module(app_core)` function. Use the
helper provided in `yam_processor.plugins.base` to register each subclass with
`AppCore`:

```python
from yam_processor.plugins import base


def register_module(app_core):
    base.register_module(app_core, MyNormalisationModule)
```

The helper enforces that at least one module class is supplied and that every
class inherits from `ModuleBase`. During application bootstrap the registry adds
these classes to the pipeline catalogue so the UI can present them for
insertion.

## Logging expectations

Logging is configured globally by `yam_processor.core.logging_config.LoggingConfigurator`.
It installs a rotating file handler and (optionally) a console handler with the
format:

```
%(asctime)s | %(levelname)s | %(component)s | %(message)s
```

To keep log output consistent:

- Acquire loggers with `logging.getLogger(__name__)` so records can be traced
  back to their module.
- When emitting structured diagnostics, provide the `component` field through the
  `extra` dictionary (for example, `logger.info("Started", extra={"component": "MyModule"})`).
  The formatter falls back to `record.name` if no component is supplied, but
  setting it explicitly allows UI surfaces (like the diagnostics panel) to group
  related messages.
- Use the `developer_diagnostics` flag in `LoggingOptions` when you need verbose
  debugging output; this automatically elevates the root logger to `DEBUG` and
  prints non-anonymised file and line information to the console.
- Always prefer structured metadata over interpolated strings when logging error
  conditions so the diagnostics panel can present actionable context.

## Accessibility and UI conventions

`yam_processor.ui.main_window.MainWindow` enforces a number of accessibility
standards that UI contributors should maintain when adding or modifying
widgets:

- **High-contrast theme** – the window installs the Qt "Fusion" style and
  applies a dark, high-contrast palette that meets WCAG colour contrast goals.
  Any new widgets should respect palette roles rather than hard-coding colours.
- **High-DPI scaling** – layout margins, spacing, and fonts are scaled based on
  the logical DPI of the current screen. Avoid fixed pixel values; instead,
  derive measurements from the helper methods in `MainWindow` or use Qt layout
  defaults.
- **Keyboard navigation** – the main window builds tab order chains across the
  central widget and each dock, and assigns mnemonic shortcuts (`Alt+0` –
  `Alt+3`) for quick focus changes. Preserve focus policies on new widgets and
  connect them into the shortcut map when adding additional docks.

Adhering to these conventions ensures that accessibility settings remain
predictable for keyboard and screen-reader users while keeping the UI visually
coherent with the rest of the application.
