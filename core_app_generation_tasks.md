# Outstanding Tasks for Core App Generation

The current repository centers around a monolithic `preprocessing22.py` that wires UI widgets, pipeline logic, and persistence directly together, providing the preprocessing menu flow and JSON import/export helpers in a single module. To fulfill the modular architecture and robustness goals laid out in the core design document, the following work remains.

## Foundational Setup
- Restructure the codebase into explicit `core/`, `processing/`, `ui/`, and `plugins/` packages so that pipeline logic, Qt widgets, and future module discovery are not colocated in a single file.
- Introduce an `AppCore` bootstrap that owns application lifecycle, logging initialization, settings manager instantiation, and background thread services instead of having the main window construct these concerns itself.
- Replace the global `logging.basicConfig` usage with a centralized logger supporting rotation, anonymized formatting, and developer diagnostics toggles configurable at runtime.
- Promote the scattered `QtCore.QSettings` access into a dedicated settings manager that exposes JSON import/export through explicit APIs and can be reused by other modules without duplicating keys.

## Processing and Pipeline Management
- Extract a pipeline manager service that tracks ordered steps, supports reordering and enable/disable toggles, and surfaces undo/redo history independent of the main window widgets.
- Implement asynchronous execution helpers (e.g., thread pools with progress and cancellation signals) so long-running preprocessing or future segmentation steps do not block the GUI thread.
- Define base classes and registration contracts for preprocessing/segmentation/analysis modules to enable plugin discovery instead of hardcoding menu actions in the window class.

## User Interface Foundations
- Rebuild the main window using dockable panels for pipeline overview, diagnostics, and module parameters while keeping the layout DPI-aware and keyboard accessible.
- Expand the menu system to match the design spec (File, Edit, View, Modules, Help) and allow context menus/shortcuts to map to the refactored core actions.
- Convert modal parameter dialogs into non-blocking panels or dialogs that stream preview updates via signals without halting the event loop.

## Data Handling and Persistence
- Broaden image I/O utilities to cover the required formats (including `.npy`) and persist metadata alongside processed outputs instead of the current limited save helpers.
- Implement autosave, backup-before-overwrite safeguards, and JSON sidecar generation that are driven by the centralized settings manager rather than manual file dialogs.
- Cache intermediate pipeline states in a reproducibility store so undo/redo and module previews share the same data layer rather than recomputing directly off the UI state.

## Accessibility and Localization
- Provide high-contrast themes, scalable icons/fonts, and verified keyboard navigation across all widgets beyond the current basic shortcuts.
- Wire Qt translation (`QTranslator`) hooks so text resources can load language packs at runtime.
- Audit all controls to supply descriptive tooltips and recommended ranges through a shared metadata system rather than ad hoc widget defaults.

## Reliability, Security, and Diagnostics
- Replace simple message boxes with structured error dialogs that capture stack traces, anonymize paths, and offer safe recovery actions.
- Add an in-app diagnostics panel surfaced via the refactored logger to monitor threads, module health, and recent log entries.
- Implement crash recovery workflows, isolated temp directories, and input sanitization utilities as part of the core services layer.

## Performance and Extensibility Enhancements
- Optimize pipelines using in-place NumPy operations, caching, and optional GPU hooks exposed through the processing layer rather than embedding logic per dialog callback.
- Build plugin discovery under a `/modules` directory using `importlib`, with graceful degradation when modules fail to load.
- Surface version metadata, update-checker stubs, and debug-only telemetry toggles inside the new core layer to satisfy maintenance requirements.

## Quality Assurance and Documentation
- Establish automated tests covering pipeline execution, settings round-trips, and module failure isolation instead of relying on manual experimentation.
- Produce developer documentation describing module authoring, logging conventions, accessibility standards, and recovery procedures for contributors.
- Draft a roadmap for future GPU acceleration and OpenCV/scikit-image optimizations so the architecture anticipates performance extensions beyond the current CPU-only code.
