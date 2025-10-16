# Core Design Compliance Follow-Up Tasks

The latest review of the **Design Document: Modular Microscopic Image Processing App (YamImageProcessor)** highlights that the
remaining gaps are concentrated in the scalability requirements (Section 15 of the design document).

## Outstanding Scalability Features

### 1. Add tiled and lazy image loading
* Extend `core/io_manager.py` so callers can request image handles that expose tile-wise/lazy access instead of always materialising a full NumPy array in memory.
* Introduce lightweight data structures (e.g., a tiled image record) in the processing layer for tracking tile geometry and fetching regions on demand.
* Update pipeline execution helpers to accept these tiled records and stream tiles through each enabled `PipelineStep` without forcing a full-frame copy first.
* Provide comprehensive tests (unit tests plus performance regressions) that exercise the new streaming path with large synthetic images.

### 2. Implement progressive preview rendering
* Replace the static `ImageDisplayWidget` preview in `ui/preprocessing.py` with a widget capable of requesting tiles progressively so that large datasets render incrementally while background processing continues.
* Wire the preview widget into the background `ThreadController` so partial results can be pushed to the UI without blocking the main thread.
* Ensure preview updates remain responsive when interacting with pipeline controls (zoom, pan, module parameter edits) and fall back gracefully for small images.
* Add UI tests (via `pytest-qt`) validating that progressive updates occur and that cancellation/cleanup paths leave the interface in a consistent state.

## Completed Design Requirements

Recent implementation work already covers the update notifications, changelog previews, telemetry gating, diagnostics panel, cryptographic plugin signature checks, and the repository-wide quality tooling described in Sections 12–16 of the design document.
