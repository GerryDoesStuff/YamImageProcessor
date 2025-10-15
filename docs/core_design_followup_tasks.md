# Core Design Compliance Follow-Up Tasks

This document tracks remaining work required to align the current implementation with the **Design Document: Modular Microscopic Image Processing App (YamImageProcessor)**.

## Updates and Maintenance
- Implement a real update polling workflow that checks a remote endpoint, surfaces update availability in the UI, and blocks background execution until the changelog preview acknowledged.
- Present release notes/changelogs before applying updates so users can review changes prior to installation.
- Add cryptographic signature verification for downloaded modules/updates to ensure supply-chain integrity during plugin loading.

## Scalability Enhancements
- Introduce tiled and/or lazy image loading so that very large microscopy files can be inspected without eagerly materialising the whole array (`yam_processor/data/image_io.py`).
- Extend preview rendering widgets to stream or progressively refine large images instead of rendering the full-resolution frame in one pass (`yam_processor/ui/dialogs.py`).

## Testing and Quality Tooling
- Add UI-focused automated tests (for example via `pytest-qt`) that exercise window/dock behaviour in line with the design specification.
- Configure repository-wide formatting and linting (e.g., Black + Flake8) together with mypy type checking and ensure they run in CI.
- Introduce a CI pipeline (GitHub Actions or similar) that executes the full test suite and style/type checks on every change.
