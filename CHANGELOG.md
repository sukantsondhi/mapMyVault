# Changelog

## [Unreleased]

### Fixed

- Mixed PDFs no longer skip scanned pages or image text when native text exists.
  Extraction retains native text, preserves page order, and counts processed OCR
  pages accurately. Previous OCR caches refresh on the next indexing run.
- Images with OCR disabled report `ocr_required` instead of unsupported-binary
  status. PDF candidates outside the OCR page limit report `ocr_skipped`.
- Missing-index HTTP status requests return JSON errors instead of disconnecting.
- Move and rollback plans revalidate the full batch, reject conflicting paths and
  blocked destinations, and cannot reapprove an already-applied plan.
- Subtree removal treats SQL wildcard characters in folder names literally.
- Zero-result counts no longer fall back to unrelated search results.
- Explorer updates preserve saved OCR and vision settings.
- Index changes clear stale chat, attachments, folder state, and session-owned MCP.
- Removed chat attachments stay removed; evidence remains available across reruns.
- Source selection no longer silently stops at 300 top-level items.
- Fresh installs retain MCP 1.x compatibility instead of loading the incompatible
  MCP 2.x API.

### Changed

- Redesigned Studio with shared index selection, local-service status, compact
  responsive layouts, and separate build and integration workflows.
- Replaced Graph View with a filterable Explorer, file previews, restorable index
  exclusions, and a graph/table of recorded relationships.
- OCR is opt-in. Streamlit usage statistics are disabled in repository configuration.
- Require Streamlit 1.49+ and declare the graph visualization dependency.
- Added HTTP, action-safety, index-correctness, and Streamlit workflow regressions.
- Added generated text/image/PDF fixtures and real Tesseract tests for recognition,
  mixed content, limits, renderer fallback, errors, and searchable persistence.

## [2.0.0] - 2026-06-05

### Added

- Strictly local Ollama generation and embedding endpoints.
- Resumable SQLite stage tracking and dependency-based invalidation.
- Content extraction, deterministic parsing evidence, semantic candidates, and
  evidence-backed relationships.
- Mirrored Obsidian vault plus canonical SQLite, Chroma, graph, manifest, and
  structured summary outputs.
- Persisted full-text queries and loopback-only MCP tools.
- Approval-gated move, rename, apply, rollback, and audit workflows.
- Offline doctor, v2 CLI, local integration guides, and automated tests.
- Interactive terminal setup for source, output, and dynamic local model selection.
- Pipeline-stage and per-file progress bars for mapping and resume operations.
- Beginner-friendly end-to-end guide covering Ollama, Obsidian, CLI queries,
  native Open WebUI MCP integration, and optional OpenClaw local-agent setup.
- Fully local content extraction for PDF, DOCX, XLSX, PPTX, CSV, and TSV files.
- Rich document explanations, topics, entities, dates, properties, suggested
  actions, and document-format MCP queries.
- Open WebUI/MCPO working setup documentation and `answer_question` evidence tool.
- Optional local OCR for scanned PDFs and images using Tesseract/Poppler.
- Local desktop app architecture notes for a future drag-and-drop UI.

### Changed

- Replaced the filename-only prototype pipeline and legacy JSON layout.
- Replaced duplicated prototype documentation with focused v2 documentation.
- Reduced documentation to a minimal maintained set: setup, Open WebUI/MCPO,
  reference, and troubleshooting.
- Raised the minimum supported Python version to 3.10.

## [1.0.0] - 2026-06-05

- Initial filename-based Obsidian vault mapping prototype.
