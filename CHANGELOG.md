# Changelog

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
