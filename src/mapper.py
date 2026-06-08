"""Resumable, content-aware, fully local repository mapper."""

from pathlib import Path
from typing import Callable, Dict, Optional
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
import sys

from tqdm import tqdm

from .analyzer import (
    _resolve_local_yolo_model,
    deterministic_edges,
    extract_content,
    parse_content,
    scan_repository,
)
from .config import MapperConfig
from .local_ai import LocalAI
from .storage import IndexStore, utc_now
from .vault_export import atomic_write, export_vault
from .vector_index import VectorIndex


PIPELINE_VERSION = "2.3.0"


RETRYABLE_EXTRACTION_STATUSES = {
    "ocr_failed",
    "ocr_unavailable",
}
RETRYABLE_VISION_STATUSES = {
    "vision_failed",
    "vision_unavailable",
}


class OCRUnavailableError(RuntimeError):
    pass


def _metadata_status(row) -> str:
    try:
        return json.loads(row["document_metadata_json"] or "{}").get(
            "extraction_status", ""
        )
    except json.JSONDecodeError:
        return ""


def _metadata_value(row, key: str) -> str:
    try:
        return json.loads(row["document_metadata_json"] or "{}").get(key, "")
    except json.JSONDecodeError:
        return ""


class RepositoryMapper:
    def __init__(
        self,
        repository: Path,
        output: Path,
        config: Optional[MapperConfig] = None,
        ai=None,
        use_chroma: bool = True,
        vector_index=None,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        file_progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        warning_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.repository = repository.resolve()
        self.output = output.resolve()
        if self.repository == self.output or self.repository in self.output.parents:
            raise ValueError("Output must not be inside the indexed repository")
        if not self.repository.is_dir():
            raise ValueError(f"Repository not found: {self.repository}")
        self.config = config or MapperConfig()
        self.data_dir = self.output / "data"
        self.vault_dir = self.output / "obsidian"
        self.summary_dir = self.data_dir / "summaries"
        self.store = IndexStore(self.data_dir / "index.sqlite")
        self.store.set_metadata("repository", str(self.repository))
        self.store.set_metadata("output", str(self.output))
        self.store.set_metadata("ollama_url", self.config.ollama_url)
        self.store.set_metadata("generation_model", self.config.generation_model)
        self.store.set_metadata("embedding_model", self.config.embedding_model)
        self.store.connection.commit()
        self.ai = ai or LocalAI(
            self.config.ollama_url,
            self.config.generation_model,
            self.config.embedding_model,
        )
        self.vector = (
            vector_index
            if vector_index is not None
            else (VectorIndex(self.data_dir / "chroma") if use_chroma else None)
        )
        self.show_progress = show_progress
        self.progress_callback = progress_callback
        self.file_progress_callback = file_progress_callback
        self.warning_callback = warning_callback
        self.log_callback = log_callback
        self.issue_log = {
            "pdf_parser_warnings": [],
            "ocr_required": [],
            "ocr_failures": [],
            "vision_failures": [],
            "extraction_failures": [],
        }
        self.ocr_failure_count = 0

    def close(self) -> None:
        if self.vector and hasattr(self.vector, "close"):
            self.vector.close()
        self.store.close()

    def map(self, export_vault_notes: bool = True) -> Dict:
        os.environ.setdefault("OLLAMA_NO_CLOUD", "1")
        installed = self.ai.models()
        if self.config.generation_model not in installed:
            raise RuntimeError(f"Missing local Ollama model: {self.config.generation_model}")
        if self.config.embedding_model not in installed:
            raise RuntimeError(f"Missing local Ollama model: {self.config.embedding_model}")
        self._validate_vision_setup()
        total_stages = 8 if export_vault_notes else 7
        stages = tqdm(
            total=total_stages,
            desc="🗺️  Mapping repository",
            unit="stage",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        completed_stages = 0
        try:
            self._stage_start(completed_stages, "Discovery", total_stages)
            self._apply_version_invalidation()
            self._discover()
            completed_stages = self._stage_done(
                stages, completed_stages, "Discovery", total_stages
            )
            self._stage_start(completed_stages, "Extraction and parsing", total_stages)
            self._extract_parse()
            completed_stages = self._stage_done(
                stages, completed_stages, "Extraction and parsing", total_stages
            )
            self._stage_start(completed_stages, "Summaries and embeddings", total_stages)
            self._summarize_embed()
            completed_stages = self._stage_done(
                stages, completed_stages, "Summaries and embeddings", total_stages
            )
            self._stage_start(completed_stages, "Relationships", total_stages)
            self._relationships()
            completed_stages = self._stage_done(
                stages, completed_stages, "Relationships", total_stages
            )
            self._stage_start(completed_stages, "Graph export", total_stages)
            self._export_graph()
            completed_stages = self._stage_done(
                stages, completed_stages, "Graph export", total_stages
            )
            if export_vault_notes:
                self._stage_start(completed_stages, "Obsidian vault export", total_stages)
                self.export_obsidian()
                completed_stages = self._stage_done(
                    stages, completed_stages, "Obsidian vault export", total_stages
                )
            self._stage_start(completed_stages, "Search index", total_stages)
            self.store.rebuild_fts()
            completed_stages = self._stage_done(
                stages, completed_stages, "Search index", total_stages
            )
            self._stage_start(completed_stages, "Manifest", total_stages)
            self._manifest()
            completed_stages = self._stage_done(
                stages, completed_stages, "Manifest", total_stages
            )
        finally:
            self._log_issue_summary()
            stages.close()
        return self.store.status()

    def _validate_vision_setup(self) -> None:
        if not self.config.enable_vision:
            return
        if importlib.util.find_spec("ultralytics") is None:
            raise RuntimeError(
                "YOLO vision is enabled, but the local Python package `ultralytics` "
                "is not installed in this environment. Install it with "
                "`pip install -r requirements-vision.txt`, then rerun indexing."
            )
        model_path = _resolve_local_yolo_model(self.config.vision_model)
        if model_path is None:
            raise RuntimeError(
                "YOLO vision is enabled, but no local `.pt` weights file was found. "
                "Put `yolov8n.pt` or another YOLO `.pt` file in the repo `models` "
                "folder, then refresh the app or rerun indexing."
            )

    def _stage_start(self, completed: int, label: str, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(f"Running: {label}", completed, total)

    def _stage_done(self, stages, completed: int, label: str, total: int) -> int:
        stages.update()
        completed += 1
        if self.progress_callback:
            self.progress_callback(f"Finished: {label}", completed, total)
        return completed

    def _progress(self, items, description: str):
        items = list(items)
        progress = tqdm(
            items,
            desc=description,
            unit="file",
            disable=not self.show_progress,
            leave=False,
            dynamic_ncols=True,
        )
        total = len(items)
        for index, item in enumerate(progress, 1):
            if self.file_progress_callback:
                path = item.get("path", "") if isinstance(item, dict) else item["path"]
                self.file_progress_callback(description, path, index, total)
            yield item

    def _warn(self, message: str) -> None:
        if self.warning_callback:
            self.warning_callback(message)
        elif self.show_progress:
            tqdm.write(f"⚠️  {message}", file=sys.stderr)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
        elif self.show_progress:
            tqdm.write(message, file=sys.stderr)

    def _record_issue(self, category: str, path: str, detail: str) -> None:
        self.issue_log.setdefault(category, []).append(
            {"path": path, "detail": detail}
        )

    def _log_issue_summary(self) -> None:
        if not any(self.issue_log.values()):
            return
        self._log("")
        self._log("========== mapMyVault indexing issue summary ==========")
        for category, label in (
            ("pdf_parser_warnings", "PDF parser warnings"),
            ("ocr_required", "OCR needed but not run"),
            ("ocr_failures", "OCR failed/unavailable/empty"),
            ("vision_failures", "YOLO vision failed/unavailable/empty"),
            ("extraction_failures", "Extraction/OCR stage failures"),
        ):
            items = self.issue_log.get(category, [])
            if not items:
                continue
            self._log(f"{label}: {len(items)}")
            for item in items[:20]:
                self._log(f"  - file=`{item['path']}` | {item['detail']}")
            if len(items) > 20:
                self._log(f"  ... {len(items) - 20} more")
        self._log("======================================================")

    def _raise_ocr_unavailable(self, path: str, detail: str) -> None:
        raise OCRUnavailableError(
            "OCR is unavailable, so indexing stopped before processing more files. "
            f"First failing file: `{path}`. {detail} Check local OCR setup, "
            "especially Tesseract, Poppler/PyMuPDF, and OCR language settings. "
            "Run `tesseract --version`, `pdftoppm -v`, and "
            "`python -c \"import fitz\"` in the same terminal."
        )

    @contextmanager
    def _file_log_context(self, path: str):
        class FileContextHandler(logging.Handler):
            def __init__(handler_self, emit_warning, record_issue):
                super().__init__(level=logging.WARNING)
                handler_self.emit_warning = emit_warning
                handler_self.record_issue = record_issue

            def emit(handler_self, record):
                text = handler_self.format(record)
                handler_self.record_issue("pdf_parser_warnings", path, text)
                handler_self.emit_warning(f"PDF parser warning for `{path}`: {text}")

        logger = logging.getLogger("pypdf")
        previous_level = logger.level
        previous_propagate = logger.propagate
        handler = FileContextHandler(self._warn, self._record_issue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate

    def _warn_ocr_status(self, path: str, metadata: Dict) -> None:
        statuses = []
        for key in ("extraction_status", "ocr_status"):
            status = metadata.get(key, "")
            if status.startswith("ocr") and status not in statuses:
                statuses.append(status)
        for status in statuses:
            if status == "ocr_required":
                self._record_issue("ocr_required", path, "OCR is disabled for a file that appears to need OCR.")
                self._warn(
                    f"OCR not run for scanned PDF/image `{path}` because OCR is disabled. "
                    "Enable OCR to extract text."
                )
            elif status in {"ocr_failed", "ocr_unavailable", "ocr_empty"}:
                detail = metadata.get("ocr_error") or "No text returned by OCR."
                self._record_issue("ocr_failures", path, f"{status}. {detail}")
                self.ocr_failure_count += 1
                self._warn(f"OCR issue for `{path}`: {status}. {detail}")
                if status == "ocr_unavailable":
                    self._raise_ocr_unavailable(path, detail)

    def _warn_vision_status(self, path: str, metadata: Dict) -> None:
        status = metadata.get("vision_status", "")
        if status in {"vision_failed", "vision_unavailable"}:
            detail = metadata.get("vision_error") or "No objects returned by YOLO."
            self._record_issue("vision_failures", path, f"{status}. {detail}")
            self._warn(f"YOLO vision issue for `{path}`: {status}. {detail}")

    def _warn_extraction_status(self, path: str, metadata: Dict) -> None:
        self._warn_ocr_status(path, metadata)
        self._warn_vision_status(path, metadata)

    def _log_extraction_status(self, path: str, metadata: Dict) -> None:
        status = metadata.get("extraction_status", "unknown")
        characters = metadata.get("extracted_characters", 0)
        if status == "ocr_extracted":
            pages = metadata.get("ocr_pages", "?")
            renderer = metadata.get("ocr_pdf_renderer") or metadata.get("ocr_engine", "ocr")
            self._log(
                f"✅ OCR successful | file=`{path}` | chars={characters} "
                f"| pages={pages} | renderer={renderer}"
            )
        elif status == "extracted":
            self._log(f"✅ text extracted | file=`{path}` | chars={characters}")
        elif status in {"vision_extracted", "image_extracted"}:
            labels = metadata.get("vision_label_counts") or {}
            label_text = ", ".join(
                f"{label} x{count}" for label, count in sorted(labels.items())
            ) or "no labels"
            self._log(
                f"âœ… YOLO vision indexed | file=`{path}` | chars={characters} "
                f"| labels={label_text}"
            )
        elif status == "vision_empty":
            self._log(
                f"â„¹ï¸  YOLO ran with no matching object classes | "
                f"file=`{path}` | chars={characters}"
            )
        elif status == "ocr_required":
            self._log(f"⚠️  OCR needed but disabled | file=`{path}` | chars=0")
        elif status in {"ocr_failed", "ocr_unavailable", "ocr_empty"}:
            detail = metadata.get("ocr_error") or "No text returned by OCR."
            self._log(f"⚠️  OCR not successful | file=`{path}` | status={status} | {detail}")
        else:
            self._log(f"ℹ️  extraction status | file=`{path}` | status={status} | chars={characters}")

    def _apply_version_invalidation(self) -> None:
        rules = {
            "summary_version": (
                self.config.summary_prompt_version,
                ["summary", "embedding", "relationships", "export"],
            ),
            "embedding_version": (
                self.config.embedding_model,
                ["embedding", "relationships", "export"],
            ),
            "ocr_version": (
                self.config.versions()["ocr"],
                ["extraction", "parsing", "summary", "embedding", "relationships", "export"],
            ),
            "vision_version": (
                self.config.versions()["vision"],
                ["extraction", "parsing", "summary", "embedding", "relationships", "export"],
            ),
            "relationship_version": (
                self.config.relationship_prompt_version,
                ["relationships", "export"],
            ),
            "export_version": (self.config.export_template_version, ["export"]),
        }
        for key, (current, stages) in rules.items():
            previous = self.store.get_metadata(key)
            if previous is not None and previous != current:
                for row in self.store.files():
                    self.store.invalidate(row["id"], stages)
            self.store.set_metadata(key, current)
        self.store.connection.commit()

    def _discover(self) -> None:
        records = scan_repository(
            self.repository,
            self.output,
            self.config.excludes,
            self.config.max_file_bytes,
        )
        ids = []
        for record in self._progress(records, "📂 Discovering files"):
            ids.append(record["id"])
            previous = self.store.get_file(record["id"])
            changed = previous and previous["content_hash"] != record["content_hash"]
            record["updated_at"] = utc_now()
            self.store.upsert_file(record)
            if changed:
                self.store.invalidate(
                    record["id"],
                    ["extraction", "parsing", "summary", "embedding", "relationships", "export"],
                )
            self.store.stage_complete(record["id"], "discovery", PIPELINE_VERSION)
        removed_ids = self.store.mark_missing_deleted(ids)
        if self.vector and removed_ids:
            self.vector.delete(removed_ids)
        self.store.connection.commit()

    def _run_stage(self, row, stage: str, version: str, callback) -> None:
        if self.store.stage_valid(row["id"], stage, version):
            return
        try:
            callback()
            self.store.stage_complete(row["id"], stage, version, row["content_hash"] or "")
        except OCRUnavailableError:
            raise
        except Exception as exc:
            self.store.stage_failed(row["id"], stage, version, str(exc))
            self._log(f"❌ stage failed | file=`{row['path']}` | stage={stage} | {exc}")
            if stage == "extraction" and row["extension"] in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
                self._record_issue("extraction_failures", row["path"], str(exc))
                self._warn(
                    f"Extraction/OCR could not run for `{row['path']}`: {exc}"
                )

    def _extract_parse(self) -> None:
        for row in self._progress(self.store.files("file"), "📖 Extracting and parsing"):
            source = self.repository / row["path"]
            retry_failed_ocr = (
                self.config.enable_ocr
                and _metadata_status(row) in RETRYABLE_EXTRACTION_STATUSES
            )
            retry_failed_vision = (
                self.config.enable_vision
                and _metadata_value(row, "vision_status") in RETRYABLE_VISION_STATUSES
            )
            if retry_failed_ocr or retry_failed_vision:
                self.store.invalidate(
                    row["id"],
                    ["extraction", "parsing", "summary", "embedding", "relationships", "export"],
                )
            if not self.store.stage_valid(row["id"], "extraction", PIPELINE_VERSION):
                self.store.invalidate(
                    row["id"], ["parsing", "summary", "embedding", "relationships", "export"]
                )

            def extract(row=row, source=source):
                self._log(f"🔎 extracting | file=`{row['path']}`")
                with self._file_log_context(row["path"]):
                    text, document_metadata = extract_content(
                        source,
                        self.config.max_file_bytes,
                        self.config.max_content_chars,
                        self.config.enable_ocr,
                        self.config.ocr_engine,
                        self.config.ocr_language,
                        self.config.ocr_dpi,
                        self.config.ocr_max_pages,
                        self.config.enable_vision,
                        self.config.vision_engine,
                        self.config.vision_model,
                        self.config.vision_confidence,
                        self.config.vision_max_detections,
                    )
                self.store.update_artifact(row["id"], "extracted_text", text)
                self.store.update_artifact(
                    row["id"],
                    "document_metadata_json",
                    json.dumps(document_metadata, sort_keys=True),
                )
                self._warn_extraction_status(row["path"], document_metadata)
                self._log_extraction_status(row["path"], document_metadata)

            self._run_stage(row, "extraction", PIPELINE_VERSION, extract)
            current = self.store.get_file(row["id"])
            if not self.store.stage_valid(current["id"], "parsing", PIPELINE_VERSION):
                self.store.invalidate(
                    current["id"], ["summary", "embedding", "relationships", "export"]
                )

            def parse(current=current):
                parsed = parse_content(current["path"], current["extracted_text"] or "")
                self.store.update_artifact(
                    current["id"], "parse_data", json.dumps(parsed, sort_keys=True)
                )
                self._log(f"✅ parsed | file=`{current['path']}`")

            self._run_stage(current, "parsing", PIPELINE_VERSION, parse)

    def _summarize_embed(self) -> None:
        for row in self._progress(self.store.files("file"), "🧠 Summarizing and embedding"):
            if not row["extracted_text"]:
                status = _metadata_status(row) or "no_extracted_text"
                self._log(
                    f"⏭️  indexing skipped | file=`{row['path']}` | reason={status}"
                )
                continue
            if not self.store.stage_valid(
                row["id"], "summary", self.config.summary_prompt_version
            ):
                self.store.invalidate(
                    row["id"], ["embedding", "relationships", "export"]
                )

            def summarize(row=row):
                summary = self.ai.summarize(
                    row["path"],
                    row["extracted_text"],
                    row["parse_data"] or "{}",
                    row["document_metadata_json"] or "{}",
                )
                payload = summary.model_dump_json()
                self.store.update_artifact(row["id"], "summary_json", payload)
                atomic_write(self.summary_dir / f"{row['id']}.json", payload)

            self._run_stage(
                row, "summary", self.config.summary_prompt_version, summarize
            )
            current = self.store.get_file(row["id"])
            if not current["summary_json"]:
                self._log(f"⚠️  summary missing | file=`{row['path']}`")
                continue
            if not self.store.stage_valid(
                current["id"], "embedding", self.config.embedding_model
            ):
                self.store.invalidate(current["id"], ["relationships", "export"])

            def embed(current=current):
                vector = self.ai.embed([current["summary_json"]])[0]
                payload = json.dumps(vector)
                self.store.update_artifact(current["id"], "embedding_json", payload)

            self._run_stage(
                current, "embedding", self.config.embedding_model, embed
            )
            current = self.store.get_file(row["id"])
            if current["summary_json"] and current["embedding_json"]:
                self._log(
                    f"✅ indexed successfully | file=`{row['path']}` "
                    f"| summary=ready | embedding=ready"
                )
            if self.vector and current["embedding_json"]:
                self.vector.upsert(
                    current["id"],
                    current["summary_json"],
                    json.loads(current["embedding_json"]),
                    current["path"],
                )

    def _relationships(self) -> None:
        by_path = {row["path"]: row for row in self.store.files()}
        for row in self.store.files():
            if row["parent"] and row["parent"] in by_path:
                parent = by_path[row["parent"]]
                self.store.upsert_relationship(
                    parent["id"],
                    row["id"],
                    "parent_child",
                    1.0,
                    f"{parent['path']} contains {row['path']}",
                    [f"Filesystem parent: {parent['path']}"],
                    True,
                )
        file_dicts = [dict(row) for row in self.store.files("file")]
        for edge in deterministic_edges(file_dicts):
            self.store.upsert_relationship(
                edge["source"],
                edge["target"],
                edge["type"],
                1.0,
                edge["evidence"][0],
                edge["evidence"],
                True,
            )
        self.store.connection.commit()

        if self.vector:
            for row in self._progress(
                self.store.files("file"), "🔗 Evaluating relationships"
            ):
                if not row["embedding_json"] or not row["summary_json"]:
                    continue
                if self.store.stage_valid(
                    row["id"], "relationships", self.config.relationship_prompt_version
                ):
                    continue
                try:
                    candidates = self.vector.candidates(
                        json.loads(row["embedding_json"]),
                        self.config.semantic_candidates + 1,
                    )
                    for candidate in candidates:
                        if candidate["id"] == row["id"]:
                            continue
                        target = self.store.get_file(candidate["id"])
                        if not target or not target["summary_json"]:
                            continue
                        decision = self.ai.judge_relationship(
                            row["path"],
                            row["summary_json"],
                            target["path"],
                            target["summary_json"],
                        )
                        if (
                            decision.related
                            and decision.confidence >= self.config.relationship_threshold
                            and decision.evidence
                        ):
                            self.store.upsert_relationship(
                                row["id"],
                                target["id"],
                                decision.relationship_type,
                                decision.confidence,
                                decision.explanation,
                                decision.evidence,
                                False,
                            )
                    self.store.connection.commit()
                    self.store.stage_complete(
                        row["id"],
                        "relationships",
                        self.config.relationship_prompt_version,
                    )
                except Exception as exc:
                    self.store.stage_failed(
                        row["id"],
                        "relationships",
                        self.config.relationship_prompt_version,
                        str(exc),
                    )

    def _export_graph(self) -> None:
        import networkx as nx

        files = [dict(row) for row in self.store.files()]
        relationship_rows = [
            dict(row)
            for row in self.store.connection.execute(
                "SELECT * FROM relationships ORDER BY source_id,target_id,type"
            )
        ]
        network = nx.Graph()
        network.add_nodes_from(row["id"] for row in files)
        network.add_edges_from(
            (row["source_id"], row["target_id"]) for row in relationship_rows
        )
        graph = {
            "nodes": [
                {
                    "id": row["id"],
                    "path": row["path"],
                    "kind": row["kind"],
                    "degree": network.degree(row["id"]),
                }
                for row in files
            ],
            "edges": relationship_rows,
            "analysis": {
                "connected_components": nx.number_connected_components(network)
                if network.number_of_nodes()
                else 0
            },
        }
        atomic_write(self.data_dir / "graph.json", json.dumps(graph, indent=2))

    def export_obsidian(self, vault_dir: Optional[Path] = None) -> Dict:
        vault_dir = (vault_dir or self.vault_dir).resolve()
        files = [dict(row) for row in self.store.files()]
        relationships = {
            row["id"]: [dict(item) for item in self.store.relationships_for(row["id"])]
            for row in self.store.files()
        }
        export_vault(vault_dir, files, relationships)
        for row in self.store.files():
            self.store.stage_complete(
                row["id"], "export", self.config.export_template_version
            )
        self.store.connection.commit()
        return {
            "vault": str(vault_dir),
            "files": len([row for row in files if row["kind"] == "file"]),
            "folders": len([row for row in files if row["kind"] == "folder"]),
        }

    def _manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "created_at": utc_now(),
            "repository": str(self.repository),
            "output": str(self.output),
            "local_only": True,
            "endpoints": {"ollama": self.config.ollama_url},
            "models": {
                "generation": self.config.generation_model,
                "embedding": self.config.embedding_model,
            },
            "versions": self.config.versions(),
            "config": self.config.to_dict(),
            "tools": [
                "sqlite",
                "chroma",
                "ollama",
                "pathspec",
                "python-ast",
                "pypdf",
                "python-docx",
                "openpyxl",
                "python-pptx",
                "pytesseract" if self.config.enable_ocr else None,
                "pdf2image" if self.config.enable_ocr else None,
                "tesseract" if self.config.enable_ocr else None,
                "ultralytics-yolo" if self.config.enable_vision else None,
            ],
            "status": self.store.status(),
        }
        manifest["tools"] = [tool for tool in manifest["tools"] if tool]
        atomic_write(self.data_dir / "manifest.json", json.dumps(manifest, indent=2))
