"""Streamlit UI for local-first mapMyVault workflows."""

from contextlib import contextmanager
from pathlib import Path
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import streamlit as st

from src.answers import deterministic_answer
from src.config import MapperConfig
from src.exporter import export_obsidian_from_index
from src.local_ai import LocalAI
from src.mapper import RepositoryMapper
from src.query import VaultIndex, _is_count_question
from src.storage import IndexStore
from src.vector_index import VectorIndex


st.set_page_config(page_title="mapMyVault Studio", layout="wide")

MENU_ITEMS = ["Chat", "Knowledge", "Graph View"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODELS_DIR = PROJECT_ROOT / "models"


def _session_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _pick_folder(current: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=current or None)
        root.destroy()
        return selected or current
    except Exception as exc:
        st.warning(f"Folder picker unavailable: {exc}")
        return current


def _browse_into(key: str) -> None:
    st.session_state[key] = _pick_folder(st.session_state.get(key, ""))


def _sync_state(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state.get(source_key, "")


def _prepare_synced_input(widget_key: str, source_key: str) -> None:
    source_value = st.session_state.get(source_key, "")
    if st.session_state.get(widget_key, "") != source_value:
        st.session_state[widget_key] = source_value


def _browse_synced_folder(widget_key: str, source_key: str) -> None:
    selected = _pick_folder(st.session_state.get(widget_key, ""))
    st.session_state[widget_key] = selected
    st.session_state[source_key] = selected


def _request_indexing() -> None:
    st.session_state.indexing = True
    st.session_state.index_requested = True


def _request_graph_indexing() -> None:
    st.session_state.indexing = True
    st.session_state.graph_index_requested = True


def _available_yolo_models() -> list[str]:
    YOLO_MODELS_DIR.mkdir(exist_ok=True)
    return sorted(path.name for path in YOLO_MODELS_DIR.glob("*.pt") if path.is_file())


def _ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


@st.cache_data(ttl=30)
def _available_ollama_models() -> list[str]:
    try:
        return LocalAI("http://127.0.0.1:11434", "", "").models()
    except Exception:
        return []


EMBEDDING_MODEL_MARKERS = (
    "embed",
    "embedding",
    "nomic",
    "mxbai",
    "bge",
    "minilm",
    "e5",
    "jina",
    "snowflake",
    "arctic",
    "gte",
)


VISION_MODEL_MARKERS = (
    "vision",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "qwen2.5vl",
    "qwen-vl",
    "gemma3",
    "pixtral",
)


def _available_chat_models() -> list[str]:
    return [
        model
        for model in _available_ollama_models()
        if not any(marker in model.lower() for marker in EMBEDDING_MODEL_MARKERS)
    ]


def _available_embedding_models() -> list[str]:
    return [
        model
        for model in _available_ollama_models()
        if any(marker in model.lower() for marker in EMBEDDING_MODEL_MARKERS)
    ]


def _is_vision_model(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in VISION_MODEL_MARKERS)


def _index_exists(output: Path) -> bool:
    return (output / "data" / "index.sqlite").exists()


def _status(output: Path):
    index = VaultIndex(output)
    try:
        return index.status()
    finally:
        index.close()


def _ask_with_history(
    output: Path,
    question: str,
    history: list[dict],
    generation_model: str,
    images: list[str] | None = None,
    image_names: list[str] | None = None,
) -> dict:
    index = VaultIndex(output)
    try:
        evidence = index.ask_mapmyvault(question, limit=30)
    finally:
        index.close()
    if evidence.get("found_count", 0) == 0 and not images:
        evidence["model_used"] = "none"
        return evidence
    if _is_count_question(question):
        evidence["answer"] = deterministic_answer(evidence)
        evidence["deterministic"] = True
        evidence["model_used"] = "local deterministic count"
        return evidence
    ai = LocalAI("http://127.0.0.1:11434", generation_model, "nomic-embed-text:latest")
    try:
        evidence["answer"] = ai.answer_from_evidence(
            question,
            history,
            evidence,
            images=images,
            image_names=image_names,
        )
        evidence["model_used"] = generation_model
    except Exception as exc:
        evidence["answer"] = (
            f"{evidence.get('answer', 'Local evidence was found.')} "
            f"(The local model could not generate a fuller answer: {exc})"
        )
        evidence["model_used"] = f"{generation_model} failed"
    return evidence


def _tree_lines(output: Path):
    store = IndexStore(output / "data" / "index.sqlite")
    try:
        rows = sorted(store.files(), key=lambda row: row["path"].lower())
        lines = []
        for row in rows:
            depth = row["path"].count("/")
            icon = "[D]" if row["kind"] == "folder" else "[F]"
            lines.append(f"{'  ' * depth}{icon} {row['path']}")
        return lines
    finally:
        store.close()


def _start_mcp(output: Path, port: int):
    existing = st.session_state.get("mcp_process")
    if existing and existing.poll() is None:
        return existing
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.cli",
            "serve",
            str(output),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state["mcp_process"] = process
    return process


def _stop_mcp():
    process = st.session_state.get("mcp_process")
    if process and process.poll() is None:
        process.terminate()
    st.session_state["mcp_process"] = None


@contextmanager
def _index_lock(output: Path):
    lock_path = output / "data" / ".indexing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > 12 * 60 * 60:
        lock_path.unlink()
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            "This index is already being generated. Wait for the current run to finish."
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()


def _format_chat_answer(result: dict, generation_model: str) -> str:
    answer = result.get("answer", "No answer returned.")
    paths = result.get("paths", [])
    if paths:
        answer += "\n\nSources:\n" + "\n".join(f"- `{path}`" for path in paths[:20])
    if result.get("deterministic"):
        answer += (
            "\n\n_Note: this is an exact local database count, so the chat model "
            "was not used._"
        )
    elif result.get("model_used") and result["model_used"] != "none":
        answer += f"\n\n_Model used: `{generation_model}`_"
    return answer


def _uploaded_chat_images() -> tuple[list[str], list[str]]:
    images = []
    names = []
    for item in st.session_state.get("chat_uploads", []):
        images.append(base64.b64encode(item["data"]).decode("ascii"))
        names.append(item["name"])
    return images, names


def _add_chat_uploads(files) -> None:
    if not files:
        return
    existing = {
        (item["name"], item["size"])
        for item in st.session_state.get("chat_uploads", [])
    }
    uploads = list(st.session_state.get("chat_uploads", []))
    for file in files:
        data = file.getvalue()
        key = (file.name, len(data))
        if key in existing:
            continue
        uploads.append(
            {
                "name": file.name,
                "type": file.type,
                "size": len(data),
                "data": data,
            }
        )
    st.session_state.chat_uploads = uploads


def _remove_chat_upload(index: int) -> None:
    uploads = list(st.session_state.get("chat_uploads", []))
    if 0 <= index < len(uploads):
        uploads.pop(index)
    st.session_state.chat_uploads = uploads


def _chat_history_for_model() -> list[dict]:
    return [
        {
            "role": item.get("role", ""),
            "content": item.get("content", ""),
        }
        for item in st.session_state.get("chat_history", [])
    ]


def _generate_chat_answer(
    output: Path | None,
    question: str,
    history: list[dict],
    generation_model: str,
    images: list[str] | None = None,
    image_names: list[str] | None = None,
) -> tuple[str, dict | None]:
    if not output or not _index_exists(output):
        return "Generate or load a local mapMyVault index first.", None
    with st.status("Thinking with the local knowledge base...", expanded=False):
        st.write("Searching the local index...")
        result = _ask_with_history(
            output,
            question,
            history,
            generation_model,
            images=images,
            image_names=image_names,
        )
        if result.get("deterministic"):
            st.write("Using exact local database count.")
        elif result.get("found_count", 0) == 0:
            st.write("No relevant local evidence was found.")
        else:
            st.write(f"Generating answer with `{generation_model}`...")
        st.write("Preparing the grounded answer...")
    return _format_chat_answer(result, generation_model), result


def _manifest(output: Path) -> dict:
    manifest_path = output / "data" / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _config_from_manifest(output: Path) -> MapperConfig:
    manifest = _manifest(output)
    metadata_ollama_url = _index_metadata(output, "ollama_url")
    metadata_generation_model = _index_metadata(output, "generation_model")
    locked_embedding_model = _locked_embedding_model(output)
    if not manifest:
        return MapperConfig(
            ollama_url=metadata_ollama_url or "http://127.0.0.1:11434",
            generation_model=metadata_generation_model or "llama3.1:8b",
            embedding_model=locked_embedding_model or _selected_embedding_model(),
        )
    if "config" in manifest:
        config = MapperConfig(**manifest.get("config", {}))
        if locked_embedding_model:
            config.embedding_model = locked_embedding_model
        if metadata_generation_model:
            config.generation_model = metadata_generation_model
        if metadata_ollama_url:
            config.ollama_url = metadata_ollama_url
        return config
    return MapperConfig(
        ollama_url=metadata_ollama_url
        or manifest.get("endpoints", {}).get("ollama", "http://127.0.0.1:11434"),
        generation_model=metadata_generation_model
        or manifest.get("models", {}).get("generation", "llama3.1:8b"),
        embedding_model=locked_embedding_model or _selected_embedding_model(),
    )


def _locked_embedding_model(output: Path | None) -> str | None:
    if not output or not _index_exists(output):
        return None
    metadata_embedding_model = _index_metadata(output, "embedding_model")
    if metadata_embedding_model:
        return metadata_embedding_model
    manifest = _manifest(output)
    if not manifest:
        return None
    config = manifest.get("config") or {}
    return config.get("embedding_model") or manifest.get("models", {}).get("embedding")


def _selected_embedding_model() -> str:
    return st.session_state.get("embedding_model") or "nomic-embed-text:latest"


def _repository_from_index(output: Path) -> Path | None:
    manifest = _manifest(output)
    repository = (
        manifest.get("repository")
        or manifest.get("source")
        or _index_metadata(output, "repository")
        or _index_metadata(output, "source")
    )
    if not repository:
        return None
    return Path(repository).expanduser().resolve()


def _index_metadata(output: Path, key: str) -> str | None:
    database = output / "data" / "index.sqlite"
    if not database.exists():
        return None
    store = IndexStore(database)
    try:
        return store.get_metadata(key)
    finally:
        store.close()


def _index_excluded_paths(output: Path) -> list[str]:
    raw = _index_metadata(output, "graph_excluded_paths")
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return sorted({str(value).strip("/") for value in values if str(value).strip("/")})


def _set_index_excluded_paths(output: Path, paths: list[str]) -> None:
    store = IndexStore(output / "data" / "index.sqlite")
    try:
        normalized = sorted({path.strip("/") for path in paths if path.strip("/")})
        store.set_metadata("graph_excluded_paths", json.dumps(normalized))
        store.connection.commit()
    finally:
        store.close()


def _is_index_excluded(path: str, excluded_paths: list[str]) -> bool:
    normalized = path.strip("/")
    return any(
        normalized == excluded or normalized.startswith(f"{excluded}/")
        for excluded in excluded_paths
    )


def _exclude_patterns(paths: list[str]) -> list[str]:
    patterns = []
    for path in paths:
        normalized = path.strip("/")
        if normalized:
            patterns.extend([normalized, f"{normalized}/", f"{normalized}/**"])
    return patterns


def _source_tree_items(source: Path, max_items: int = 300) -> list[dict]:
    if not source or not source.is_dir():
        return []
    items = []
    for item in sorted(source.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower())):
        if len(items) >= max_items:
            break
        if item.name.startswith(".git"):
            continue
        items.append(
            {
                "path": item.relative_to(source).as_posix(),
                "name": item.name,
                "kind": "folder" if item.is_dir() else "file",
            }
        )
    return items


def _include_key(path: str) -> str:
    return f"knowledge_include_{path}"


def _render_source_selection(source: Path | None) -> list[str]:
    if not source or not source.is_dir():
        return []
    items = _source_tree_items(source)
    if not items:
        return []
    st.subheader("Choose What To Index")
    st.caption(
        "Untick folders/files you do not want in the local index. Excluded items "
        "are skipped before extraction, OCR, embeddings, and summaries."
    )
    control_cols = st.columns([0.18, 0.18, 0.64])
    if control_cols[0].button("Select all", key="knowledge_select_all"):
        for item in items:
            st.session_state[_include_key(item["path"])] = True
        st.rerun()
    if control_cols[1].button("Select none", key="knowledge_select_none"):
        for item in items:
            st.session_state[_include_key(item["path"])] = False
        st.rerun()

    excluded = []
    header_cols = st.columns([0.10, 0.08, 0.60, 0.22])
    header_cols[0].caption("Index")
    header_cols[1].caption("")
    header_cols[2].caption("Top-level item")
    header_cols[3].caption("Type")
    for item in items:
        st.session_state.setdefault(_include_key(item["path"]), True)
        row_cols = st.columns([0.10, 0.08, 0.60, 0.22])
        with row_cols[0]:
            include = st.checkbox(
                "Index item",
                key=_include_key(item["path"]),
                label_visibility="collapsed",
            )
        with row_cols[1]:
            st.write("📁" if item["kind"] == "folder" else "📄")
        row_cols[2].write(item["name"])
        row_cols[3].caption(item["kind"])
        if not include:
            excluded.append(item["path"])
    if excluded:
        st.warning(f"{len(excluded)} top-level item(s) will be skipped.")
    else:
        st.caption("Everything shown here will be indexed.")
    return excluded


def _children(output: Path, folder: str) -> list[dict]:
    index = VaultIndex(output)
    try:
        return index.get_folder_children(folder)
    finally:
        index.close()


def _to_posix_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_target_dir(root: Path, relative_folder: str) -> Path:
    target = (root / relative_folder).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Selected folder is outside the indexed source root.")
    return target


def _valid_child_name(name: str) -> str:
    clean = name.strip()
    if not clean:
        raise ValueError("Enter a folder name first.")
    if clean in {".", ".."}:
        raise ValueError("Folder name cannot be . or ..")
    if Path(clean).name != clean or "/" in clean or "\\" in clean:
        raise ValueError("Use a single folder name, not a path.")
    return clean


def _parent_folder(relative_folder: str) -> str:
    if not relative_folder:
        return ""
    parent = Path(relative_folder).parent.as_posix()
    return "" if parent == "." else parent


def _child_path(parent: str, child_name: str) -> str:
    return f"{parent}/{child_name}" if parent else child_name


def _combined_children(output: Path, repository: Path, folder: str) -> list[dict]:
    """Merge indexed children with live source children so new folders are browsable."""
    excluded_paths = _index_excluded_paths(output)
    indexed = {}
    for item in _children(output, folder):
        if _is_index_excluded(item["path"], excluded_paths):
            continue
        source_item = _safe_target_dir(repository, item["path"])
        indexed[item["path"]] = {
            **item,
            "indexed": True,
            "exists": source_item.exists(),
        }
    target_dir = _safe_target_dir(repository, folder)
    if target_dir.exists() and target_dir.is_dir():
        for child in sorted(target_dir.iterdir(), key=lambda item: item.name.lower()):
            relative = _to_posix_relative(repository, child)
            if _is_index_excluded(relative, excluded_paths):
                continue
            if relative in indexed:
                continue
            if child.is_dir():
                indexed[relative] = {
                    "path": relative,
                    "kind": "folder",
                    "indexed": False,
                    "exists": True,
                }
            elif child.is_file():
                indexed[relative] = {
                    "path": relative,
                    "kind": "file",
                    "indexed": False,
                    "exists": True,
                }
    return sorted(
        indexed.values(),
        key=lambda item: (item["kind"] != "folder", item["path"].lower()),
    )


def _ensure_graph_folder_available(repository: Path, current: str) -> str:
    """Keep Graph View from getting stuck on a deleted or moved folder."""
    if not current:
        return ""
    target_dir = _safe_target_dir(repository, current)
    if target_dir.exists() and target_dir.is_dir():
        return current
    parent = _parent_folder(current)
    st.warning(
        f"`{current}` no longer exists in the source tree. Showing "
        f"`{parent or 'Root'}` instead."
    )
    _set_graph_folder(parent)
    return parent


def _run_graph_update(repository: Path, output: Path) -> bool:
    st.session_state.indexing = True
    with st.spinner("Updating only new, changed, moved, or deleted indexed work..."):
        config = _graph_update_config(output)
        if not _ensure_required_models(config):
            st.session_state.indexing = False
            return False
        _run_index(repository, output, config)
        return True


def _graph_update_config(output: Path) -> MapperConfig:
    config = _config_from_manifest(output)
    config.enable_ocr = True
    config.ocr_language = (
        st.session_state.get("ocr_language")
        or config.ocr_language
        or "eng"
    ).strip() or "eng"
    config.ocr_max_pages = int(
        st.session_state.get("ocr_max_pages")
        or config.ocr_max_pages
        or 10
    )
    config.enable_vision = bool(st.session_state.get("enable_vision", config.enable_vision))
    config.vision_model = (
        st.session_state.get("vision_model")
        or config.vision_model
        or "yolov8n.pt"
    )
    config.vision_confidence = float(
        st.session_state.get("vision_confidence")
        or config.vision_confidence
        or 0.25
    )
    config.vision_max_detections = int(
        st.session_state.get("vision_max_detections")
        or config.vision_max_detections
        or 50
    )
    config.excludes = list(dict.fromkeys(config.excludes + _exclude_patterns(_index_excluded_paths(output))))
    return config


def _required_models(config: MapperConfig) -> list[str]:
    return list(dict.fromkeys([config.generation_model, config.embedding_model]))


def _ensure_required_models(config: MapperConfig) -> bool:
    models = _available_ollama_models()
    missing = [model for model in _required_models(config) if model not in models]
    if not missing:
        return True
    st.error(
        "This index must be updated with the same local models it was created with. "
        "Install the missing model(s), then rerun the update."
    )
    for model in missing:
        st.code(f"ollama pull {model}", language="powershell")
    return False


def _remove_index_subtree(output: Path, relative_folder: str, persist_exclusion: bool = False) -> int:
    store = IndexStore(output / "data" / "index.sqlite")
    try:
        removed = store.mark_path_tree_deleted(relative_folder)
        store.rebuild_fts()
        chroma_path = output / "data" / "chroma"
        if removed and chroma_path.exists():
            VectorIndex(chroma_path).delete(removed)
    finally:
        store.close()
    if persist_exclusion:
        excluded = _index_excluded_paths(output)
        _set_index_excluded_paths(output, excluded + [relative_folder])
    return len(removed)


def _remove_index_subtrees(output: Path, relative_folders: list[str], persist_exclusion: bool = True) -> int:
    removed_total = 0
    for folder in relative_folders:
        removed_total += _remove_index_subtree(output, folder, persist_exclusion)
    return removed_total


def _unique_destination(folder: Path, name: str) -> Path:
    clean_name = Path(name).name
    target = folder / clean_name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = folder / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _run_index(source: Path, output: Path, config: MapperConfig) -> None:
    progress = st.progress(0)
    status = st.empty()
    current_file = st.empty()
    warning_box = st.empty()
    warnings = []

    def progress_callback(label: str, done: int, total: int):
        percent = done / total
        state, _, stage_name = label.partition(": ")
        stage_number = min(done + 1, total) if state == "Running" else done
        progress.progress(percent)
        status.info(
            f"Stage {stage_number}/{total}: {stage_name or label}\n\n"
            f"Status: {state if stage_name else 'Running'}\n\n"
            f"Overall progress: {percent:.0%} ({done}/{total} stages complete)"
        )

    def file_progress_callback(label: str, path: str, done: int, total: int):
        current_file.caption(
            f"Current file in this stage: {path or 'folder'} "
            f"({done}/{total} files) during {label}"
        )

    def warning_callback(message: str):
        warnings.append(message)
        warning_box.warning(
            "Indexing warning:\n\n"
            + "\n".join(f"- {item}" for item in warnings[-5:])
        )

    mapper = None
    try:
        with _index_lock(output):
            mapper = RepositoryMapper(
                source,
                output,
                config,
                show_progress=False,
                progress_callback=progress_callback,
                file_progress_callback=file_progress_callback,
                warning_callback=warning_callback,
            )
            result = mapper.map(export_vault_notes=False)
            progress.progress(1.0)
            status.success("100% - Local LLM index is ready.")
            st.json(result)
    finally:
        if mapper:
            mapper.close()
        st.session_state.indexing = False


def _selected_output() -> Path | None:
    output_text = st.session_state.get("output_path", "")
    return Path(output_text).expanduser().resolve() if output_text else None


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("mapMyVault")
        st.caption("Menu")
        for item in MENU_ITEMS:
            active = item == st.session_state.view
            if st.button(
                item,
                key=f"nav_{item}",
                type="primary" if active else "secondary",
                use_container_width=True,
                disabled=active,
            ):
                st.session_state.view = item
                st.rerun()
        st.divider()
        st.caption("Selected local index")
        output = _selected_output()
        if output and _index_exists(output):
            st.success("Local index loaded")
        elif output:
            st.warning("No index.sqlite found here yet")
            st.caption(str(output))
        else:
            st.info("Choose an output folder in Chat, Knowledge, or Graph View.")


def _render_output_picker(label: str, widget_key: str) -> str:
    _prepare_synced_input(widget_key, "output_path")
    output_col, output_button_col = st.columns([0.84, 0.16])
    with output_col:
        output_text = st.text_input(
            label,
            key=widget_key,
            placeholder=r"C:\path\to\mapmyvault-output",
            on_change=_sync_state,
            args=(widget_key, "output_path"),
        )
    with output_button_col:
        st.write("")
        st.button(
            "Browse",
            key=f"browse_{widget_key}",
            on_click=_browse_synced_folder,
            args=(widget_key, "output_path"),
        )
    return output_text


def _render_chat() -> None:
    output = _selected_output()

    st.title("Chat")
    st.caption("Answers come from the selected local mapMyVault index only.")
    model_col, _spacer = st.columns([0.34, 0.66])
    with model_col:
        generation_model = _render_chat_model_control()
    _render_output_picker("Index/output folder", "chat_output_path_input")
    output = _selected_output()
    if output:
        st.caption(f"Using index: `{output}`")
    else:
        st.info("Choose an index/output folder in the sidebar or build one from Knowledge.")
    st.info("Chat history is kept in memory for this app session and sent as context with each answer.")

    with st.expander("Upload local image(s) for a vision model", expanded=False):
        if _is_vision_model(generation_model):
            st.caption("Uploaded images are sent only to your selected local Ollama model for the next questions.")
        else:
            st.warning(
                "The selected model does not look like a vision model. You can upload "
                "images, but Ollama may reject them unless the model supports vision."
            )
        uploads = st.file_uploader(
            "Upload image file(s)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="chat_image_uploader",
        )
        if uploads:
            _add_chat_uploads(uploads)
        if st.session_state.chat_uploads:
            st.write("Attached image(s)")
            for index, item in enumerate(list(st.session_state.chat_uploads)):
                cols = st.columns([0.72, 0.18, 0.10])
                cols[0].caption(f"{item['name']} ({item['size']} bytes)")
                cols[1].caption(item.get("type") or "image")
                if cols[2].button("Remove", key=f"remove_chat_upload_{index}"):
                    _remove_chat_upload(index)
                    st.rerun()
        else:
            st.caption("No images attached.")

    if st.button("Clear chat", key="clear_chat_history"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask the local knowledge base")
    if not question:
        return

    image_payloads, image_names = _uploaded_chat_images()
    user_content = question
    if image_names:
        user_content += "\n\nAttached image(s): " + ", ".join(image_names)
    history = _chat_history_for_model()
    st.session_state.chat_history.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        st.markdown(user_content)

    with st.chat_message("assistant"):
        answer, result = _generate_chat_answer(
            output,
            question,
            history,
            generation_model,
            images=image_payloads,
            image_names=image_names,
        )
        st.markdown(answer)
        if result:
            with st.expander("Raw local evidence"):
                st.json(result)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _model_value(models: list[str], key: str, preferred: str) -> str:
    current = st.session_state.get(key, preferred)
    if models:
        if current not in models:
            current = preferred if preferred in models else models[0]
        st.session_state[key] = current
    return current


def _render_chat_model_control() -> str:
    models = _available_chat_models()
    if models:
        generation_value = _model_value(models, "generation_model", "llama3.1:8b")
        generation_model = st.selectbox(
            "Chat model",
            models,
            index=models.index(generation_value),
            key="chat_generation_model_select",
        )
        st.session_state.generation_model = generation_model
    else:
        st.warning("Could not find local chat LLMs in Ollama. Check Ollama is running.")
        generation_model = st.text_input(
            "Chat model",
            st.session_state.generation_model,
        )
        st.session_state.generation_model = generation_model
    return generation_model


def _render_embedding_model_control(locked_model: str | None = None) -> str:
    if locked_model:
        st.text_input("Ollama embedding model", value=locked_model, disabled=True)
        st.caption("Locked to the embedding model recorded in this existing index.")
        st.session_state.embedding_model = locked_model
        return locked_model
    models = _available_embedding_models()
    if models:
        embedding_value = _model_value(models, "embedding_model", "nomic-embed-text:latest")
        embedding_model = st.selectbox(
            "Ollama embedding model",
            models,
            index=models.index(embedding_value),
            key="embedding_model_select",
        )
        st.session_state.embedding_model = embedding_model
    else:
        st.warning(
            "No local embedding models were found in Ollama. Install one, then refresh."
        )
        st.code("ollama pull nomic-embed-text:latest", language="powershell")
        embedding_model = st.text_input(
            "Embedding model to install/use",
            st.session_state.embedding_model,
        )
        st.session_state.embedding_model = embedding_model
    return embedding_model


def _render_knowledge() -> None:
    st.title("Knowledge")
    st.caption("Build or update the local LLM index. Existing valid work is reused.")

    source_col, source_button_col = st.columns([0.84, 0.16])
    with source_col:
        source_text = st.text_input(
            "Source folder path",
            key="source_path",
            placeholder=r"C:\path\to\source",
        )
    with source_button_col:
        st.write("")
        st.button("Browse", key="browse_source", on_click=_browse_into, args=("source_path",))

    _render_output_picker("Local index/output folder", "knowledge_output_path_input")
    selected_output = _selected_output()
    existing_repository = (
        _repository_from_index(selected_output)
        if selected_output and _index_exists(selected_output)
        else None
    )
    if selected_output and _index_exists(selected_output):
        if existing_repository and existing_repository.is_dir():
            st.info(
                "Existing index found. Running Generate / Update will reuse completed "
                f"work for `{existing_repository}` and add any non-indexed or changed "
                "folders/files."
            )
            if not source_text:
                source_text = str(existing_repository)
                st.caption(
                    "Source folder is empty, so mapMyVault will use the source recorded "
                    "inside the existing index."
                )
        else:
            st.warning(
                "Existing index found, but its source folder is missing or not recorded. "
                "Choose the source folder before updating so the manifest can be repaired."
            )

    source_for_selection = (
        Path(source_text).expanduser().resolve()
        if source_text
        else existing_repository
    )
    selected_exclusions = _render_source_selection(source_for_selection)

    st.subheader("Local Embeddings")
    locked_embedding_model = _locked_embedding_model(selected_output)
    embedding_model = _render_embedding_model_control(locked_embedding_model)
    st.caption(
        "The chat/summarizer model is selected from the top-left of the Chat page."
    )

    st.subheader("OCR")
    enable_ocr = st.checkbox("Enable local OCR", key="enable_ocr")
    ocr_language = st.text_input("Tesseract language", key="ocr_language")
    ocr_max_pages = st.number_input(
        "OCR max PDF pages",
        min_value=1,
        max_value=500,
        key="ocr_max_pages",
    )

    st.subheader("Image Vision")
    enable_vision = st.checkbox(
        "Enable local YOLO object labels for image files",
        key="enable_vision",
    )
    ultralytics_ready = _ultralytics_available()
    if enable_vision and not ultralytics_ready:
        st.error(
            "`ultralytics` is not installed in this Python environment, so YOLO "
            "cannot run yet."
        )
        st.code("pip install -r requirements-vision.txt", language="powershell")
    yolo_models = _available_yolo_models()
    if yolo_models:
        if st.session_state.vision_model not in yolo_models:
            st.session_state.vision_model = yolo_models[0]
        vision_model = st.selectbox(
            "YOLO model from repo models folder",
            yolo_models,
            index=yolo_models.index(st.session_state.vision_model),
            key="vision_model",
            help=f"Files are loaded from {YOLO_MODELS_DIR}.",
        )
    else:
        vision_model = ""
        st.caption(f"YOLO models are loaded from `{YOLO_MODELS_DIR}`.")
        if enable_vision:
            st.warning(
                "No YOLO `.pt` files found. Download a YOLO weights file into "
                f"`{YOLO_MODELS_DIR}` and refresh this page."
            )
            st.code(
                'Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" -OutFile "models\\yolov8n.pt"',
                language="powershell",
            )
    vision_confidence = st.slider(
        "YOLO confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.vision_confidence),
        step=0.05,
        key="vision_confidence",
    )
    vision_max_detections = st.number_input(
        "YOLO max detections per image",
        min_value=1,
        max_value=500,
        key="vision_max_detections",
    )

    st.subheader("Index")
    st.write(
        "This writes SQLite, Chroma, summaries, and graph data. Obsidian notes are "
        "generated only when you click the Obsidian export button. If the output "
        "folder already contains an index for the same source, this updates it "
        "incrementally instead of reprocessing unchanged files."
    )
    st.button(
        "Generate / Update Local LLM Index",
        type="primary",
        disabled=st.session_state.indexing,
        on_click=_request_indexing,
    )
    if st.session_state.indexing:
        st.info("Index generation is already running. Please wait.")
    if st.session_state.index_requested:
        st.session_state.index_requested = False
        source = Path(source_text).expanduser().resolve() if source_text else None
        output = Path(st.session_state.output_path).expanduser().resolve() if st.session_state.output_path else None
        if not source or not source.is_dir():
            st.error("Choose a valid source folder path.")
            st.session_state.indexing = False
        elif not output:
            st.error("Choose a local index/output folder.")
            st.session_state.indexing = False
        elif enable_vision and not vision_model:
            st.error("Enable YOLO only after placing a `.pt` file in the repo `models` folder.")
            st.session_state.indexing = False
        elif enable_vision and not ultralytics_ready:
            st.error("Install `ultralytics` before running YOLO image vision.")
            st.session_state.indexing = False
        else:
            config = MapperConfig(
                generation_model=st.session_state.generation_model,
                embedding_model=embedding_model,
                enable_ocr=enable_ocr,
                ocr_language=ocr_language,
                ocr_max_pages=int(ocr_max_pages),
                enable_vision=enable_vision,
                vision_model=vision_model,
                vision_confidence=float(vision_confidence),
                vision_max_detections=int(vision_max_detections),
            )
            if selected_exclusions:
                config.excludes = list(
                    dict.fromkeys(config.excludes + _exclude_patterns(selected_exclusions))
                )
            try:
                if output and _index_exists(output):
                    existing_config = _config_from_manifest(output)
                    config.embedding_model = existing_config.embedding_model
                    config.excludes = list(
                        dict.fromkeys(
                            config.excludes
                            + _exclude_patterns(_index_excluded_paths(output))
                        )
                    )
                    st.info(
                        "Using the existing index embedding model: "
                        f"`{config.embedding_model}`."
                    )
                if _ensure_required_models(config):
                    _run_index(source, output, config)
                else:
                    st.session_state.indexing = False
            except Exception as exc:
                st.error(str(exc))
                st.session_state.indexing = False

    output = _selected_output()
    if output and _index_exists(output):
        st.subheader("Indexed Folder View")
        with st.expander("Show indexed files and folders", expanded=True):
            st.code("\n".join(_tree_lines(output)) or "No indexed files.", language="text")
        with st.expander("Index status"):
            st.json(_status(output))
    else:
        st.info("Load or generate a local index to see the folder tree.")

    st.subheader("Obsidian")
    vault_path = output / "obsidian" if output else None
    if vault_path:
        st.caption(f"Obsidian notes will be saved to `{vault_path}`.")
    else:
        st.caption("Choose an output folder first.")
    if st.button("Generate Obsidian Graph / Vault"):
        if not output or not _index_exists(output):
            st.error("Generate or load a local index first.")
        else:
            with st.spinner("Generating Obsidian Markdown notes locally..."):
                try:
                    st.json(export_obsidian_from_index(output, vault_path))
                    st.success("Obsidian vault generated.")
                except Exception as exc:
                    st.error(str(exc))

    st.subheader("MCP For External Local Clients")
    mcp_port = st.number_input("MCP port", min_value=1024, max_value=65535, value=8765)
    process = st.session_state.get("mcp_process")
    running = process is not None and process.poll() is None
    st.caption("Internal chat uses Python directly. MCP is only needed for Open WebUI/OpenClaw.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start MCP", disabled=not output or not _index_exists(output)):
            _start_mcp(output, int(mcp_port))
            st.success(f"MCP running at http://127.0.0.1:{mcp_port}/mcp")
    with col_b:
        if st.button("Stop MCP", disabled=not running):
            _stop_mcp()
            st.info("MCP stopped.")


def _set_graph_folder(path: str) -> None:
    st.session_state.graph_folder = path


def _render_breadcrumbs(current: str) -> None:
    parts = [part for part in current.split("/") if part]
    cols = st.columns(max(1, len(parts) + 1))
    with cols[0]:
        if st.button("Root", key="crumb_root"):
            _set_graph_folder("")
            st.rerun()
    built = []
    for index, part in enumerate(parts, 1):
        built.append(part)
        with cols[index]:
            if st.button(part, key=f"crumb_{index}_{part}"):
                _set_graph_folder("/".join(built))
                st.rerun()


def _render_graph_source_repair(output: Path, repository: Path | None) -> None:
    if repository:
        st.error(
            "This index points to a source folder that is not available anymore: "
            f"`{repository}`"
        )
    else:
        st.error(
            "This index does not record a usable source repository path in "
            "`manifest.json` or SQLite metadata."
        )
    st.info(
        "Choose the original source folder for this index, then repair/update. "
        "mapMyVault will scan that source root, reuse valid existing stages, add "
        "non-indexed folders/files, and mark missing indexed files as deleted."
    )
    repair_col, repair_button_col = st.columns([0.84, 0.16])
    with repair_col:
        repair_source = st.text_input(
            "Source folder for this existing index",
            key="graph_repair_source_path",
            placeholder=r"C:\path\to\original\source",
        )
    with repair_button_col:
        st.write("")
        st.button(
            "Browse",
            key="browse_graph_repair_source",
            on_click=_browse_into,
            args=("graph_repair_source_path",),
        )
    if st.button(
        "Repair source path + update KB",
        type="primary",
        disabled=st.session_state.indexing,
    ):
        source = Path(repair_source).expanduser().resolve() if repair_source else None
        if not source or not source.is_dir():
            st.error("Choose a valid source folder.")
            st.session_state.indexing = False
        else:
            try:
                _set_graph_folder("")
                if _run_graph_update(source, output):
                    st.success("Index repaired and updated.")
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))
                st.session_state.indexing = False


def _item_select_key(path: str) -> str:
    return f"graph_select_{path}"


def _selected_item_paths(items: list[dict]) -> list[str]:
    return [
        item["path"]
        for item in items
        if st.session_state.get(_item_select_key(item["path"]), False)
    ]


def _clear_item_selection(paths: list[str]) -> None:
    for path in paths:
        st.session_state[_item_select_key(path)] = False


def _item_status(item: dict) -> str:
    if not item.get("indexed", True):
        return "Local only"
    if not item.get("exists", True):
        return "Stale index"
    return "Indexed"


def _render_graph_view() -> None:
    st.title("Graph View")
    st.caption(
        "Browse the existing indexed source tree, add local files or folders, then "
        "incrementally update the KB."
    )

    _render_output_picker("Existing index/output folder", "graph_output_path_input")

    output = _selected_output()
    if not output or not _index_exists(output):
        st.info("Choose an existing mapMyVault output folder in the sidebar first.")
        return

    repository = _repository_from_index(output)
    if not repository or not repository.is_dir():
        _render_graph_source_repair(output, repository)
        return

    current = _ensure_graph_folder_available(repository, st.session_state.graph_folder)
    children = _combined_children(output, repository, current)
    folders = [item for item in children if item["kind"] == "folder"]
    files = [item for item in children if item["kind"] == "file"]
    target_dir = _safe_target_dir(repository, current)
    items = folders + files
    selected_paths = _selected_item_paths(items)
    locked_embedding_model = _locked_embedding_model(output)

    st.caption(f"Source root: `{repository}`")
    if locked_embedding_model:
        st.caption(f"Embedding model locked to this index: `{locked_embedding_model}`")
        if locked_embedding_model not in _available_ollama_models():
            st.warning("The locked embedding model is not installed locally.")
            st.code(f"ollama pull {locked_embedding_model}", language="powershell")
    else:
        st.warning("No embedding model is locked to this index yet. Select one before updating.")
        _render_embedding_model_control()

    toolbar = st.columns([0.08, 0.30, 0.28, 0.16, 0.18])
    with toolbar[0]:
        if st.button("←", key="graph_back", disabled=not current):
            _set_graph_folder(_parent_folder(current))
            st.rerun()
    with toolbar[1]:
        st.text_input(
            "Folder name",
            value=Path(current).name if current else repository.name,
            disabled=True,
            key=f"graph_current_name_{current}",
        )
    with toolbar[2]:
        uploads = st.file_uploader(
            "Upload file",
            accept_multiple_files=True,
            key=f"upload_{current}",
        )
    with toolbar[3]:
        st.write("")
        delete_clicked = st.button(
            "Delete",
            disabled=st.session_state.indexing or not selected_paths,
            key="graph_delete_selected",
            use_container_width=True,
        )
    with toolbar[4]:
        st.write("")
        update_clicked = st.button(
            "Update KB",
            type="primary",
            disabled=st.session_state.indexing,
            key="graph_update_toolbar",
            use_container_width=True,
        )

    if uploads:
        if st.button("Save uploaded file(s) here", key=f"save_uploads_{current}"):
            saved = []
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                for upload in uploads:
                    destination = _unique_destination(target_dir, upload.name)
                    destination.write_bytes(upload.getbuffer())
                    saved.append(destination.name)
                st.success(f"Saved {len(saved)} file(s). Click Update KB to index them.")
                st.write(saved)
            except Exception as exc:
                st.error(str(exc))

    if delete_clicked:
        try:
            removed = _remove_index_subtrees(output, selected_paths, persist_exclusion=True)
            _clear_item_selection(selected_paths)
            st.success(
                f"Deleted {len(selected_paths)} item(s) from this index "
                f"and removed {removed} indexed item(s). Source files were not deleted."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if update_clicked:
        try:
            if _run_graph_update(repository, output):
                st.success("KB updated from the graph view.")
        except Exception as exc:
            st.error(str(exc))
            st.session_state.indexing = False

    st.divider()
    st.subheader(current or "Root")
    st.caption("Select folders or files, open a folder, delete selected items from the index, or update the KB.")

    if items:
        header_cols = st.columns([0.08, 0.07, 0.55, 0.30])
        header_cols[0].caption("Select")
        header_cols[1].caption("")
        header_cols[2].caption("Item")
        header_cols[3].caption("Index status")
        for item in items:
            name = Path(item["path"]).name
            row_cols = st.columns([0.08, 0.07, 0.55, 0.30])
            with row_cols[0]:
                st.checkbox(
                    "Select item",
                    key=_item_select_key(item["path"]),
                    label_visibility="collapsed",
                )
            with row_cols[1]:
                st.write("📁")
            with row_cols[2]:
                label = name
                if not item.get("indexed", True):
                    label += " (not indexed yet)"
                elif not item.get("exists", True):
                    label += " (missing on disk)"
                if item["kind"] == "folder":
                    if st.button(label, key=f"folder_{item['path']}", use_container_width=True):
                        _set_graph_folder(item["path"])
                        st.rerun()
                else:
                    st.write(label)
            with row_cols[3]:
                st.caption(_item_status(item))
    else:
        st.caption("No folders or files in this folder.")

    if current:
        st.divider()
        st.subheader("Manage Selected Folder")
        st.caption(
            "Rename or delete the selected source folder, then update the KB. "
            "The mapper reuses completed stages for everything else."
        )

        rename_col, rename_button_col = st.columns([0.72, 0.28])
        current_name = Path(current).name
        with rename_col:
            rename_to = st.text_input(
                "New folder name",
                value=current_name,
                key=f"rename_folder_{current}",
            )
        with rename_button_col:
            st.write("")
            rename_clicked = st.button(
                "Rename + update KB",
                key=f"rename_button_{current}",
                disabled=st.session_state.indexing,
            )
        if rename_clicked:
            try:
                clean_name = _valid_child_name(rename_to)
                if clean_name == current_name:
                    st.info("The folder already has that name.")
                else:
                    source_dir = _safe_target_dir(repository, current)
                    destination = source_dir.with_name(clean_name).resolve()
                    if not destination.is_relative_to(repository.resolve()):
                        raise ValueError("Rename destination is outside the source root.")
                    if destination.exists():
                        raise ValueError(f"`{destination}` already exists.")
                    source_dir.rename(destination)
                    new_relative = _to_posix_relative(repository, destination)
                    _set_graph_folder(new_relative)
                    st.success(f"Renamed folder to `{new_relative}`.")
                    if _run_graph_update(repository, output):
                        st.rerun()
            except Exception as exc:
                st.error(str(exc))
                st.session_state.indexing = False

        with st.expander("Delete this folder from source and index"):
            st.warning(
                "This deletes the selected source folder from disk. The next KB update "
                "marks its old index rows as deleted."
            )
            confirm_delete = st.checkbox(
                f"I understand this will delete `{current}` from disk.",
                key=f"confirm_delete_{current}",
            )
            delete_text = st.text_input(
                "Type the folder name to confirm deletion",
                key=f"delete_text_{current}",
                placeholder=current_name,
            )
            delete_ready = confirm_delete and delete_text == current_name
            if st.button(
                "Delete folder + update KB",
                key=f"delete_button_{current}",
                disabled=st.session_state.indexing or not delete_ready,
                type="secondary",
            ):
                try:
                    source_dir = _safe_target_dir(repository, current)
                    if not source_dir.exists() or not source_dir.is_dir():
                        raise ValueError("Selected folder no longer exists.")
                    parent = _parent_folder(current)
                    shutil.rmtree(source_dir)
                    _set_graph_folder(parent)
                    st.success(f"Deleted `{current}` from disk.")
                    if _run_graph_update(repository, output):
                        st.rerun()
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state.indexing = False

    st.divider()
    st.subheader("Create Folder Here")
    st.caption(
        "Create a local folder inside the current folder. Use the top Upload file "
        "control to add files."
    )

    new_folder = st.text_input("New folder name", key="graph_new_folder")
    if st.button("Create folder"):
        try:
            folder_name = _valid_child_name(new_folder)
            folder_path = _safe_target_dir(target_dir, folder_name)
            folder_path.mkdir(parents=True, exist_ok=True)
            st.session_state.graph_folder = _child_path(current, folder_name)
            st.success(f"Created `{folder_path}`. Click Update KB to index it.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


_session_default("mcp_process", None)
_session_default("indexing", False)
_session_default("index_requested", False)
_session_default("graph_index_requested", False)
_session_default("source_path", "")
_session_default("output_path", "")
_session_default("chat_output_path_input", "")
_session_default("knowledge_output_path_input", "")
_session_default("graph_output_path_input", "")
_session_default("view", "Chat")
_session_default("generation_model", "llama3.1:8b")
_session_default("embedding_model", "nomic-embed-text:latest")
_session_default("chat_history", [])
_session_default("chat_uploads", [])
_session_default("enable_ocr", True)
_session_default("ocr_language", "eng")
_session_default("ocr_max_pages", 10)
_session_default("enable_vision", False)
_session_default("vision_model", "yolov8n.pt")
_session_default("vision_confidence", 0.25)
_session_default("vision_max_detections", 50)
_session_default("graph_folder", "")

_render_sidebar()

if st.session_state.view == "Chat":
    _render_chat()
elif st.session_state.view == "Knowledge":
    _render_knowledge()
elif st.session_state.view == "Graph View":
    _render_graph_view()
