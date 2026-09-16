"""Streamlit UI for local-first mapMyVault workflows."""

from contextlib import contextmanager
from pathlib import Path
import base64
import importlib.util
import json
import os
import shutil
import sqlite3
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


MENU_ITEMS = ["Chat", "Knowledge", "Explorer"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODELS_DIR = PROJECT_ROOT / "models"


def _studio_style() -> None:
    st.html("""
        <style>
        :root { --vault-ink: #242a2e; --vault-green: #15765d; --vault-line: #dce3e0; }
        .stApp { background: #fcfcfc; color: var(--vault-ink); }
        .stApp, .stApp input, .stApp textarea, .stApp button {
            font-family: "Aptos", "Trebuchet MS", sans-serif; letter-spacing: 0;
        }
        h1, h2, h3 { font-family: "Bahnschrift", "Trebuchet MS", sans-serif !important;
            font-weight: 600 !important; letter-spacing: 0 !important; }
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stMainBlockContainer { max-width: 1180px; padding: 2.5rem 2.5rem 2rem; }
        [data-testid="stSidebar"] { background: #f1f4f3; border-right: 1px solid var(--vault-line); }
        [data-testid="stSidebar"] h1 { font-size: 1.7rem !important; }
        [data-testid="stToolbar"] { pointer-events: none; }
        [data-testid="stToolbar"] button { pointer-events: auto; }
        [data-testid="stSidebar"] [data-testid="stButton"] button { justify-content: flex-start; }
        [data-testid="stSidebar"] .st-key-vault_identity { padding: .5rem 0 1rem; }
        .stApp button { border-radius: 6px !important; min-height: 2.5rem; }
        [class*="st-key-browse_"] button p,
        [class*="st-key-remove_chat_upload_"] button p {
            position: absolute; width: 1px; height: 1px; overflow: hidden;
            clip-path: inset(50%);
        }
        .stApp button[kind="primary"] { background: var(--vault-green); border-color: var(--vault-green); }
        .stApp button:focus-visible { outline: 2px solid #a34c27; outline-offset: 3px; }
        .stApp [data-testid="stMetric"] { border-bottom: 2px solid var(--vault-line); padding: .3rem 0 .8rem; }
        .stApp [data-testid="stMetricValue"] { font-family: "Bahnschrift", sans-serif; font-size: 1.8rem; }
        .stApp [data-testid="stMetricLabel"] { color: #55645e; }
        .stApp [data-testid="stExpander"] details { border-radius: 6px; border-color: var(--vault-line); }
        .stApp [data-testid="stChatMessage"] { border-radius: 6px; background: #f1f4f3; }
        .st-key-chat_empty { padding: 3rem 1rem; margin: 1rem 0;
            border-top: 1px solid var(--vault-line); border-bottom: 1px solid var(--vault-line);
            background-image: radial-gradient(#dce3e0 .7px, transparent .7px);
            background-size: 16px 16px; }
        .st-key-chat_empty h3 { font-size: 1.5rem !important; }
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stCaptionContainer"], .stApp button p {
            overflow-wrap: anywhere; white-space: normal;
        }
        .stApp [data-testid="stTabs"] [role="tab"] { font-size: .95rem; }
        .stApp [data-testid="stGraphVizChart"] svg { max-height: 420px; }
        @media (max-width: 640px) {
            .stMainBlockContainer { padding: 1.5rem 1rem 1rem; }
            h1 { font-size: 1.7rem !important; }
            .st-key-chat_empty { padding: 1.5rem .5rem; }
            .st-key-index_summary [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .75rem; }
            .st-key-index_summary [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; }
            .st-key-index_summary [data-testid="stMetricLabel"] { min-height: 2.5rem; }
        }
        </style>
    """)


def _go_to(view: str) -> None:
    st.session_state.view = view


def _render_index_summary(output: Path) -> None:
    status = _status(output)
    with st.container(key="index_summary"):
        for column, label, key in zip(
            st.columns(3), ["Indexed files", "Folders", "Connections"],
            ["files", "folders", "relationships"],
        ):
            column.metric(label, f"{status.get(key, 0):,}")


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
    value = st.session_state.get(source_key, "")
    if target_key == "output_path":
        _activate_output(value)
    else:
        st.session_state[target_key] = value


def _clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.chat_uploads = []
    st.session_state.chat_upload_version = st.session_state.get("chat_upload_version", 0) + 1


def _activate_output(value: str) -> None:
    value = value.strip()
    if value == st.session_state.get("output_path", ""):
        return
    _stop_mcp()
    _clear_chat()
    st.session_state.output_path = value
    st.session_state.source_path = ""
    st.session_state.graph_folder = ""
    st.session_state.explorer_filter = ""
    st.session_state.index_error = ""
    for key in list(st.session_state):
        if key.startswith(("graph_select_", "knowledge_include_", "source_exclusions_", "open_folder_")) or key in {
            "chat_generation_model_select", "embedding_model_select"
        }:
            del st.session_state[key]
    try:
        output = _selected_output()
        config = _config_from_manifest(output) if output and _index_exists(output) else MapperConfig()
        for key in ("generation_model", "embedding_model", "enable_ocr", "ocr_language", "ocr_max_pages",
                    "enable_vision", "vision_model", "vision_confidence", "vision_max_detections"):
            st.session_state[key] = getattr(config, key)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.session_state.index_error = str(exc)


def _prepare_synced_input(widget_key: str, source_key: str) -> None:
    source_value = st.session_state.get(source_key, "")
    if st.session_state.get(widget_key, "") != source_value:
        st.session_state[widget_key] = source_value


def _browse_synced_folder(widget_key: str, source_key: str) -> None:
    selected = _pick_folder(st.session_state.get(widget_key, ""))
    st.session_state[widget_key] = selected
    _sync_state(widget_key, source_key)


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
    if _is_count_question(question):
        evidence["answer"] = deterministic_answer(evidence)
        evidence["deterministic"] = True
        evidence["model_used"] = "local deterministic count"
        return evidence
    if evidence.get("found_count", 0) == 0 and not images:
        evidence["model_used"] = "none"
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
    model_used = result.get("model_used") or generation_model
    paths = result.get("paths", [])
    if paths:
        answer += "\n\nSources:\n" + "\n".join(f"- `{path}`" for path in paths[:20])
    if result.get("deterministic"):
        answer += (
            "\n\n_Note: this is an exact local database count, so the chat model "
            "was not used._"
        )
    elif result.get("model_used") and result["model_used"] != "none":
        answer += f"\n\n_Model used: `{model_used}`_"
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
        existing.add(key)
    st.session_state.chat_uploads = uploads


def _remove_chat_upload(index: int) -> None:
    uploads = list(st.session_state.get("chat_uploads", []))
    if 0 <= index < len(uploads):
        uploads.pop(index)
    st.session_state.chat_uploads = uploads
    st.session_state.chat_upload_version = st.session_state.get("chat_upload_version", 0) + 1


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


def _source_tree_items(source: Path, max_items: int | None = None) -> list[dict]:
    if not source or not source.is_dir():
        return []
    items = []
    for item in sorted(source.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower())):
        if max_items is not None and len(items) >= max_items:
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
    return f"knowledge_include_{st.session_state.get('source_path', '')}_{path}"


def _select_exclusions(key: str, paths: list[str]) -> None:
    st.session_state[key] = paths


def _render_source_selection(source: Path | None) -> list[str]:
    if not source or not source.is_dir():
        return []
    items = _source_tree_items(source)
    if not items:
        return []
    paths = [item["path"] for item in items]
    key = f"source_exclusions_{source}"
    controls = st.columns(2)
    controls[0].button("Include all", icon=":material/done_all:",
                       on_click=_select_exclusions, args=(key, []))
    controls[1].button("Exclude all", icon=":material/remove_done:",
                       on_click=_select_exclusions, args=(key, paths))
    excluded = st.multiselect(
        "Excluded source items", paths,
        default=[path for path in paths if not st.session_state.get(_include_key(path), True)],
        key=key,
    )
    for path in paths:
        st.session_state[_include_key(path)] = path not in excluded
    st.dataframe(
        [{"Item": item["name"], "Type": item["kind"],
          "Index": "Excluded" if item["path"] in excluded else "Included"} for item in items],
        hide_index=True, width="stretch", height=240,
    )
    st.caption(f"{len(items) - len(excluded)} included / {len(excluded)} excluded")
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
    try:
        with st.spinner("Updating the index..."):
            config = _graph_update_config(output)
            if not _ensure_required_models(config):
                return False
            _run_index(repository, output, config)
            return True
    finally:
        st.session_state.indexing = False


def _graph_update_config(output: Path) -> MapperConfig:
    config = _config_from_manifest(output)
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


def _restore_index_paths(output: Path, paths: list[str]) -> None:
    with _index_lock(output):
        excluded = _index_excluded_paths(output)
        patterns = set(_exclude_patterns(paths))
        manifest = _manifest(output)
        config = manifest.get("config")
        if config:
            config["excludes"] = [pattern for pattern in config.get("excludes", []) if pattern not in patterns]
            (output / "data" / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        _set_index_excluded_paths(output, [path for path in excluded if path not in paths])


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
            status.success("Index ready.")
            with st.expander("Run details"):
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
        with st.container(key="vault_identity"):
            st.title("mapMyVault")
            st.caption("STUDIO / LOCAL WORKSPACE")
        icons = {"Chat": ":material/chat_bubble_outline:", "Knowledge": ":material/database:", "Explorer": ":material/folder_open:"}
        for item in MENU_ITEMS:
            active = item == st.session_state.view
            st.button(
                item,
                key=f"nav_{item}",
                type="primary" if active else "secondary",
                icon=icons[item],
                use_container_width=True,
                on_click=_go_to,
                args=(item,),
                disabled=st.session_state.indexing,
            )
        st.divider()
        st.subheader("Active index")
        _render_output_picker("Index folder", "active_output_path_input")
        output = _selected_output()
        if output and _index_exists(output):
            st.success("Index connected", icon=":material/check_circle:")
            st.caption(output.name)
        elif output:
            st.warning("Index not built yet", icon=":material/info:")
        else:
            st.caption("No index selected")
        st.divider()
        models = _available_ollama_models()
        st.caption("LOCAL SERVICES")
        st.write("Ollama connected" if models else "Ollama unavailable")
        st.caption(f"{len(models)} installed models" if models else "127.0.0.1:11434")
        if st.button("Refresh services", icon=":material/refresh:", use_container_width=True):
            _available_ollama_models.clear()
            st.rerun()


def _render_output_picker(label: str, widget_key: str) -> str:
    _prepare_synced_input(widget_key, "output_path")
    output_col, output_button_col = st.columns([4, 1], vertical_alignment="bottom")
    with output_col:
        output_text = st.text_input(
            label,
            key=widget_key,
            placeholder=r"C:\path\to\mapmyvault-output",
            on_change=_sync_state,
            args=(widget_key, "output_path"),
            disabled=st.session_state.indexing,
        )
    with output_button_col:
        st.button(
            "Browse for index folder",
            icon=":material/folder_open:",
            help="Browse for index folder",
            key=f"browse_{widget_key}",
            on_click=_browse_synced_folder,
            args=(widget_key, "output_path"),
            disabled=st.session_state.indexing,
        )
    return output_text


def _render_chat() -> None:
    output = _selected_output()
    ready = bool(output and _index_exists(output))
    st.caption("WORKSPACE / CHAT")
    st.title("Chat")
    model_col, clear_col = st.columns([3, 1], vertical_alignment="bottom")
    with model_col:
        generation_model = _render_chat_model_control()
    with clear_col:
        st.button("New chat", key="clear_chat_history", icon=":material/edit_square:",
                  on_click=_clear_chat, use_container_width=True,
                  disabled=not st.session_state.chat_history and not st.session_state.chat_uploads)

    if not st.session_state.chat_history:
        with st.container(key="chat_empty"):
            st.subheader("A new conversation" if ready else "No index connected")
            if ready:
                _render_index_summary(output)
            else:
                st.button("Set up an index", icon=":material/add:", type="primary",
                          on_click=_go_to, args=("Knowledge",))

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence"):
                with st.expander("Local evidence"):
                    st.json(message["evidence"])

    with st.expander("Image attachments", icon=":material/attach_file:"):
        if not _is_vision_model(generation_model):
            st.caption("The selected model may not support images.")
        uploads = st.file_uploader(
            "Attach images",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key=f"chat_image_uploader_{st.session_state.chat_upload_version}",
            disabled=not ready,
        )
        if uploads:
            _add_chat_uploads(uploads)
        for position, item in enumerate(st.session_state.chat_uploads):
            columns = st.columns([1, 4, 1], vertical_alignment="center")
            columns[0].image(item["data"], width=56)
            columns[1].caption(f"{item['name']} / {item['size']:,} bytes")
            columns[2].button("Remove attachment", icon=":material/close:",
                              help="Remove attachment",
                              key=f"remove_chat_upload_{position}",
                              on_click=_remove_chat_upload, args=(position,))

    question = st.chat_input("Ask your local vault", disabled=not ready)
    if not question or not question.strip():
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
        try:
            answer, result = _generate_chat_answer(
                output, question, history, generation_model,
                images=image_payloads, image_names=image_names,
            )
        except Exception as exc:
            answer, result = f"Unable to query this index: {exc}", None
        st.markdown(answer)
        if result:
            with st.expander("Local evidence"):
                st.json(result)
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "evidence": result})
    st.session_state.chat_uploads = []
    st.session_state.chat_upload_version += 1
    st.rerun()


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
    st.caption("WORKSPACE / KNOWLEDGE")
    st.title("Knowledge")
    output = _selected_output()
    if output and _index_exists(output):
        _render_index_summary(output)
    build_tab, integrations_tab = st.tabs(["Build index", "Export & connect"])
    with build_tab:
        _render_index_builder()
    with integrations_tab:
        _render_integrations()


def _render_index_builder() -> None:
    selected_output = _selected_output()
    existing_repository = (
        _repository_from_index(selected_output)
        if selected_output and _index_exists(selected_output)
        else None
    )
    if not st.session_state.source_path and existing_repository:
        st.session_state.source_path = str(existing_repository)
    st.subheader("Source & models")
    source_col, source_button_col = st.columns([8, 1], vertical_alignment="bottom")
    with source_col:
        source_text = st.text_input(
            "Source folder path",
            key="source_path",
            placeholder=r"C:\path\to\source",
        )
    with source_button_col:
        st.button("Browse for source folder", icon=":material/folder_open:", help="Browse for source folder",
                  key="browse_source", on_click=_browse_into, args=("source_path",))

    if selected_output and _index_exists(selected_output):
        if not existing_repository or not existing_repository.is_dir():
            st.warning(
                "The original source folder is unavailable. Confirm its new location before updating."
            )

    source_for_selection = (
        Path(source_text).expanduser().resolve()
        if source_text
        else existing_repository
    )
    with st.expander("Source contents", icon=":material/checklist:"):
        selected_exclusions = _render_source_selection(source_for_selection)
        if not source_for_selection or not source_for_selection.is_dir():
            st.caption("No source folder selected")

    model_columns = st.columns(2)
    locked_embedding_model = _locked_embedding_model(selected_output)
    with model_columns[0]:
        _render_chat_model_control()
    with model_columns[1]:
        embedding_model = _render_embedding_model_control(locked_embedding_model)

    with st.expander("Text recognition / OCR", icon=":material/document_scanner:"):
        enable_ocr = st.toggle("Enable local OCR", key="enable_ocr")
        ocr_columns = st.columns(2)
        ocr_language = ocr_columns[0].text_input("Tesseract language", key="ocr_language", disabled=not enable_ocr)
        ocr_max_pages = ocr_columns[1].number_input(
            "OCR max PDF pages", min_value=1, max_value=500, key="ocr_max_pages", disabled=not enable_ocr,
        )

    with st.expander("Object detection / YOLO", icon=":material/image_search:"):
        enable_vision = st.toggle("Enable local object detection", key="enable_vision")
        ultralytics_ready = _ultralytics_available()
        if enable_vision and not ultralytics_ready:
            st.error("YOLO requires the optional vision dependencies.")
            st.code("pip install -r requirements-vision.txt", language="powershell")
        yolo_models = _available_yolo_models()
        if yolo_models:
            if st.session_state.vision_model not in yolo_models:
                st.session_state.vision_model = yolo_models[0]
            vision_model = st.selectbox(
                "YOLO model", yolo_models, key="vision_model", disabled=not enable_vision,
                help=f"Local weights in {YOLO_MODELS_DIR}",
            )
        else:
            vision_model = ""
            if enable_vision:
                st.warning(f"No local YOLO weights found in {YOLO_MODELS_DIR}.")
        vision_columns = st.columns(2)
        vision_confidence = vision_columns[0].slider(
            "Confidence threshold", min_value=0.05, max_value=0.95, step=0.05,
            key="vision_confidence", disabled=not enable_vision,
        )
        vision_max_detections = vision_columns[1].number_input(
            "Max detections per image", min_value=1, max_value=500,
            key="vision_max_detections", disabled=not enable_vision,
        )

    st.divider()
    if not selected_output:
        st.info("No output folder selected.")
    st.button(
        "Update index" if selected_output and _index_exists(selected_output) else "Build index",
        icon=":material/sync:",
        type="primary",
        disabled=st.session_state.indexing or not source_text or not selected_output,
        on_click=_request_indexing,
        key="build_index",
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


def _render_integrations() -> None:
    output = _selected_output()
    ready = bool(output and _index_exists(output))
    st.subheader("Obsidian export")
    vault_path = output / "obsidian" if output else None
    if vault_path:
        st.caption(str(vault_path))
    if st.button("Export vault", icon=":material/ios_share:", disabled=not ready):
        if not output or not _index_exists(output):
            st.error("Generate or load a local index first.")
        else:
            with st.spinner("Generating Obsidian Markdown notes locally..."):
                try:
                    st.json(export_obsidian_from_index(output, vault_path))
                    st.success("Obsidian vault generated.")
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    st.subheader("MCP server")
    process = st.session_state.get("mcp_process")
    running = process is not None and process.poll() is None
    mcp_port = st.number_input("MCP port", min_value=1024, max_value=65535, value=8765, disabled=running)
    st.caption("Running" if running else "Stopped")
    if running:
        st.code(f"http://127.0.0.1:{mcp_port}/mcp", language="text")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start MCP", icon=":material/play_arrow:", disabled=not ready or running):
            _start_mcp(output, int(mcp_port))
            st.rerun()
    with col_b:
        if st.button("Stop MCP", icon=":material/stop:", disabled=not running):
            _stop_mcp()
            st.rerun()


def _set_graph_folder(path: str) -> None:
    st.session_state.graph_folder = path
    st.session_state.reset_explorer_filter = True


def _open_graph_folder(widget_key: str) -> None:
    folder = st.session_state.get(widget_key, "")
    if folder:
        _set_graph_folder(folder)
    st.session_state[widget_key] = ""


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
    st.caption("WORKSPACE / EXPLORER")
    st.title("Explorer")
    output = _selected_output()
    if not output or not _index_exists(output):
        st.info("No index connected.")
        st.button("Set up an index", icon=":material/add:", on_click=_go_to, args=("Knowledge",))
        return

    _render_index_summary(output)
    files_tab, connections_tab = st.tabs(["Files", "Connections"])
    with connections_tab:
        _render_connections(output)
    with files_tab:
        _render_file_explorer(output)


def _render_connections(output: Path) -> None:
    store = IndexStore(output / "data" / "index.sqlite")
    try:
        relationships = [dict(row) for row in store.connection.execute(
            "SELECT source.path source, target.path target, relation.type, "
            "relation.confidence, relation.explanation "
            "FROM relationships relation "
            "JOIN files source ON source.id=relation.source_id "
            "JOIN files target ON target.id=relation.target_id "
            "WHERE source.deleted=0 AND target.deleted=0 "
            "ORDER BY relation.confidence DESC, source.path, target.path LIMIT 80"
        )]
    finally:
        store.close()
    if not relationships:
        st.info("No connections recorded in this index.")
        return
    st.caption("Up to 80 highest-confidence connections")
    paths = sorted({row[key] for row in relationships for key in ("source", "target")})
    nodes = [
        f'{json.dumps(path)} [label={json.dumps(Path(path).name)}, tooltip={json.dumps(path)}];'
        for path in paths
    ]
    edges = [
        f'{json.dumps(row["source"])} -> {json.dumps(row["target"])};'
        for row in relationships
    ]
    st.graphviz_chart(
        'digraph { graph [rankdir=LR, bgcolor="transparent", pad="0.2"]; '
        'node [shape=box, style="rounded,filled", fillcolor="#e7f1ec", '
        'color="#6e9e87", fontname="Helvetica", fontsize=11]; '
        'edge [color="#7297a5", arrowsize=0.5]; '
        + "\n".join(nodes + edges) + "}",
        width="stretch",
    )
    st.dataframe(relationships, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1)})


def _render_file_explorer(output: Path) -> None:
    repository = _repository_from_index(output)
    if not repository or not repository.is_dir():
        _render_graph_source_repair(output, repository)
        return

    current = _ensure_graph_folder_available(repository, st.session_state.graph_folder)
    items = _combined_children(output, repository, current)
    target_dir = _safe_target_dir(repository, current)
    st.caption(str(repository / current))
    toolbar = st.columns(2)
    toolbar[0].button("Parent folder", icon=":material/arrow_upward:", key="graph_back", disabled=not current,
                      on_click=_set_graph_folder, args=(_parent_folder(current),))
    if toolbar[1].button("Update index", icon=":material/sync:", type="primary",
                         disabled=st.session_state.indexing, key="graph_update_toolbar"):
        try:
            if _run_graph_update(repository, output):
                st.rerun()
        except Exception as exc:
            st.error(str(exc))
    folders = [item["path"] for item in items if item["kind"] == "folder" and item.get("exists")]
    if folders:
        folder_key = f"open_folder_{output}_{current}"
        st.selectbox("Open folder", [""] + folders,
                     format_func=lambda path: Path(path).name if path else "Select a folder",
                     key=folder_key, on_change=_open_graph_folder, args=(folder_key,))

    search = st.text_input("Filter current folder", key="explorer_filter", placeholder="File or folder name")
    visible = [item for item in items if search.casefold() in Path(item["path"]).name.casefold()]
    selected_paths = []
    if visible:
        row_signature = hash(tuple(item["path"] for item in visible))
        table = st.dataframe(
            [{"Name": Path(item["path"]).name, "Type": item["kind"], "Status": _item_status(item)} for item in visible],
            hide_index=True, width="stretch", height=320,
            on_select="rerun", selection_mode="multi-row",
            key=f"explorer_items_{output}_{current}_{search}_{row_signature}_{st.session_state.get('explorer_version', 0)}",
        )
        selected_paths = [visible[position]["path"] for position in table.selection.rows if position < len(visible)]
    else:
        st.info("No matching items." if search else "This folder is empty.")
    st.caption(f"{len(visible)} items / {len(selected_paths)} selected")
    if st.button("Remove from index", icon=":material/remove_circle_outline:", key="graph_delete_selected",
                 disabled=not selected_paths or st.session_state.indexing):
        try:
            removed = _remove_index_subtrees(output, selected_paths, persist_exclusion=True)
            st.session_state.explorer_version = st.session_state.get("explorer_version", 0) + 1
            st.session_state.notice = f"Removed {removed} indexed items. Source files were not deleted."
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if len(selected_paths) == 1:
        index = VaultIndex(output)
        try:
            preview = index.get_file_summary(selected_paths[0])
        finally:
            index.close()
        if preview:
            with st.expander("File details", expanded=True):
                st.write(preview.get("summary", {}).get("purpose", ""))
                st.text(preview.get("excerpt") or "No extracted text.")

    excluded = _index_excluded_paths(output)
    if excluded:
        with st.expander(f"Excluded paths ({len(excluded)})"):
            restore = st.multiselect("Paths to allow again", excluded)
            if st.button("Allow indexing again", icon=":material/restore:", disabled=not restore):
                _restore_index_paths(output, restore)
                st.session_state.notice = "Exclusions cleared. These paths will be included in the next index update."
                st.rerun()

    with st.expander("Add files", icon=":material/upload_file:"):
        upload_key = f"upload_{output}_{current}_{st.session_state.get('graph_upload_version', 0)}"
        uploads = st.file_uploader("Local files", accept_multiple_files=True, key=upload_key)
        if st.button("Save files", icon=":material/save:", disabled=not uploads):
            try:
                for upload in uploads:
                    destination = _unique_destination(target_dir, upload.name)
                    destination.write_bytes(upload.getbuffer())
                st.session_state.graph_upload_version = st.session_state.get("graph_upload_version", 0) + 1
                st.session_state.notice = f"Saved {len(uploads)} files. Index update pending."
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with st.expander("Folder actions", icon=":material/create_new_folder:"):
        _render_folder_actions(repository, output, current, target_dir)


def _render_folder_actions(repository: Path, output: Path, current: str, target_dir: Path) -> None:
    if current:
        st.subheader("Rename folder")

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
                icon=":material/drive_file_rename_outline:",
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

        with st.container():
            st.subheader("Delete from disk")
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
                icon=":material/delete_forever:",
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
    st.subheader("Create folder")

    new_folder = st.text_input("New folder name", key="graph_new_folder")
    if st.button("Create folder", icon=":material/create_new_folder:", key="graph_create_folder"):
        try:
            folder_name = _valid_child_name(new_folder)
            folder_path = _safe_target_dir(target_dir, folder_name)
            folder_path.mkdir(parents=True, exist_ok=True)
            _set_graph_folder(_child_path(current, folder_name))
            st.session_state.notice = f"Created {folder_path}. Index update pending."
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="mapMyVault Studio", layout="wide")
    _studio_style()
    defaults = {
        "mcp_process": None,
        "indexing": False,
        "index_requested": False,
        "graph_index_requested": False,
        "source_path": "",
        "output_path": "",
        "view": "Chat",
        "generation_model": "llama3.1:8b",
        "embedding_model": "nomic-embed-text:latest",
        "chat_history": [],
        "chat_uploads": [],
        "chat_upload_version": 0,
        "enable_ocr": False,
        "ocr_language": "eng",
        "ocr_max_pages": 10,
        "enable_vision": False,
        "vision_model": "yolov8n.pt",
        "vision_confidence": 0.25,
        "vision_max_detections": 50,
        "graph_folder": "",
    }
    for key, value in defaults.items():
        _session_default(key, value)
    if st.session_state.pop("reset_explorer_filter", False):
        st.session_state.explorer_filter = ""
    _render_sidebar()
    if st.session_state.get("notice"):
        st.success(st.session_state.pop("notice"))
    if st.session_state.get("index_error"):
        st.error(f"Unable to open this index: {st.session_state.index_error}")
        return
    try:
        if st.session_state.view == "Chat":
            _render_chat()
        elif st.session_state.view == "Knowledge":
            _render_knowledge()
        elif st.session_state.view == "Explorer":
            _render_graph_view()
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error(f"Unable to read this index: {exc}")


if __name__ == "__main__":
    main()
