"""Command-line interface for mapMyVault v2."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from .actions import ActionManager
from .app_server import serve_app
from .config import MapperConfig
from .doctor import run_doctor
from .exporter import export_obsidian_from_index
from .mapper import RepositoryMapper
from .mcp_server import serve
from .query import VaultIndex
from .terminal_ui import enable_unicode_output, interactive_map_config


def _print(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _config(path: str = None) -> MapperConfig:
    return MapperConfig.load(Path(path)) if path else MapperConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapmyvault",
        description="🗺️ Fully local, resumable repository intelligence.",
    )
    sub = parser.add_subparsers(dest="command")

    map_parser = sub.add_parser("map", help="🗺️ Map or incrementally update a repository")
    map_parser.add_argument("repository", nargs="?")
    map_parser.add_argument("--output")
    map_parser.add_argument("--config")
    map_parser.add_argument(
        "--generation-model", help="Local Ollama model used for summaries and links"
    )
    map_parser.add_argument(
        "--embedding-model", help="Local Ollama model used for semantic search"
    )
    map_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for all mapping choices, using supplied paths as defaults",
    )
    map_parser.add_argument(
        "--no-progress", action="store_true", help="Disable mapping progress bars"
    )
    map_parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Run optional local OCR for scanned PDFs and images",
    )
    map_parser.add_argument("--ocr-language", help="Tesseract language code, e.g. eng")
    map_parser.add_argument("--ocr-max-pages", type=int, help="Maximum PDF pages to OCR")
    map_parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Run optional local YOLO object detection for image files",
    )
    map_parser.add_argument(
        "--vision-model",
        help="YOLO .pt filename in the repo models folder, or an absolute local path",
    )
    map_parser.add_argument(
        "--vision-confidence",
        type=float,
        help="YOLO confidence threshold, for example 0.25",
    )
    map_parser.add_argument(
        "--vision-max-detections",
        type=int,
        help="Maximum YOLO detections to store per image",
    )
    map_parser.add_argument(
        "--index-only",
        action="store_true",
        help="Generate the local LLM index without exporting Obsidian notes",
    )

    resume = sub.add_parser("resume", help="♻️ Resume an existing index")
    resume.add_argument("output")
    resume.add_argument(
        "--no-progress", action="store_true", help="Disable mapping progress bars"
    )

    status = sub.add_parser("status", help="📊 Show index and stage status")
    status.add_argument("output")

    query = sub.add_parser("query", help="🔎 Search the persisted local index")
    query.add_argument("output")
    query.add_argument("question")
    query.add_argument("--limit", type=int, default=10)

    export = sub.add_parser("export-vault", help="Generate Obsidian vault from an index")
    export.add_argument("output")
    export.add_argument("--vault", help="Vault folder to write; defaults to OUTPUT\\obsidian")

    server = sub.add_parser("serve", help="🔌 Serve local MCP tools over loopback HTTP")
    server.add_argument("output")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)

    app = sub.add_parser("app", help="Run the local browser app")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8787)
    app.add_argument("--no-open", action="store_true", help="Do not open the browser")

    studio = sub.add_parser("studio", help="Run the Streamlit local studio")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8788)

    doctor = sub.add_parser("doctor", help="🩺 Validate local-only configuration")
    doctor.add_argument("--offline", action="store_true")
    doctor.add_argument("--config")

    actions = sub.add_parser("actions", help="✅ Approve or apply action plans")
    action_sub = actions.add_subparsers(dest="action", required=True)
    for name in ("approve", "apply", "rollback"):
        command = action_sub.add_parser(name)
        command.add_argument("output")
        command.add_argument("plan_id")
    return parser


def main(argv=None) -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    enable_unicode_output()
    args = build_parser().parse_args(argv)
    try:
        if args.command is None:
            repository, output, config = interactive_map_config()
            mapper = RepositoryMapper(
                repository, output, config, show_progress=True
            )
            try:
                _print(mapper.map(export_vault_notes=not args.index_only))
            finally:
                mapper.close()
        elif args.command == "map":
            config = _config(args.config)
            if args.generation_model:
                config.generation_model = args.generation_model
            if args.embedding_model:
                config.embedding_model = args.embedding_model
            if args.enable_ocr:
                config.enable_ocr = True
            if args.ocr_language:
                config.ocr_language = args.ocr_language
            if args.ocr_max_pages is not None:
                config.ocr_max_pages = args.ocr_max_pages
            if args.enable_vision:
                config.enable_vision = True
            if args.vision_model:
                config.vision_model = args.vision_model
            if args.vision_confidence is not None:
                config.vision_confidence = args.vision_confidence
            if args.vision_max_detections is not None:
                config.vision_max_detections = args.vision_max_detections
            interactive = args.interactive or not (args.repository and args.output)
            if interactive:
                repository, output, config = interactive_map_config(
                    args.repository, args.output, config
                )
            else:
                repository, output = Path(args.repository), Path(args.output)
            mapper = RepositoryMapper(
                repository,
                output,
                config,
                show_progress=not args.no_progress,
            )
            try:
                _print(mapper.map())
            finally:
                mapper.close()
        elif args.command == "resume":
            manifest = json.loads(
                (Path(args.output) / "data" / "manifest.json").read_text(encoding="utf-8")
            )
            mapper = RepositoryMapper(
                Path(manifest["repository"]),
                Path(args.output),
                MapperConfig(**manifest.get("config", {
                    "ollama_url": manifest["endpoints"]["ollama"],
                    "generation_model": manifest["models"]["generation"],
                    "embedding_model": manifest["models"]["embedding"],
                })),
                show_progress=not args.no_progress,
            )
            try:
                _print(mapper.map())
            finally:
                mapper.close()
        elif args.command in {"status", "query"}:
            index = VaultIndex(Path(args.output))
            try:
                _print(
                    index.status()
                    if args.command == "status"
                    else index.search_files(args.question, args.limit)
                )
            finally:
                index.close()
        elif args.command == "serve":
            serve(Path(args.output), args.host, args.port)
        elif args.command == "export-vault":
            _print(
                export_obsidian_from_index(
                    Path(args.output),
                    Path(args.vault) if args.vault else None,
                )
            )
        elif args.command == "app":
            serve_app(args.host, args.port, open_browser=not args.no_open)
        elif args.command == "studio":
            if args.host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Streamlit studio must bind to a loopback host")
            script = Path(__file__).with_name("streamlit_app.py")
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(script),
                    "--server.address",
                    args.host,
                    "--server.port",
                    str(args.port),
                ]
            )
        elif args.command == "doctor":
            _print(run_doctor(_config(args.config), offline=args.offline))
        elif args.command == "actions":
            manager = ActionManager(Path(args.output))
            try:
                _print(getattr(manager, args.action)(args.plan_id))
            finally:
                manager.close()
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Cancelled.")
        return 130
    except Exception as exc:
        print(f"❌ mapmyvault: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
