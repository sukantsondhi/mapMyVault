"""Interactive terminal prompts for user-friendly local mapping setup."""

from pathlib import Path
from typing import List, Optional
import sys

import inquirer

from .config import MapperConfig
from .local_ai import LocalAI


def enable_unicode_output() -> None:
    """Use UTF-8 for emoji-friendly output on Windows and older terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _prompt_path(
    message: str,
    suggestion: Optional[str] = None,
    must_exist: bool = False,
) -> Path:
    prompt = f"{message} (example: {suggestion})" if suggestion else message
    while True:
        answer = inquirer.text(message=prompt, default="")
        if answer is None:
            raise KeyboardInterrupt
        value = answer.strip()
        if not value:
            print("⚠️  Please enter a folder path.")
            continue
        path = Path(value).expanduser().resolve()
        if must_exist and not path.is_dir():
            print(f"❌ That folder does not exist: {path}")
            continue
        return path


def _choose_model(message: str, models: List[str], default: str) -> str:
    if not models:
        raise RuntimeError("No local Ollama models were found")
    choices = list(models)
    if default in choices:
        choices.remove(default)
        choices.insert(0, default)
    selected_default = default if default in choices else choices[0]
    answer = inquirer.list_input(message, choices=choices, default=selected_default)
    if answer is None:
        raise KeyboardInterrupt
    return answer


def interactive_map_config(
    repository: Optional[str] = None,
    output: Optional[str] = None,
    base_config: Optional[MapperConfig] = None,
) -> tuple[Path, Path, MapperConfig]:
    """Prompt for source, output, and locally installed model choices."""
    enable_unicode_output()
    config = base_config or MapperConfig()
    print("\n🗺️  mapMyVault interactive setup")
    print("🔒 All model processing will use your local Ollama instance.\n")

    repository_path = _prompt_path(
        "📁 Repository or folder to analyze",
        repository or str(Path.cwd()),
        must_exist=True,
    )
    default_output = repository_path.parent / f"{repository_path.name}-mapmyvault"
    output_path = _prompt_path(
        "📦 Output folder",
        output or str(default_output),
    )
    if repository_path == output_path or repository_path in output_path.parents:
        raise ValueError("Output must not be inside the repository being analyzed")

    client = LocalAI(
        config.ollama_url, config.generation_model, config.embedding_model
    )
    models = client.models()
    embedding_models = [model for model in models if "embed" in model.lower()]
    generation_models = [model for model in models if model not in embedding_models]

    config.generation_model = _choose_model(
        "🧠 Generation model for summaries and relationships",
        generation_models or models,
        config.generation_model,
    )
    config.embedding_model = _choose_model(
        "🔎 Embedding model for semantic search",
        embedding_models or models,
        config.embedding_model,
    )
    config.enable_ocr = inquirer.confirm(
        "🔤 Enable local OCR for scanned PDFs and images? Requires Tesseract and Poppler.",
        default=config.enable_ocr,
    )
    if config.enable_ocr:
        language = inquirer.text(
            message="🔤 OCR language code",
            default=config.ocr_language,
        )
        max_pages = inquirer.text(
            message="📄 Maximum PDF pages to OCR per file",
            default=str(config.ocr_max_pages),
        )
        config.ocr_language = (language or config.ocr_language).strip() or "eng"
        try:
            config.ocr_max_pages = int(max_pages)
        except (TypeError, ValueError):
            print("⚠️  Invalid page limit. Keeping the existing OCR page limit.")

    config.enable_vision = inquirer.confirm(
        "🖼️  Enable local YOLO object labels for image files? Requires ultralytics and local .pt weights.",
        default=config.enable_vision,
    )
    if config.enable_vision:
        model = inquirer.text(
            message="🖼️  YOLO .pt filename from repo models folder, or absolute local path",
            default=config.vision_model,
        )
        confidence = inquirer.text(
            message="🎯 YOLO confidence threshold",
            default=str(config.vision_confidence),
        )
        max_detections = inquirer.text(
            message="🔢 Maximum YOLO detections per image",
            default=str(config.vision_max_detections),
        )
        config.vision_model = (model or config.vision_model).strip()
        try:
            config.vision_confidence = float(confidence)
        except (TypeError, ValueError):
            print("⚠️  Invalid confidence. Keeping the existing YOLO threshold.")
        try:
            config.vision_max_detections = int(max_detections)
        except (TypeError, ValueError):
            print("⚠️  Invalid detection limit. Keeping the existing YOLO limit.")

    print("\n✅ Configuration")
    print(f"  📁 Source:     {repository_path}")
    print(f"  📦 Output:     {output_path}")
    print(f"  🧠 Generation: {config.generation_model}")
    print(f"  🔎 Embedding:  {config.embedding_model}")
    print(f"  🔤 OCR:        {'enabled' if config.enable_ocr else 'disabled'}")
    print(
        "  🖼️  Vision:     "
        f"{'enabled (' + config.vision_model + ')' if config.enable_vision else 'disabled'}"
    )
    confirmed = inquirer.confirm("🚀 Start mapping?", default=True)
    if not confirmed:
        raise KeyboardInterrupt
    print()
    return repository_path, output_path, config
