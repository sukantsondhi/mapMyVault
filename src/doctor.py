"""Offline and installation diagnostics."""

from typing import Dict
import importlib.util
import os
import shutil
import socket

from .config import MapperConfig, require_local_url
from .analyzer import _resolve_local_yolo_model
from .local_ai import LocalAI


def run_doctor(config: MapperConfig, offline: bool = True) -> Dict:
    checks = {
        "local_endpoint": False,
        "ollama_reachable": False,
        "generation_model": False,
        "embedding_model": False,
        "ollama_no_cloud": os.environ.get("OLLAMA_NO_CLOUD") == "1",
        "tesseract_available": shutil.which("tesseract") is not None,
        "poppler_available": (
            shutil.which("pdftoppm") is not None
            or shutil.which("pdftocairo") is not None
        ),
        "pymupdf_available": importlib.util.find_spec("fitz") is not None,
        "ultralytics_available": importlib.util.find_spec("ultralytics") is not None,
        "yolo_model_local": _resolve_local_yolo_model(config.vision_model) is not None,
        "hostname": socket.gethostname(),
    }
    require_local_url(config.ollama_url)
    checks["local_endpoint"] = True
    if offline:
        os.environ["OLLAMA_NO_CLOUD"] = "1"
        checks["ollama_no_cloud"] = True
    client = LocalAI(
        config.ollama_url, config.generation_model, config.embedding_model
    )
    try:
        models = client.models()
        checks["ollama_reachable"] = True
        checks["generation_model"] = config.generation_model in models
        checks["embedding_model"] = config.embedding_model in models
    except Exception as exc:
        checks["error"] = str(exc)
    checks["ocr_pdf_renderer_available"] = (
        checks["poppler_available"] or checks["pymupdf_available"]
    )
    checks["ocr_ready"] = (
        checks["tesseract_available"] and checks["ocr_pdf_renderer_available"]
    )
    checks["vision_ready"] = (
        checks["ultralytics_available"] and checks["yolo_model_local"]
    )
    core_ready = all(
        checks[name]
        for name in (
            "local_endpoint",
            "ollama_reachable",
            "generation_model",
            "embedding_model",
            "ollama_no_cloud",
        )
    )
    checks["ready"] = (
        core_ready
        and (checks["ocr_ready"] if config.enable_ocr else True)
        and (checks["vision_ready"] if config.enable_vision else True)
    )
    return checks
