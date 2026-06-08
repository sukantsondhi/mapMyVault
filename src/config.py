"""Configuration and local-only endpoint validation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
import json


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def require_local_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_HOSTS:
        raise ValueError(f"Remote endpoint rejected by local-only policy: {url}")
    return url.rstrip("/")


@dataclass
class MapperConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    generation_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text:latest"
    max_file_bytes: int = 10_000_000
    max_content_chars: int = 24_000
    enable_ocr: bool = False
    ocr_engine: str = "tesseract"
    ocr_language: str = "eng"
    ocr_dpi: int = 200
    ocr_max_pages: int = 10
    ocr_failure_limit: int = 5
    enable_vision: bool = False
    vision_engine: str = "yolo"
    vision_model: str = "yolov8n.pt"
    vision_confidence: float = 0.25
    vision_max_detections: int = 50
    semantic_candidates: int = 8
    relationship_threshold: float = 0.72
    excludes: List[str] = field(
        default_factory=lambda: [
            ".git/",
            ".venv/",
            "venv/",
            "node_modules/",
            "__pycache__/",
            "dist/",
            "build/",
        ]
    )
    summary_prompt_version: str = "summary-v1"
    relationship_prompt_version: str = "relationship-v1"
    export_template_version: str = "vault-v1"

    def __post_init__(self) -> None:
        self.ollama_url = require_local_url(self.ollama_url)

    def versions(self) -> Dict[str, str]:
        return {
            "summary": self.summary_prompt_version,
            "relationship": self.relationship_prompt_version,
            "embedding": self.embedding_model,
            "ocr": (
                f"{self.ocr_engine}:{self.ocr_language}:"
                f"{self.ocr_dpi}:{self.ocr_max_pages}:{self.enable_ocr}"
            ),
            "vision": (
                f"{self.vision_engine}:{self.vision_model}:"
                f"{self.vision_confidence}:{self.vision_max_detections}:"
                f"{self.enable_vision}"
            ),
            "export": self.export_template_version,
        }

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> "MapperConfig":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))
