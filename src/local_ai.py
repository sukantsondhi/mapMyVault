"""Strictly local Ollama generation and embedding client."""

from typing import List
import json
import requests

from .config import require_local_url
from .models import FileSummary, RelationshipDecision


class LocalAI:
    def __init__(self, base_url: str, generation_model: str, embedding_model: str):
        self.base_url = require_local_url(base_url)
        self.generation_model = generation_model
        self.embedding_model = embedding_model

    def models(self) -> List[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]

    def _structured(self, prompt: str, model_type):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.generation_model,
                "prompt": prompt,
                "stream": False,
                "format": model_type.model_json_schema(),
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        return model_type.model_validate_json(response.json()["response"])

    def generate(self, prompt: str, images: List[str] | None = None) -> str:
        payload = {
            "model": self.generation_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if images:
            payload["images"] = images
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def answer_from_evidence(
        self,
        question: str,
        history: List[dict],
        evidence: dict,
        images: List[str] | None = None,
        image_names: List[str] | None = None,
    ) -> str:
        safe_history = [
            {
                "role": item.get("role", ""),
                "content": str(item.get("content", ""))[:1200],
            }
            for item in history
        ]
        image_instruction = ""
        if images:
            image_instruction = (
                "The user also attached local image file(s) for the current turn: "
                f"{', '.join(image_names or [])}. Use them only as current-turn visual "
                "context. Do not treat images as indexed database evidence unless the "
                "same fact appears in LOCAL_EVIDENCE.\n\n"
            )
        prompt = (
            "You are answering inside mapMyVault using only the local indexed database. "
            "Use chat history only to understand follow-up wording. Do not use chat "
            "history as factual evidence unless the same fact appears in LOCAL_EVIDENCE. "
            "If LOCAL_EVIDENCE has found_count 0, say no relevant indexed data was found. "
            "Do not use general knowledge. Do not invent paths, counts, dates, people, "
            "or documents. Cite local file paths when available.\n\n"
            f"{image_instruction}"
            f"CHAT_HISTORY:\n{json.dumps(safe_history, ensure_ascii=False)}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"LOCAL_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
            "Answer:"
        )
        return self.generate(prompt, images=images)

    def summarize(
        self, path: str, content: str, parse_data: str, document_metadata: str = "{}"
    ) -> FileSummary:
        return self._structured(
            "Analyze this local file using its CONTENT and extracted document properties, "
            "not only its name. Explain what it contains in enough detail to support search "
            "and relationships. Identify topics, entities, dates, and safe suggested actions. "
            "Generate short lowercase tags for Obsidian-style filtering, using hyphens "
            "instead of spaces. Be factual and do not invent missing information.\n"
            f"PATH: {path}\nDOCUMENT PROPERTIES: {document_metadata}\n"
            f"PARSED EVIDENCE: {parse_data}\nCONTENT:\n{content}",
            FileSummary,
        )

    def judge_relationship(
        self, source_path: str, source_summary: str, target_path: str, target_summary: str
    ) -> RelationshipDecision:
        return self._structured(
            "Decide whether these two repository files are meaningfully related based on "
            "their content summaries. Require concrete evidence and avoid name-only links.\n"
            f"SOURCE {source_path}: {source_summary}\n"
            f"TARGET {target_path}: {target_summary}",
            RelationshipDecision,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embedding_model, "input": texts},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
