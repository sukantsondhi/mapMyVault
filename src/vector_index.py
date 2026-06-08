"""Rebuildable local Chroma semantic index."""

from pathlib import Path
from typing import Dict, List


class VectorIndex:
    def __init__(self, path: Path):
        self.path = path
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.path))
            self._collection = client.get_or_create_collection(
                "mapmyvault_files", metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def upsert(self, file_id: str, document: str, embedding: List[float], path: str) -> None:
        self.collection.upsert(
            ids=[file_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[{"path": path}],
        )

    def delete(self, file_ids: List[str]) -> None:
        if file_ids:
            self.collection.delete(ids=file_ids)

    def candidates(self, embedding: List[float], limit: int) -> List[Dict]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["distances", "metadatas"],
        )
        return [
            {"id": item_id, "distance": distance, "metadata": metadata}
            for item_id, distance, metadata in zip(
                result["ids"][0], result["distances"][0], result["metadatas"][0]
            )
        ]

