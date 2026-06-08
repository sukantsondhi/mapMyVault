"""SQLite source of truth for resumable repository indexing."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import sqlite3
from datetime import datetime, timezone


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexStore:
    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = database_path
        self.connection = sqlite3.connect(str(database_path), timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                parent TEXT,
                kind TEXT NOT NULL,
                extension TEXT,
                size INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                modified_ns INTEGER,
                extracted_text TEXT,
                document_metadata_json TEXT,
                parse_data TEXT,
                summary_json TEXT,
                embedding_json TEXT,
                deleted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stages (
                file_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_hash TEXT,
                version TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(file_id, stage),
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                deterministic INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(source_id, target_id, type)
            );
            CREATE TABLE IF NOT EXISTS action_plans (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                operations_json TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                approved_at TEXT,
                applied_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                event TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                file_id UNINDEXED, path, content, summary
            );
            """
        )
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(files)")
        }
        if "document_metadata_json" not in columns:
            self.connection.execute(
                "ALTER TABLE files ADD COLUMN document_metadata_json TEXT"
            )
        self.set_metadata("schema_version", str(SCHEMA_VERSION))
        self.connection.commit()

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def upsert_file(self, record: Dict) -> None:
        self.connection.execute(
            """
            INSERT INTO files(id,path,parent,kind,extension,size,content_hash,modified_ns,deleted,updated_at)
            VALUES(:id,:path,:parent,:kind,:extension,:size,:content_hash,:modified_ns,0,:updated_at)
            ON CONFLICT(id) DO UPDATE SET
              path=excluded.path,parent=excluded.parent,kind=excluded.kind,
              extension=excluded.extension,size=excluded.size,
              content_hash=excluded.content_hash,modified_ns=excluded.modified_ns,
              deleted=0,updated_at=excluded.updated_at
            """,
            record,
        )

    def get_file(self, file_id: str):
        return self.connection.execute(
            "SELECT * FROM files WHERE id=?", (file_id,)
        ).fetchone()

    def get_file_by_path(self, path: str):
        return self.connection.execute(
            "SELECT * FROM files WHERE path=? AND deleted=0", (path,)
        ).fetchone()

    def files(self, kind: Optional[str] = None) -> List[sqlite3.Row]:
        if kind:
            return list(
                self.connection.execute(
                    "SELECT * FROM files WHERE deleted=0 AND kind=? ORDER BY path", (kind,)
                )
            )
        return list(
            self.connection.execute("SELECT * FROM files WHERE deleted=0 ORDER BY path")
        )

    def mark_missing_deleted(self, current_ids: Iterable[str]) -> List[str]:
        current = set(current_ids)
        removed = []
        for row in self.connection.execute("SELECT id,path FROM files WHERE deleted=0"):
            if row["id"] not in current:
                removed.append(row["id"])
                self.connection.execute("UPDATE files SET deleted=1 WHERE id=?", (row["id"],))
                self.connection.execute(
                    "DELETE FROM relationships WHERE source_id=? OR target_id=?",
                    (row["id"], row["id"]),
                )
        return removed

    def mark_path_tree_deleted(self, path: str) -> List[str]:
        """Soft-delete one indexed path and all indexed descendants."""
        normalized = path.strip("/")
        rows = list(
            self.connection.execute(
                """
                SELECT id,path FROM files
                WHERE deleted=0 AND (path=? OR path LIKE ?)
                ORDER BY path
                """,
                (normalized, f"{normalized}/%"),
            )
        )
        removed = []
        for row in rows:
            removed.append(row["id"])
            self.connection.execute("UPDATE files SET deleted=1 WHERE id=?", (row["id"],))
            self.connection.execute(
                "DELETE FROM relationships WHERE source_id=? OR target_id=?",
                (row["id"], row["id"]),
            )
        self.connection.commit()
        return removed

    def invalidate(self, file_id: str, stages: Iterable[str]) -> None:
        for stage in stages:
            self.connection.execute(
                "DELETE FROM stages WHERE file_id=? AND stage=?", (file_id, stage)
            )
        if "relationships" in stages:
            self.connection.execute(
                "DELETE FROM relationships WHERE source_id=? OR target_id=?",
                (file_id, file_id),
            )

    def stage_valid(self, file_id: str, stage: str, version: str) -> bool:
        row = self.connection.execute(
            "SELECT status,version FROM stages WHERE file_id=? AND stage=?",
            (file_id, stage),
        ).fetchone()
        if not (row and row["status"] == "complete" and row["version"] == version):
            return False
        artifact_columns = {
            "extraction": "extracted_text",
            "parsing": "parse_data",
            "summary": "summary_json",
            "embedding": "embedding_json",
        }
        column = artifact_columns.get(stage)
        if column:
            artifact = self.connection.execute(
                f"SELECT {column} value FROM files WHERE id=?", (file_id,)
            ).fetchone()
            return bool(artifact and artifact["value"] is not None)
        return True

    def stage_complete(
        self, file_id: str, stage: str, version: str, artifact_hash: str = ""
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO stages(file_id,stage,status,artifact_hash,version,attempts,error,updated_at)
            VALUES(?,?,?,?,?,1,NULL,?)
            ON CONFLICT(file_id,stage) DO UPDATE SET
              status='complete',artifact_hash=excluded.artifact_hash,version=excluded.version,
              attempts=stages.attempts+1,error=NULL,updated_at=excluded.updated_at
            """,
            (file_id, stage, "complete", artifact_hash, version, utc_now()),
        )
        self.connection.commit()

    def stage_failed(self, file_id: str, stage: str, version: str, error: str) -> None:
        self.connection.execute(
            """
            INSERT INTO stages(file_id,stage,status,version,attempts,error,updated_at)
            VALUES(?,?, 'failed',?,1,?,?)
            ON CONFLICT(file_id,stage) DO UPDATE SET
              status='failed',version=excluded.version,attempts=stages.attempts+1,
              error=excluded.error,updated_at=excluded.updated_at
            """,
            (file_id, stage, version, error[:1000], utc_now()),
        )
        self.connection.commit()

    def update_artifact(self, file_id: str, column: str, value) -> None:
        if column not in {
            "extracted_text",
            "document_metadata_json",
            "parse_data",
            "summary_json",
            "embedding_json",
        }:
            raise ValueError(f"Unsupported artifact column: {column}")
        self.connection.execute(
            f"UPDATE files SET {column}=?,updated_at=? WHERE id=?",
            (value, utc_now(), file_id),
        )
        self.connection.commit()

    def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        explanation: str,
        evidence: List[str],
        deterministic: bool,
    ) -> None:
        if source_id == target_id:
            return
        source_id, target_id = sorted((source_id, target_id))
        self.connection.execute(
            """
            INSERT INTO relationships(source_id,target_id,type,confidence,explanation,evidence_json,deterministic,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(source_id,target_id,type) DO UPDATE SET
              confidence=excluded.confidence,explanation=excluded.explanation,
              evidence_json=excluded.evidence_json,deterministic=excluded.deterministic,
              updated_at=excluded.updated_at
            """,
            (
                source_id,
                target_id,
                relation_type,
                confidence,
                explanation,
                json.dumps(evidence),
                int(deterministic),
                utc_now(),
            ),
        )

    def relationships_for(self, file_id: str) -> List[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT r.*, fs.path source_path, ft.path target_path
                FROM relationships r
                JOIN files fs ON fs.id=r.source_id
                JOIN files ft ON ft.id=r.target_id
                WHERE r.source_id=? OR r.target_id=?
                ORDER BY r.confidence DESC
                """,
                (file_id, file_id),
            )
        )

    def rebuild_fts(self) -> None:
        self.connection.execute("DELETE FROM files_fts")
        for row in self.files("file"):
            self.connection.execute(
                "INSERT INTO files_fts(file_id,path,content,summary) VALUES(?,?,?,?)",
                (
                    row["id"],
                    row["path"],
                    row["extracted_text"] or "",
                    row["summary_json"] or "",
                ),
            )
        self.connection.commit()

    def search(self, query: str, limit: int = 10) -> List[sqlite3.Row]:
        safe_query = " OR ".join(
            f'"{token}"' for token in query.replace('"', " ").split() if token
        )
        if not safe_query:
            return []
        return list(
            self.connection.execute(
                """
                SELECT f.*, bm25(files_fts) rank
                FROM files_fts JOIN files f ON f.id=files_fts.file_id
                WHERE files_fts MATCH ? AND f.deleted=0
                ORDER BY rank LIMIT ?
                """,
                (safe_query, limit),
            )
        )

    def status(self) -> Dict:
        result = {
            "files": self.connection.execute(
                "SELECT COUNT(*) count FROM files WHERE deleted=0 AND kind='file'"
            ).fetchone()["count"],
            "folders": self.connection.execute(
                "SELECT COUNT(*) count FROM files WHERE deleted=0 AND kind='folder'"
            ).fetchone()["count"],
            "relationships": self.connection.execute(
                "SELECT COUNT(*) count FROM relationships"
            ).fetchone()["count"],
            "stages": {},
        }
        for row in self.connection.execute(
            "SELECT stage,status,COUNT(*) count FROM stages GROUP BY stage,status"
        ):
            result["stages"].setdefault(row["stage"], {})[row["status"]] = row["count"]
        return result
