# 🗃️ SQLite Schema

The canonical database is:

```text
<mapmyvault-output>/data/index.sqlite
```

SQLite is the source of truth. Chroma, graph JSON, summaries, and Obsidian notes are rebuildable derived artifacts.

## `metadata`

Global key/value metadata.

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Typical keys:

- `schema_version`
- `repository`
- `output`
- `ollama_url`
- `generation_model`
- `embedding_model`
- `summary_version`
- `embedding_version`
- `ocr_version`
- `vision_version`
- `relationship_version`
- `export_version`
- `graph_excluded_paths`

## `files`

Main file/folder table.

```sql
CREATE TABLE files (
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
```

Important fields:

- `id`: stable source-relative path ID.
- `path`: source-relative path.
- `parent`: source-relative parent folder.
- `kind`: `file` or `folder`.
- `content_hash`: detects content changes.
- `extracted_text`: extracted text, OCR text, or YOLO-generated local image analysis text.
- `document_metadata_json`: extraction/OCR/YOLO status and properties.
- `parse_data`: deterministic evidence such as symbols, imports, links, and references.
- `summary_json`: validated local LLM summary.
- `embedding_json`: local embedding vector payload.
- `deleted`: soft-delete marker.

Example `document_metadata_json` for OCR:

```json
{
  "format": "pdf",
  "extraction_status": "ocr_extracted",
  "ocr_engine": "tesseract",
  "ocr_language": "eng",
  "ocr_pages": 8
}
```

Example `document_metadata_json` for YOLO:

```json
{
  "format": "jpg",
  "extraction_status": "vision_extracted",
  "vision_engine": "yolo",
  "vision_status": "vision_extracted",
  "vision_label_counts": {
    "car": 1
  }
}
```

## `stages`

Tracks resumable work per file.

```sql
CREATE TABLE stages (
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
```

Stage names:

- `discovery`
- `extraction`
- `parsing`
- `summary`
- `embedding`
- `relationships`
- `export`

Stage statuses:

- `complete`
- `failed`

## `relationships`

Evidence-backed links between files.

```sql
CREATE TABLE relationships (
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
```

Relationship examples:

- `parent_child`
- `reference`
- `import`
- `content`
- semantic relationship types returned by the local generation model

## `files_fts`

SQLite full-text search index.

```sql
CREATE VIRTUAL TABLE files_fts USING fts5(
    file_id UNINDEXED,
    path,
    content,
    summary
);
```

Search covers:

- file paths
- extracted text
- OCR text
- YOLO analysis text
- summary JSON

## `action_plans`

Stores proposed file operations.

```sql
CREATE TABLE action_plans (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    operations_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    approved_at TEXT,
    applied_at TEXT,
    created_at TEXT NOT NULL
);
```

Agents can propose actions, but applying changes is a separate approval step.

## `action_audit`

Audit history for action plans.

```sql
CREATE TABLE action_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    event TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

## Data Ownership

| Artifact | Owner | Rebuildable? |
|---|---|---|
| `data/index.sqlite` | SQLite / mapMyVault | No, canonical |
| `data/chroma/` | Chroma | Yes |
| `data/graph.json` | Graph export | Yes |
| `data/summaries/` | Summary stage | Yes |
| `obsidian/` | Obsidian export | Yes |
| `data/manifest.json` | Manifest stage | Yes |

