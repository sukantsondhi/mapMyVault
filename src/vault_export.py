"""Generate an Obsidian vault that mirrors the source hierarchy."""

from pathlib import Path
from typing import Dict, List
import json
import os
import re
import tempfile


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".mapmyvault-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def note_path(vault: Path, source_path: str, kind: str) -> Path:
    if kind == "folder":
        folder_name = Path(source_path).name
        return vault / source_path / f"{folder_name}-index.md"
    return vault / f"{source_path}.md"


def link_for(source_path: str, kind: str) -> str:
    if kind == "folder":
        folder_name = Path(source_path).name
        return f"{source_path}/{folder_name}-index"
    return source_path


def _slug_tag(value: str) -> str:
    clean = re.sub(r"[^a-z0-9/_-]+", "-", str(value).strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-_/")
    return clean


def _summary_tags(summary: Dict, document_metadata: Dict) -> List[str]:
    tags = []

    def add(prefix: str, value: str) -> None:
        slug = _slug_tag(value)
        if slug:
            tags.append(f"{prefix}/{slug}")

    add("type", summary.get("document_type") or document_metadata.get("format") or "file")
    if document_metadata.get("extraction_status"):
        add("status", document_metadata["extraction_status"])
    for item in summary.get("tags", []):
        slug = _slug_tag(item)
        if slug:
            tags.append(slug)
    for item in summary.get("topics", []):
        add("topic", item)
    for item in summary.get("concepts", []):
        add("concept", item)
    for item in summary.get("key_entities", [])[:10]:
        add("entity", item)
    return list(dict.fromkeys(tags))


def export_vault(vault: Path, files: List[Dict], relationships: Dict[str, List[Dict]]) -> None:
    active_notes = set()
    by_parent: Dict[str, List[Dict]] = {}
    for row in files:
        by_parent.setdefault(row.get("parent") or "", []).append(row)

    index_lines = ["# Repository Index", "", "## Root items", ""]
    for row in by_parent.get("", []):
        index_lines.append(f"- [[{link_for(row['path'], row['kind'])}]]")
    atomic_write(vault / "INDEX.md", "\n".join(index_lines) + "\n")
    active_notes.add((vault / "INDEX.md").resolve())

    for row in files:
        output = note_path(vault, row["path"], row["kind"])
        active_notes.add(output.resolve())
        parent_link = (
            f"[[{link_for(row['parent'], 'folder')}]]" if row.get("parent") else "[[INDEX]]"
        )
        if row["kind"] == "folder":
            lines = [
                "---",
                json.dumps({"source_path": row["path"], "kind": "folder"}, indent=2),
                "---",
                "",
                f"# {Path(row['path']).name}",
                "",
                f"Parent: {parent_link}",
                "",
                "## Children",
                "",
            ]
            for child in by_parent.get(row["path"], []):
                lines.append(f"- [[{link_for(child['path'], child['kind'])}]]")
        else:
            summary = json.loads(row.get("summary_json") or "{}")
            parsed = json.loads(row.get("parse_data") or "{}")
            document_metadata = json.loads(row.get("document_metadata_json") or "{}")
            tags = _summary_tags(summary, document_metadata)
            lines = [
                "---",
                json.dumps(
                    {
                        "source_path": row["path"],
                        "kind": "file",
                        "concepts": summary.get("concepts", []),
                        "topics": summary.get("topics", []),
                        "tags": tags,
                        "document_type": summary.get("document_type", ""),
                    },
                    indent=2,
                ),
                "---",
                "",
                f"# {summary.get('title') or Path(row['path']).name}",
                "",
                f"Parent: {parent_link}",
                "",
                "## Purpose",
                summary.get("purpose", "No summary available."),
                "",
                "## Detailed Description",
                summary.get("detailed_description", "No detailed description available."),
                "",
                "## Responsibilities",
            ]
            lines.extend(f"- {item}" for item in summary.get("responsibilities", []))
            lines.extend(["", "## Topics"])
            lines.extend(f"- {item}" for item in summary.get("topics", []))
            lines.extend(["", "## Key Entities"])
            lines.extend(f"- {item}" for item in summary.get("key_entities", []))
            lines.extend(["", "## Important Dates"])
            lines.extend(f"- {item}" for item in summary.get("important_dates", []))
            lines.extend(["", "## Tags"])
            lines.extend(f"- #{tag}" for tag in tags)
            lines.extend(["", "## Suggested Actions"])
            lines.extend(f"- {item}" for item in summary.get("suggested_actions", []))
            lines.extend(["", "## Extracted Document Properties"])
            for key, value in document_metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.extend(["", "## Deterministic Evidence"])
            lines.append(f"- Imports: {', '.join(parsed.get('imports', [])) or 'None'}")
            lines.append(f"- Symbols: {', '.join(parsed.get('symbols', [])) or 'None'}")
            lines.extend(["", "## Related Files"])
            for relation in relationships.get(row["id"], []):
                other_path = (
                    relation["target_path"]
                    if relation["source_id"] == row["id"]
                    else relation["source_path"]
                )
                lines.append(
                    f"- [[{other_path}]] ({relation['type']}, "
                    f"{relation['confidence']:.2f}): {relation['explanation']}"
                )
        atomic_write(output, "\n".join(lines) + "\n")

    if vault.exists():
        for note in vault.rglob("*.md"):
            if note.resolve() not in active_notes:
                note.unlink()
