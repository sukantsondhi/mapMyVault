"""Export derived artifacts from an existing mapMyVault index."""

from pathlib import Path
from typing import Dict, Optional
import json

from .config import MapperConfig
from .storage import IndexStore
from .vault_export import export_vault


def export_obsidian_from_index(
    output: Path,
    vault_dir: Optional[Path] = None,
    config: Optional[MapperConfig] = None,
) -> Dict:
    output = output.resolve()
    database = output / "data" / "index.sqlite"
    if not database.exists():
        raise ValueError(f"mapMyVault index not found: {database}")
    vault_dir = (vault_dir or output / "obsidian").resolve()
    store = IndexStore(database)
    config = config or _config_from_manifest(output)
    try:
        files = [dict(row) for row in store.files()]
        relationships = {
            row["id"]: [dict(item) for item in store.relationships_for(row["id"])]
            for row in store.files()
        }
        export_vault(vault_dir, files, relationships)
        for row in store.files():
            store.stage_complete(row["id"], "export", config.export_template_version)
        store.connection.commit()
        return {
            "vault": str(vault_dir),
            "files": len([row for row in files if row["kind"] == "file"]),
            "folders": len([row for row in files if row["kind"] == "folder"]),
        }
    finally:
        store.close()


def _config_from_manifest(output: Path) -> MapperConfig:
    manifest_path = output / "data" / "manifest.json"
    if not manifest_path.exists():
        return MapperConfig()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return MapperConfig(**manifest.get("config", {}))
