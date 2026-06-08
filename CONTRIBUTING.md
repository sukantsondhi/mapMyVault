# Contributing

For setup and local integration notes, see [Setup](docs/SETUP.md) and
[Integrations](docs/INTEGRATIONS.md).

## Development Setup

```powershell
git clone https://github.com/sukantsondhi/mapMyVault.git
cd mapMyVault
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
python -m src.cli doctor --offline
```

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```

Changes to commands, schemas, stage invalidation, privacy boundaries, MCP tools,
or action safety must include focused tests and corresponding documentation.

## Design Rules

- Keep SQLite canonical and derived artifacts rebuildable.
- Do not add remote model providers, telemetry, or non-loopback services.
- Relationships must retain evidence and must not rely only on filenames.
- MCP tools may propose filesystem actions but must not approve or apply them.
- Preserve resumability: completed valid stages should not rerun unnecessarily.
