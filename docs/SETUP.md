# 🧰 Build From Source

This guide takes a fresh clone to a working local mapMyVault install.

The examples use Windows PowerShell because that is where most setup problems have appeared. macOS/Linux commands are included where they differ.

## ✅ Requirements

Required:

- Python 3.10 or newer
- Git
- Ollama installed and running locally
- At least one Ollama chat model
- At least one Ollama embedding model

Optional:

- Tesseract OCR for scanned PDFs/images
- Poppler for PDF-to-image conversion
- YOLO `.pt` weights plus `ultralytics` for image object labels

## 🧬 1. Clone And Create The Virtual Environment

```powershell
git clone https://github.com/sukantsondhi/mapMyVault.git
cd mapMyVault
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
git clone https://github.com/sukantsondhi/mapMyVault.git
cd mapMyVault
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 🤖 2. Install Local Ollama Models

Start Ollama, then pull the default models:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama list
```

Optional stronger embedding model:

```powershell
ollama pull mxbai-embed-large
```

Run the local readiness check:

```powershell
mapmyvault doctor --offline
```

Expected core checks:

```text
local_endpoint: true
ollama_reachable: true
generation_model: true
embedding_model: true
ollama_no_cloud: true
```

If a model check is false, pull that model or select a model you already have in the UI.

## 🖥️ 3. Run Streamlit Studio

```powershell
mapmyvault studio
```

Open:

```text
http://127.0.0.1:8788
```

Use the sidebar:

- `Chat`: ask questions against an existing local index.
- `Knowledge`: create or update the local index.
- `Graph View`: browse indexed folders, remove indexed folders/files, and update only changed work.

## 🗂️ 4. Create Your First Local Index

In `Knowledge`:

1. Choose your source folder.
2. Choose an output folder outside the source folder.
3. Select which top-level folders/files should be indexed.
4. Select the embedding model.
5. Enable OCR only if you need scanned PDF/image text extraction.
6. Enable YOLO only if you have image files and have completed the YOLO setup below.
7. Click `Generate / Update Local LLM Index`.

Good output path example:

```text
Source: C:\Users\me\Documents\MyFolder
Output: C:\Users\me\Documents\MyFolder-mmv
```

Bad output path example:

```text
Source: C:\Users\me\Documents\MyFolder
Output: C:\Users\me\Documents\MyFolder\mapmyvault-output
```

The output must not be inside the source folder.

## 🔤 5. Optional OCR Setup

OCR is needed when PDFs/images contain text that is not extractable normally.

Install OCR tools on Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
winget install oschwartz10612.Poppler
```

Restart PowerShell after installation, reactivate the virtual environment, then verify:

```powershell
tesseract --version
pdftoppm -v
python -c "import fitz; print('PyMuPDF OK')"
```

Important:

- Tesseract reads text.
- Poppler helps convert PDF pages into images.
- PyMuPDF is a fallback PDF renderer.
- If Tesseract is not on `PATH`, OCR cannot run.

CLI example:

```powershell
mapmyvault map C:\path\to\source --output C:\path\to\output `
  --enable-ocr `
  --ocr-language eng `
  --ocr-max-pages 10
```

## 🖼️ 6. Optional YOLO Image Vision Setup

YOLO adds object labels to normal image files. It is optional and local.

Install the optional package:

```powershell
pip install -r requirements-vision.txt
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

Download a model into the repo-local `models` folder:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" `
  -OutFile "models\yolov8n.pt"
```

Verify:

```powershell
dir models\*.pt
```

In Streamlit Studio:

1. Open `Knowledge`.
2. Enable `Enable local YOLO object labels for image files`.
3. Pick `yolov8n.pt` from the dropdown.

CLI example:

```powershell
mapmyvault map C:\path\to\source --output C:\path\to\output `
  --enable-vision `
  --vision-model yolov8n.pt
```

Notes:

- `.pt` files in `models/` are ignored by Git.
- mapMyVault does not auto-download YOLO weights during indexing.
- `yolov8n.pt` is a general object detector. It may not detect specialist details such as scratches, scuffs, dents, or alloy wheel damage.

## 📝 7. Optional Obsidian Export

Index first. Export Obsidian after the index is ready.

```powershell
mapmyvault export-vault C:\path\to\mapmyvault-output
```

Obsidian notes are written to:

```text
C:\path\to\mapmyvault-output\obsidian
```

## ⌨️ 8. CLI-Only Workflow

```powershell
mapmyvault map C:\path\to\source --output C:\path\to\mapmyvault-output
mapmyvault status C:\path\to\mapmyvault-output
mapmyvault query C:\path\to\mapmyvault-output "what documents mention rent?"
mapmyvault export-vault C:\path\to\mapmyvault-output
```

## ✅ 9. Verify The Repo

Run before opening a PR:

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
git diff --check
```

For Streamlit UI changes:

```powershell
mapmyvault studio
```
