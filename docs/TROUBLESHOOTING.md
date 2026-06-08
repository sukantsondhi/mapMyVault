# 🧯 Troubleshooting

Use this page when indexing, OCR, YOLO, chat, or local integrations behave unexpectedly.

## 🩺 First Health Check

Run these from the activated virtual environment:

```powershell
ollama list
mapmyvault doctor --offline
python -m compileall -q src tests
```

For OCR:

```powershell
tesseract --version
pdftoppm -v
python -c "import fitz; print('PyMuPDF OK')"
```

For YOLO:

```powershell
python -c "import importlib.util; print(importlib.util.find_spec('ultralytics') is not None)"
dir models\*.pt
```

## 📁 Output Folder Error

Problem:

```text
Output must not be inside the indexed repository
```

Good:

```text
Source: C:\Users\me\Documents\source
Output: C:\Users\me\Documents\source-mmv
```

Bad:

```text
Source: C:\Users\me\Documents\source
Output: C:\Users\me\Documents\source\mapmyvault-output
```

Why: if output is inside source, the indexer would index its own generated files.

## 💬 Chat Says No Data Found

Check whether the right index is loaded:

```powershell
mapmyvault status C:\path\to\mapmyvault-output
mapmyvault query C:\path\to\mapmyvault-output "your question here"
```

Common causes:

- The wrong output folder is selected in Streamlit.
- The source folder was not indexed.
- The files were excluded in the Knowledge selection list.
- OCR was needed but not enabled.
- The question asks for information that is only visible in an image, and no OCR/vision extraction captured it.

## 🔢 Count Answers Look The Same Across Models

This is expected for count-style questions. mapMyVault uses deterministic SQLite results for counts so the model cannot invent totals.

## 🔌 Open WebUI Asks For A Valid JSON Spec

Use MCPO between Open WebUI and the mapMyVault MCP server.

Terminal 1:

```powershell
mapmyvault serve C:\path\to\mapmyvault-output
```

Terminal 2:

```powershell
python -m pip install mcpo
mcpo --port 8000 --server-type "streamable-http" -- http://127.0.0.1:8765/mcp
```

Check:

```text
http://127.0.0.1:8000/docs
```

Open WebUI connection:

```text
Type: OpenAPI
Name: mapMyVault
ID: mapmyvault
URL: http://127.0.0.1:8000
OpenAPI Spec: http://127.0.0.1:8000/openapi.json
Auth: None
Headers: {}
```

## 🌐 `GET /` Shows `404`

This is normal for the MCP server root.

Use:

```text
http://127.0.0.1:8765/mcp
```

Or use MCPO:

```text
http://127.0.0.1:8000/docs
```

## 🔤 OCR Is Unavailable

Problem examples:

```text
tesseract : The term 'tesseract' is not recognized
extraction_status: ocr_unavailable
Tesseract executable was not found in PATH
```

Fix:

```powershell
winget install UB-Mannheim.TesseractOCR
winget install oschwartz10612.Poppler
```

Restart PowerShell, activate `.venv`, then run:

```powershell
tesseract --version
pdftoppm -v
python -c "import fitz; print('PyMuPDF OK')"
```

If `tesseract --version` still fails, add the Tesseract install directory to PATH. A common Windows path is:

```text
C:\Program Files\Tesseract-OCR
```

## 🔤 OCR Ran But Found No Text

Problem:

```text
extraction_status: ocr_empty
OCR issue: No text returned by OCR
```

This is not always an error. It happens for normal photos or images without readable text.

If the file should contain text:

- Increase OCR page limit for PDFs.
- Check the language code, for example `eng`.
- Open the file manually and confirm the text is visible.
- Try rerunning after installing both Tesseract and Poppler.

## 📄 PDF Parser Warnings

Examples:

```text
Early EOD in RunLengthDecode of inline image, using fallback.
Ignoring wrong pointing object 8 0 (offset 0)
```

These usually come from malformed or unusual PDFs. They are warnings from the PDF parser and indexing can often continue.

If extracted text is missing, enable OCR and rerun.

## 🖼️ YOLO Is Unavailable

Problem:

```text
vision_status: vision_unavailable
ultralytics False
```

Fix:

```powershell
pip install -r requirements-vision.txt
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

Then confirm a local `.pt` file exists:

```powershell
dir models\*.pt
```

If not:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" `
  -OutFile "models\yolov8n.pt"
```

## 🖼️ YOLO Dropdown Is Empty

The Streamlit dropdown only reads:

```text
<repo>\models\*.pt
```

Fix:

```powershell
dir models
```

Then add a `.pt` file and refresh the Streamlit page.

## 🖼️ YOLO Says `vision_empty`

This means YOLO ran successfully but found no object class it knows about above the configured confidence threshold.

This is common for close-up detail images such as:

- damaged alloys
- scratches
- dents
- scuffs
- small surface defects
- specialist paperwork or image-quality issues

What to try:

- Lower confidence from `0.25` to `0.10`.
- Try a larger model such as `yolov8s.pt` or `yolov8m.pt`.
- Use a custom-trained YOLO model for the exact target class.
- Use a local vision-language model for descriptive image understanding.

## 🔒 Local-Only Concerns

mapMyVault rejects non-loopback Ollama URLs.

Allowed:

```text
http://127.0.0.1:11434
http://localhost:11434
```

Rejected:

```text
https://api.example.com
```

For strict local operation:

- Do not enable web search in external chat tools.
- Do not configure cloud model providers.
- Keep MCP/MCPO bound to `127.0.0.1`.
- Download model files manually, then run indexing offline.

## 🧪 Final Verification

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
git diff --check
```
