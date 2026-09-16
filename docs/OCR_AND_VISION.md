# 🔤 OCR And 🖼️ Local Vision

mapMyVault has two separate image/document helpers:

| Feature | Purpose                                    | Best For                                        |
| ------- | ------------------------------------------ | ----------------------------------------------- |
| OCR     | Extract text from scanned PDFs and images  | PDFs, letters, forms, screenshots with text     |
| YOLO    | Detect known object classes in image files | Broad labels such as person, car, laptop, chair |

They solve different problems. OCR reads text. YOLO detects trained object classes.

## 🔤 OCR Pipeline

OCR runs only when enabled.

For PDFs:

```text
PDF -> pypdf text per page -> pages without text or with images -> render selected pages -> Tesseract OCR
                         -> merge native and recognized text in page order
```

With OCR enabled, mixed PDFs retain native text and recognize scanned pages,
including images containing text on a page that also has native text. OCR renders
the entire candidate page. Recognized lines matching native lines after whitespace
and case normalization are omitted; other recognized lines are appended to that
page's native text. This does not reconstruct the original page layout. Native
text is retained if OCR fails, with the failure recorded in `ocr_status`.

`ocr_max_pages` defaults to 10 and applies to the first 10 pages of the original
PDF, not the first 10 OCR candidates. Set it to 0 to remove the page limit. Native
text outside that range is still extracted until the content character limit.
`ocr_candidate_pages` lists detected candidates using one-based page numbers;
`ocr_skipped_pages` lists candidates excluded by the page limit. Candidate discovery
also stops when the native-text character budget is reached. `ocr_pages` counts
pages actually processed by Tesseract, which can be fewer than rendered pages when
the character limit is reached. Files over the configured byte limit are not read.

Rerunning an existing index refreshes pre-fix extraction caches so mixed PDFs are
not left with their old native-text-only results.

For image files:

```text
Image -> Tesseract OCR
```

Stored metadata examples:

```json
{
  "format": "pdf",
  "extraction_status": "ocr_extracted",
  "ocr_engine": "tesseract",
  "ocr_language": "eng",
  "ocr_pdf_renderer": "pdf2image",
  "ocr_pages": 8,
  "extracted_characters": 4200
}
```

Common OCR statuses:

| Status            | Meaning                                           | What To Do                                     |
| ----------------- | ------------------------------------------------- | ---------------------------------------------- |
| `extracted`       | Normal text extraction worked                     | Nothing                                        |
| `ocr_required`    | PDF/image probably needs OCR but OCR was disabled | Enable OCR and rerun                           |
| `ocr_extracted`   | OCR worked and produced text                      | Nothing                                        |
| `ocr_empty`       | OCR ran but found no text                         | Usually fine for photos                        |
| `ocr_skipped`     | All detected PDF candidates exceed the page limit | Raise `ocr_max_pages` if needed                |
| `ocr_failed`      | OCR ran but failed on that file                   | Check file and logs                            |
| `ocr_unavailable` | OCR tool is missing                               | Install Tesseract/Poppler and restart terminal |

## ✅ OCR Setup Checklist

```powershell
winget install UB-Mannheim.TesseractOCR
winget install oschwartz10612.Poppler
```

Restart PowerShell, reactivate `.venv`, then verify:

```powershell
tesseract --version
pdftoppm -v
python -c "import fitz; print('PyMuPDF OK')"
```

If `tesseract --version` fails, OCR cannot run. Add Tesseract to PATH or reinstall it.

For an existing installation in the usual Windows location, this enables it only
for the current PowerShell session and programs started from that session:

```powershell
$env:Path = "C:\Program Files\Tesseract-OCR;" + $env:Path
tesseract --list-langs
```

Start or restart Studio from that terminal so it inherits the updated PATH. The
English recognition tests require `eng` in the installed language list.

## Recognition Tests

With the project dependencies installed and the virtual environment activated:

```powershell
python -m unittest tests.test_mapper.TextRecognitionTests tests.test_mapper.LiveTextRecognitionTests -v
```

The tests generate temporary documents locally; no user files, downloads, model
services, or external OCR providers are used. Live tests call the real Tesseract
engine and skip explicitly if it is absent from PATH. A run with skipped live
tests is not evidence that recognition works.

Coverage includes UTF-8 text with a BOM, native PDFs without OCR, PNG/JPEG/BMP/WebP
and single-frame TIFF images, scanned PDFs, mixed native/scanned pages in both
orders, and native text plus an image on the same page. Tests also cover the
PyMuPDF renderer fallback, page/character limits, blank and malformed images,
missing engine/language errors, cache invalidation, and SQLite search persistence.
Only LLM summarization and embeddings are faked in the indexing test.

Recognition assertions use clear printed English text. Handwriting, rotated or
noisy scans, multi-frame TIFFs, and non-English recognition quality are not covered.
Embedded-image OCR inside DOCX, XLSX, or PPTX is not implemented.

## 🖼️ YOLO Pipeline

YOLO runs only when enabled and only for image files:

```text
Image -> local YOLO .pt model -> labels/counts/detections -> SQLite metadata + extracted text
```

Stored metadata examples:

```json
{
  "format": "jpg",
  "extraction_status": "vision_extracted",
  "vision_engine": "yolo",
  "vision_model": "C:\\Users\\me\\GitHub\\mapMyVault\\models\\yolov8n.pt",
  "vision_status": "vision_extracted",
  "vision_label_counts": {
    "car": 1
  }
}
```

Common YOLO statuses:

| Status               | Meaning                                                        | What To Do                             |
| -------------------- | -------------------------------------------------------------- | -------------------------------------- |
| `vision_extracted`   | YOLO detected at least one known class                         | Nothing                                |
| `vision_empty`       | YOLO ran but found no trained class above confidence threshold | Not broken; see below                  |
| `vision_unavailable` | Package or `.pt` weights are missing                           | Install dependency or add model file   |
| `vision_failed`      | YOLO runtime failed                                            | Check logs and test the model manually |

## ✅ YOLO Setup Checklist

Install optional dependency:

```powershell
pip install -r requirements-vision.txt
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

Download a model into the repo `models` folder:

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" `
  -OutFile "models\yolov8n.pt"
```

Verify:

```powershell
dir models\*.pt
```

In Streamlit Studio, the YOLO dropdown is populated from:

```text
<repo>\models\*.pt
```

If the dropdown is empty, the `.pt` file is missing or the app needs a refresh.

## ❔ Why `vision_empty` Happens

`vision_empty` means YOLO ran successfully but did not detect any object class it knows about.

This is common with close-up/detail photos. For example, the default `yolov8n.pt` model is not trained to identify:

- alloy wheel damage
- scratches
- scuffs
- dents
- small surface defects
- document quality problems

Options:

1. Lower the confidence threshold from `0.25` to `0.10`.
2. Try a larger general model such as `yolov8s.pt` or `yolov8m.pt`.
3. Use a custom YOLO model trained for your target defect classes.
4. Add a local vision-language captioning step for richer image descriptions.

## 🧭 OCR vs YOLO Decision Guide

Use OCR when the user will ask:

```text
What address is on this PDF?
Which files mention John?
What date is on this scanned letter?
```

Use YOLO when the user will ask:

```text
Which images contain cars?
Do any photos include people?
Which image folders contain laptops or documents?
```

Use a local vision-language model when the user will ask:

```text
Which alloy photos show damage?
What is happening in this image?
Describe this uploaded photo.
```

YOLO is object detection. It is not a general image understanding model.

## 🔒 Local-Only Guarantee

mapMyVault does not auto-upload images, PDFs, OCR text, or YOLO outputs.

Important caveat: if you manually download a `.pt` model using `Invoke-WebRequest`, that download is a network action you started. After the file is local, indexing uses the local file.
