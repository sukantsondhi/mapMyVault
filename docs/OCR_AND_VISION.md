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
PDF -> pypdf text extraction -> if no text and OCR enabled -> render pages -> Tesseract OCR
```

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
