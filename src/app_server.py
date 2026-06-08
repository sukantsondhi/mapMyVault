"""Local-only browser app for mapMyVault."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import parse_qs, urlparse
import json
import threading
import uuid
import webbrowser

from .config import LOCAL_HOSTS, MapperConfig
from .mapper import RepositoryMapper
from .query import VaultIndex


APP_HOSTS = LOCAL_HOSTS


def require_loopback_host(host: str) -> str:
    if host not in APP_HOSTS:
        raise ValueError("mapMyVault app must bind to localhost, 127.0.0.1, or ::1")
    return host


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: Dict[str, Dict] = {}

    def create_job(self, source: Path, output: Path, config: MapperConfig) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self.jobs[job_id] = {
                "id": job_id,
                "state": "queued",
                "source": str(source),
                "output": str(output),
                "config": config.to_dict(),
                "result": None,
                "error": None,
            }
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, source, output, config),
            daemon=True,
        )
        thread.start()
        return job_id

    def snapshot(self) -> Dict:
        with self._lock:
            return {"jobs": list(self.jobs.values())}

    def get_job(self, job_id: str) -> Dict:
        with self._lock:
            return dict(self.jobs.get(job_id, {}))

    def _set_job(self, job_id: str, **updates) -> None:
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)

    def _run_job(
        self, job_id: str, source: Path, output: Path, config: MapperConfig
    ) -> None:
        self._set_job(job_id, state="running")
        mapper = None
        try:
            mapper = RepositoryMapper(source, output, config, show_progress=False)
            result = mapper.map()
            self._set_job(job_id, state="done", result=result)
        except Exception as exc:
            self._set_job(job_id, state="failed", error=str(exc))
        finally:
            if mapper:
                mapper.close()


def _html() -> bytes:
    return HTML.encode("utf-8")


def _json(data: Dict, status: int = 200) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def _read_body(handler: BaseHTTPRequestHandler) -> Dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    body = handler.rfile.read(length).decode("utf-8")
    return json.loads(body or "{}")


def require_index(output: Path) -> Path:
    database = output / "data" / "index.sqlite"
    if not database.exists():
        raise ValueError(f"mapMyVault index not found: {database}")
    return output


def create_handler(state: AppState):
    class LocalAppHandler(BaseHTTPRequestHandler):
        server_version = "mapMyVaultLocalApp/1.0"

        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(_html(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send_json(state.snapshot())
                return
            if parsed.path == "/api/job":
                job_id = parse_qs(parsed.query).get("id", [""])[0]
                job = state.get_job(job_id)
                self._send_json(job if job else {"error": "Job not found"}, 200 if job else 404)
                return
            if parsed.path == "/api/status":
                output = parse_qs(parsed.query).get("output", [""])[0]
                self._status(output)
                return
            self._send_json({"error": "Not found"}, 404)

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/api/map":
                    self._map()
                    return
                if self.path == "/api/chat":
                    self._chat()
                    return
                self._send_json({"error": "Not found"}, 404)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body"}, 400)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)

        def _map(self) -> None:
            payload = _read_body(self)
            source_raw = payload.get("source", "")
            output_raw = payload.get("output", "")
            if not source_raw:
                raise ValueError("Source folder is required")
            if not output_raw:
                raise ValueError("Output folder is required")
            source = Path(source_raw).expanduser().resolve()
            output = Path(output_raw).expanduser().resolve()
            if not source.is_dir():
                raise ValueError(f"Source folder not found: {source}")
            config = MapperConfig(
                generation_model=payload.get("generation_model") or "llama3.1:8b",
                embedding_model=payload.get("embedding_model")
                or "nomic-embed-text:latest",
                enable_ocr=bool(payload.get("enable_ocr", False)),
                ocr_language=payload.get("ocr_language") or "eng",
                ocr_max_pages=int(payload.get("ocr_max_pages") or 10),
            )
            job_id = state.create_job(source, output, config)
            self._send_json({"job_id": job_id, "state": "queued"})

        def _chat(self) -> None:
            payload = _read_body(self)
            output_raw = payload.get("output", "")
            if not output_raw:
                raise ValueError("Output folder is required")
            output = require_index(Path(output_raw).expanduser().resolve())
            question = payload.get("question", "")
            index = VaultIndex(output)
            try:
                answer = index.ask_mapmyvault(question, int(payload.get("limit") or 20))
            finally:
                index.close()
            self._send_json(answer)

        def _status(self, output: str) -> None:
            if not output:
                self._send_json({"error": "Output folder is required"}, 400)
                return
            index = VaultIndex(require_index(Path(output).expanduser().resolve()))
            try:
                self._send_json(index.status())
            finally:
                index.close()

        def _send_json(self, data: Dict, status: int = 200) -> None:
            self._send(_json(data), "application/json; charset=utf-8", status)

        def _send(
            self, body: bytes, content_type: str, status: int = 200
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return LocalAppHandler


def serve_app(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True):
    host = require_loopback_host(host)
    state = AppState()
    server = ThreadingHTTPServer((host, port), create_handler(state))
    url = f"http://{host}:{port}"
    print(f"mapMyVault local app running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mapMyVault Local App</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f6f7fb; color: #1f2937; }
    header { background: #111827; color: white; padding: 18px 24px; }
    main { display: grid; grid-template-columns: 380px 1fr; gap: 16px; padding: 16px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; }
    label { display: block; font-weight: 600; margin-top: 12px; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; }
    textarea { min-height: 90px; }
    button { margin-top: 12px; padding: 10px 14px; border: 0; border-radius: 8px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button.secondary { background: #374151; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 10px; min-height: 140px; overflow: auto; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .hint { color: #6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <h1>mapMyVault Local App</h1>
    <div>Local indexing, OCR, Obsidian export, and grounded chat on this machine.</div>
  </header>
  <main>
    <section>
      <h2>Knowledge Base</h2>
      <p class="hint">Use real local paths. Browser folder drag-and-drop does not expose the original folder path; a future Tauri/Electron wrapper can add native drag-and-drop.</p>
      <label>Source folder</label>
      <input id="source" placeholder="C:\\path\\to\\source">
      <label>mapMyVault output folder</label>
      <input id="output" placeholder="C:\\path\\to\\mapmyvault-output">
      <div class="row">
        <div>
          <label>Chat model</label>
          <input id="generationModel" value="llama3.1:8b">
        </div>
        <div>
          <label>Embedding model</label>
          <input id="embeddingModel" value="nomic-embed-text:latest">
        </div>
      </div>
      <div class="row">
        <div>
          <label>OCR language</label>
          <input id="ocrLanguage" value="eng">
        </div>
        <div>
          <label>OCR max pages</label>
          <input id="ocrMaxPages" type="number" value="10">
        </div>
      </div>
      <label><input id="enableOcr" type="checkbox" style="width:auto"> Enable local OCR</label>
      <button onclick="startMap()">Start or Update Index</button>
      <button class="secondary" onclick="refreshStatus()">Refresh Status</button>
      <h3>Job / Status</h3>
      <pre id="status">Ready.</pre>
    </section>
    <section>
      <h2>Chat With Local Data</h2>
      <label>Question</label>
      <textarea id="question" placeholder="Ask using only the indexed local database"></textarea>
      <button onclick="ask()">Ask mapMyVault</button>
      <h3>Answer</h3>
      <pre id="answer">No question asked yet.</pre>
    </section>
  </main>
  <script>
    let activeJob = null;
    const value = id => document.getElementById(id).value.trim();
    const checked = id => document.getElementById(id).checked;
    const show = (id, data) => document.getElementById(id).textContent =
      typeof data === "string" ? data : JSON.stringify(data, null, 2);

    async function post(url, body) {
      const res = await fetch(url, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function startMap() {
      try {
        const data = await post("/api/map", {
          source: value("source"),
          output: value("output"),
          generation_model: value("generationModel"),
          embedding_model: value("embeddingModel"),
          enable_ocr: checked("enableOcr"),
          ocr_language: value("ocrLanguage"),
          ocr_max_pages: value("ocrMaxPages")
        });
        activeJob = data.job_id;
        show("status", data);
        pollJob();
      } catch (err) { show("status", err.message); }
    }

    async function pollJob() {
      if (!activeJob) return;
      const res = await fetch("/api/job?id=" + encodeURIComponent(activeJob));
      const data = await res.json();
      show("status", data);
      if (data.state === "queued" || data.state === "running") setTimeout(pollJob, 1500);
    }

    async function refreshStatus() {
      try {
        const res = await fetch("/api/status?output=" + encodeURIComponent(value("output")));
        show("status", await res.json());
      } catch (err) { show("status", err.message); }
    }

    async function ask() {
      try {
        const data = await post("/api/chat", { output: value("output"), question: value("question"), limit: 20 });
        show("answer", data);
      } catch (err) { show("answer", err.message); }
    }
  </script>
</body>
</html>
"""
