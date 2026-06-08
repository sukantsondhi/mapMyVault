# 🔌 Local Integrations

mapMyVault can be used from:

- Streamlit Studio
- CLI
- local MCP clients
- Open WebUI through MCPO
- OpenClaw or another local agent client

All integrations should stay on loopback addresses.

## 🌐 Open WebUI With MCPO

Open WebUI may ask for a valid JSON/OpenAPI spec when connecting directly to MCP. The reliable local route is:

```text
Open WebUI -> MCPO -> mapMyVault MCP -> SQLite/Chroma
```

Terminal 1:

```powershell
mapmyvault serve C:\path\to\mapmyvault-output
```

Terminal 2:

```powershell
python -m pip install mcpo
mcpo --port 8000 --server-type "streamable-http" -- http://127.0.0.1:8765/mcp
```

Check MCPO:

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

## 💬 Slash Command Prompt

Use a prompt like this for a mapMyVault slash command:

```text
You are using a local tool-backed database called mapMyVault.

Question:
{{question | textarea:required}}

Rules:
- You MUST call ask_mapmyvault with the user's exact question before answering.
- Use the answer field returned by ask_mapmyvault as the main answer.
- Cite local file paths from paths or results when available.
- For count questions, use found_count and paths from the tool result.
- Do not invent paths, counts, files, shell commands, or tool outputs.
- Do not say you searched mapMyVault unless a tool result was actually returned.
- If found_count is 0, say no relevant indexed data was found in the selected local index.
- Answer only from local mapMyVault evidence.
- If the user asks for something outside the local index, say it is outside the indexed data.
```

If Open WebUI shows the whole slash prompt in the text box, that is Open WebUI behavior. The prompt can still work, but the model must actually call the tool. If it writes fake tool output, tighten the prompt and use a model that reliably supports tools.

## 🧰 MCP Tools

| Tool                   | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `ask_mapmyvault`       | Best all-purpose grounded answer tool                |
| `answer_question`      | Compact evidence for agent reasoning                 |
| `count_matches`        | Exact local count matching meaningful terms          |
| `search_files`         | Search paths, extracted text, summaries, and vectors |
| `get_file_summary`     | Structured summary and metadata                      |
| `get_file_details`     | Alias for detailed file metadata                     |
| `get_related_files`    | Relationship list for one file                       |
| `list_files_by_format` | List PDFs, DOCX, XLSX, images, etc.                  |
| `explain_relationship` | Explain link between two files                       |
| `get_folder_children`  | Navigate indexed folders                             |
| `read_file_excerpt`    | Bounded extracted text excerpt                       |
| `propose_move_files`   | Create action plan only; no direct apply             |

## ⌨️ CLI Query

```powershell
mapmyvault query C:\path\to\mapmyvault-output "how many documents mention [Specific thing]?"
```

This reads the local SQLite/Chroma index. It does not search the web.

## 🔌 Local Server Ports

| Service          | Default                     |
| ---------------- | --------------------------- |
| Ollama           | `http://127.0.0.1:11434`    |
| mapMyVault MCP   | `http://127.0.0.1:8765/mcp` |
| MCPO             | `http://127.0.0.1:8000`     |
| Streamlit Studio | `http://127.0.0.1:8788`     |

Do not expose these ports publicly.

## 🔒 Keep External Tools Local

For Open WebUI, OpenClaw, or similar tools:

- Disable web search.
- Disable remote/cloud model providers if strict privacy is required.
- Use local Ollama models.
- Keep mapMyVault and MCPO URLs on `127.0.0.1`.
- Do not upload `data/index.sqlite`, `data/chroma/`, or source files to cloud tools.
