"""Loopback-only MCP server for local query and action-plan tools."""

from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from .actions import ActionManager
from .query import VaultIndex


def create_server(output: Path, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MCP server must bind to a loopback host")
    index = VaultIndex(output)
    actions = ActionManager(output)
    server = FastMCP(
        "mapMyVault",
        instructions="Query a fully local repository index. File changes require CLI approval.",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    def search_files(query: str = "", limit: int = 10) -> List[Dict]:
        """Search indexed paths, content, and summaries."""
        return index.search_files(query, limit)

    @server.tool()
    def answer_question(question: str = "", limit: int = 20) -> Dict:
        """Return compact local evidence for answering a user question."""
        return index.answer_question(question, limit)

    @server.tool()
    def ask_mapmyvault(question: str = "", limit: int = 20) -> Dict:
        """Return a grounded final answer and local evidence."""
        return index.ask_mapmyvault(question, limit)

    @server.tool()
    def count_matches(query: str = "", limit: int = 100) -> Dict:
        """Count indexed files matching all meaningful query terms."""
        return index.count_matches(query, limit)

    @server.tool()
    def get_file_summary(path: str = "") -> Dict:
        """Return a structured summary and deterministic evidence."""
        return index.get_file_summary(path)

    @server.tool()
    def get_file_details(path: str = "") -> Dict:
        """Return document properties, detailed explanation, and parsed evidence."""
        return index.get_file_summary(path)

    @server.tool()
    def get_related_files(path: str = "") -> List[Dict]:
        """Return evidence-backed relationships for a file."""
        return index.get_related_files(path)

    @server.tool()
    def list_files_by_format(file_format: str = "") -> List[Dict]:
        """List indexed documents by format, such as pdf, docx, xlsx, or pptx."""
        return index.list_files_by_format(file_format)

    @server.tool()
    def explain_relationship(source: str = "", target: str = "") -> List[Dict]:
        """Explain indexed relationships between two files."""
        return [
            relation
            for relation in index.get_related_files(source)
            if target in {relation["source_path"], relation["target_path"]}
        ]

    @server.tool()
    def get_folder_children(path: str = "") -> List[Dict]:
        """List direct children of an indexed folder."""
        return index.get_folder_children(path)

    @server.tool()
    def read_file_excerpt(path: str, max_chars: int = 2000) -> str:
        """Read a bounded excerpt from the locally persisted extraction."""
        return index.read_file_excerpt(path, max_chars)

    @server.tool()
    def propose_move_files(operations: List[Dict[str, str]] = None) -> Dict:
        """Create a validated move plan. Approval and execution are CLI-only."""
        return actions.propose_moves(operations or [])

    return server


def serve(output: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    create_server(output, host, port).run(transport="streamable-http")
