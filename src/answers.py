"""Deterministic answer formatting for grounded local evidence."""


def deterministic_answer(evidence: dict) -> str:
    found_count = evidence.get("found_count", 0)
    question = evidence.get("question", "this question")
    paths = evidence.get("paths") or []
    if not paths:
        paths = [item.get("path") for item in evidence.get("results", []) if item.get("path")]
    noun = "matching file" if found_count == 1 else "matching files"
    lines = [
        f"mapMyVault found {found_count} {noun} for: {question}",
    ]
    if paths:
        lines.extend(["", "Sources:"])
        lines.extend(f"- `{path}`" for path in paths[:50])
    return "\n".join(lines)
