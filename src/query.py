"""Query the persisted local index."""

from pathlib import Path
from functools import lru_cache
from typing import Dict, List
import json
import re

from .local_ai import LocalAI
from .storage import IndexStore
from .vector_index import VectorIndex


SNIPPET_CHARS = 500
COUNT_WORDS = {"count", "how many", "number of", "total"}
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "documents",
    "file",
    "files",
    "find",
    "for",
    "give",
    "got",
    "get",
    "had",
    "has",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "me",
    "mention",
    "mentions",
    "named",
    "need",
    "of",
    "on",
    "or",
    "person",
    "please",
    "show",
    "should",
    "that",
    "the",
    "there",
    "these",
    "this",
    "to",
    "use",
    "using",
    "want",
    "what",
    "where",
    "which",
    "who",
    "would",
    "with",
}

ALIASES = {
    "uni": ["uni", "university"],
    "universities": ["university", "universities"],
    "offer": ["offer", "offers", "offered"],
    "offers": ["offer", "offers", "offered"],
    "offered": ["offer", "offers", "offered"],
    "undergrad": ["undergrad", "undergraduate"],
    "undergraduate": ["undergrad", "undergraduate"],
    "degree": ["degree", "degrees", "bsc", "ba", "bachelor"],
    "degrees": ["degree", "degrees", "bsc", "ba", "bachelor"],
}


@lru_cache(maxsize=1)
def _stopwords() -> set:
    words = set(STOPWORDS)
    try:
        from nltk.corpus import stopwords

        words.update(stopwords.words("english"))
    except Exception:
        # NLTK is optional at runtime; no corpus download is attempted.
        pass
    return words


@lru_cache(maxsize=512)
def _yake_keywords(query: str) -> tuple:
    try:
        import yake

        extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.9,
            top=8,
            features=None,
        )
        keywords = []
        for phrase, _score in extractor.extract_keywords(query):
            tokens = [
                token
                for token in re.findall(r"[\w']+", phrase.lower())
                if token not in _stopwords()
            ]
            normalized = " ".join(tokens)
            if normalized and normalized not in keywords:
                keywords.append(normalized)
        return tuple(keywords)
    except Exception:
        return tuple()


def _terms(query: str) -> List[str]:
    terms = []
    stopwords = _stopwords()
    for token in re.findall(r"[\w']+", query.lower()):
        if len(token) > 1 and token not in stopwords:
            terms.append(token)
    return terms


def _term_groups(query: str) -> List[List[str]]:
    groups = []
    terms = _terms(query)
    for phrase in _yake_keywords(query):
        terms.extend(token for token in re.findall(r"[\w']+", phrase) if len(token) > 1)
    deduped_terms = []
    for term in terms:
        if term not in deduped_terms and term not in _stopwords():
            deduped_terms.append(term)
    for term in deduped_terms:
        aliases = ALIASES.get(term)
        if aliases:
            group = aliases
        elif term.endswith("s") and len(term) > 3:
            group = [term, term[:-1]]
        else:
            group = [term]
        if group not in groups:
            groups.append(group)
    return groups


def _query_analysis(query: str) -> Dict:
    terms = _terms(query)
    groups = _term_groups(query)
    keyword_phrases = list(_yake_keywords(query))
    expanded = []
    for group in groups:
        for term in group:
            if term not in expanded:
                expanded.append(term)
    return {
        "original": query,
        "keywords": terms,
        "keyword_phrases": keyword_phrases,
        "expanded_keywords": expanded,
        "term_groups": groups,
        "keyword_query": " ".join(keyword_phrases + terms),
    }


def _contains_term(haystack: str, term: str) -> bool:
    if len(term) <= 3:
        return re.search(rf"\b{re.escape(term)}\b", haystack) is not None
    return term in haystack


def _matches_group(haystack: str, group: List[str]) -> bool:
    return any(_contains_term(haystack, term) for term in group)


def _group_match_count(haystack: str, groups: List[List[str]]) -> int:
    return sum(1 for group in groups if _matches_group(haystack, group))


def _haystack(row) -> str:
    return " ".join(
        [
            row["path"] or "",
            row["extracted_text"] or "",
            row["summary_json"] or "",
        ]
    ).lower()


def _is_count_question(question: str) -> bool:
    lowered = question.lower()
    return any(word in lowered for word in COUNT_WORDS)


class VaultIndex:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.store = IndexStore(self.output / "data" / "index.sqlite")
        chroma_path = self.output / "data" / "chroma"
        self.vector = VectorIndex(chroma_path) if chroma_path.exists() else None
        self.ai = None
        manifest_path = self.output / "data" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.ai = LocalAI(
                manifest["endpoints"]["ollama"],
                manifest["models"]["generation"],
                manifest["models"]["embedding"],
            )

    def close(self) -> None:
        self.store.close()

    def status(self) -> Dict:
        return self.store.status()

    def search_files(self, query: str, limit: int = 10) -> List[Dict]:
        if not query.strip():
            return []
        results = {}
        analysis = _query_analysis(query)
        terms = analysis["keywords"]
        groups = analysis["term_groups"]
        fts_query = analysis["keyword_query"] or query
        for row in self.store.search(fts_query, max(limit * 3, 10)):
            results[row["id"]] = self._result(row, "full_text", query)
        if groups:
            minimum_matches = max(1, min(len(groups), round(len(groups) * 0.6)))
            for row in self.store.files("file"):
                if row["id"] in results:
                    continue
                haystack = _haystack(row)
                score = _group_match_count(haystack, groups)
                if score >= minimum_matches:
                    result = self._result(row, "keyword", query)
                    result["keyword_score"] = score
                    result["keyword_total"] = len(groups)
                    results[row["id"]] = result
        if self.ai and self.vector:
            try:
                embedding = self.ai.embed([query])[0]
                for candidate in self.vector.candidates(embedding, max(limit * 2, 10)):
                    row = self.store.get_file(candidate["id"])
                    if row and not row["deleted"] and candidate["id"] not in results:
                        result = self._result(row, "semantic", query)
                        result["distance"] = candidate["distance"]
                        results[candidate["id"]] = result
            except Exception:
                # SQLite FTS remains available if a derived vector artifact is absent.
                pass
        ranked = sorted(
            results.values(),
            key=lambda item: (
                -item.get("keyword_score", 0),
                -self._match_score(item, terms, path_only=True),
                -self._match_score(item, terms),
                item["document_metadata"].get("extracted_characters", 0) <= 0,
                item["match_type"] != "full_text",
                item["path"].lower(),
            ),
        )
        return ranked[:limit]

    def answer_question(self, question: str = "", limit: int = 20) -> Dict:
        """Return compact evidence for an agent to answer from local data."""
        if not question.strip():
            return {
                "question": question,
                "found_count": 0,
                "results": [],
                "instruction": "A question is required. Ask the user for a question.",
            }
        count_result = self.count_matches(question, limit=max(limit, 50))
        if _is_count_question(question) and count_result["paths"]:
            matches = [
                self.get_file_summary(path)
                for path in count_result["paths"][:limit]
            ]
            for item in matches:
                item["match_type"] = "count_match"
        else:
            matches = self.search_files(question, limit)
        if count_result["count"] == 0 and matches:
            count_result = {
                **count_result,
                "count": len(matches),
                "paths": [item["path"] for item in matches],
                "fallback": "search_results",
            }
        compact = [
            {
                "path": item["path"],
                "match_type": item["match_type"],
                "format": item["document_metadata"].get("format"),
                "extraction_status": item["document_metadata"].get("extraction_status"),
                "title": item["summary"].get("title", ""),
                "purpose": item["summary"].get("purpose", ""),
                "topics": item["summary"].get("topics", []),
                "key_entities": item["summary"].get("key_entities", []),
                "important_dates": item["summary"].get("important_dates", []),
                "evidence_excerpt": item.get("excerpt", ""),
            }
            for item in matches
        ]
        return {
            "question": question,
            "query_analysis": _query_analysis(question),
            "found_count": count_result["count"],
            "returned_count": len(compact),
            "results": compact,
            "all_matching_paths": count_result["paths"],
            "instruction": (
                "Answer only from these local mapMyVault results. If found_count is 0, "
                "say no relevant indexed data was found. For count questions, use "
                "found_count as the answer and cite the matching paths. Use "
                "query_analysis keywords for retrieval meaning, but answer the original "
                "question in normal English."
            ),
        }

    def ask_mapmyvault(self, question: str = "", limit: int = 20) -> Dict:
        """Return a grounded final answer plus evidence from the local index."""
        if not question.strip():
            return {
                "question": question,
                "answer": "A question is required.",
                "found_count": 0,
                "paths": [],
                "tool": "ask_mapmyvault",
            }
        evidence = self.answer_question(question, limit)
        paths = evidence.get("all_matching_paths") or [
            item["path"] for item in evidence.get("results", [])
        ]
        found_count = evidence.get("found_count", len(paths))
        if found_count == 0:
            return {
                "question": question,
                "answer": "No relevant indexed data was found in the local mapMyVault database.",
                "found_count": 0,
                "paths": [],
                "tool": "ask_mapmyvault",
            }
        if _is_count_question(question):
            answer = (
                f"The local mapMyVault index contains {found_count} matching file"
                f"{'' if found_count == 1 else 's'} for this question."
            )
        else:
            answer = (
                f"The local mapMyVault index found {found_count} relevant matching file"
                f"{'' if found_count == 1 else 's'}. Use the cited paths and excerpts below."
            )
        return {
            "question": question,
            "answer": answer,
            "found_count": found_count,
            "returned_count": evidence.get("returned_count", len(evidence.get("results", []))),
            "paths": paths,
            "results": evidence.get("results", []),
            "tool": "ask_mapmyvault",
            "instruction": (
                "Use the answer field as the direct answer. Cite paths from paths/results. "
                "Do not invent extra files or shell-style tool calls."
            ),
        }

    def count_matches(self, query: str = "", limit: int = 100) -> Dict:
        """Count files whose path, extracted text, or summary contains query terms."""
        groups = _term_groups(query)
        if not groups:
            return {
                "query": query,
                "terms": [],
                "term_groups": [],
                "count": 0,
                "paths": [],
                "instruction": "No searchable terms were provided.",
            }
        strict_matches = []
        relaxed_matches = []
        rows = self.store.files("file")
        lowered_query = query.lower()
        if "undergrad" in lowered_query or "undergraduate" in lowered_query:
            path_matches = [
                row
                for row in rows
                if re.search(r"(^|/)undergrad(uate)?(/|$)", row["path"].lower())
            ]
            if path_matches:
                rows = path_matches
                groups = [
                    group
                    for group in groups
                    if not set(group).intersection(
                        {"undergrad", "undergraduate", "degree", "degrees", "bsc", "ba", "bachelor"}
                    )
                ]
        for row in rows:
            haystack = _haystack(row)
            score = _group_match_count(haystack, groups)
            if score == len(groups):
                strict_matches.append(row)
            elif len(groups) > 2 and score >= max(2, len(groups) - 1):
                relaxed_matches.append(row)
        matches = strict_matches or relaxed_matches
        matches = sorted(matches, key=lambda row: row["path"].lower())
        return {
            "query": query,
            "terms": _terms(query),
            "term_groups": groups,
            "match_policy": "strict_all_terms" if strict_matches else "relaxed_keyword_terms",
            "count": len(matches),
            "paths": [row["path"] for row in matches[:limit]],
            "truncated": len(matches) > limit,
            "instruction": "Use count as the exact number of local indexed files that match the normalized term groups.",
        }

    def get_file_summary(self, path: str) -> Dict:
        row = self.store.get_file_by_path(path)
        return self._result(row) if row else {}

    def get_folder_children(self, path: str = "") -> List[Dict]:
        return [
            {"path": row["path"], "kind": row["kind"]}
            for row in self.store.connection.execute(
                "SELECT path,kind FROM files WHERE deleted=0 AND COALESCE(parent,'')=? ORDER BY path",
                (path,),
            )
        ]

    def get_related_files(self, path: str) -> List[Dict]:
        row = self.store.get_file_by_path(path)
        return (
            [dict(item) for item in self.store.relationships_for(row["id"])]
            if row
            else []
        )

    def list_files_by_format(self, file_format: str) -> List[Dict]:
        target = file_format.lower().lstrip(".")
        results = []
        for row in self.store.files("file"):
            metadata = json.loads(row["document_metadata_json"] or "{}")
            if metadata.get("format", "").lower() == target:
                results.append(self._result(row))
        return results

    def read_file_excerpt(self, path: str, max_chars: int = 2000) -> str:
        row = self.store.get_file_by_path(path)
        return (row["extracted_text"] or "")[:max_chars] if row else ""

    def _result(self, row, match_type: str = "direct", query: str = "") -> Dict:
        if not row:
            return {}
        return {
            "path": row["path"],
            "kind": row["kind"],
            "match_type": match_type,
            "document_metadata": json.loads(row["document_metadata_json"] or "{}"),
            "summary": json.loads(row["summary_json"] or "{}"),
            "excerpt": self._excerpt(row, query),
            "parsed": json.loads(row["parse_data"] or "{}"),
        }

    def _excerpt(self, row, query: str) -> str:
        text = row["extracted_text"] or ""
        if not text:
            return ""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        start = 0
        for term in _terms(query or row["path"]):
            index = lowered.find(term)
            if index >= 0:
                start = max(index - 120, 0)
                break
        excerpt = normalized[start:start + SNIPPET_CHARS]
        if start:
            excerpt = "..." + excerpt
        if start + SNIPPET_CHARS < len(normalized):
            excerpt += "..."
        return excerpt

    def _match_score(
        self, item: Dict, terms: List[str], path_only: bool = False
    ) -> int:
        if not terms:
            return 0
        path = item["path"].lower()
        if path_only:
            return sum(1 for term in terms if term in path)
        summary = item.get("summary", {})
        haystack = " ".join(
            [
                path,
                summary.get("title", ""),
                summary.get("purpose", ""),
                " ".join(summary.get("topics", [])),
                " ".join(summary.get("key_entities", [])),
                item.get("excerpt", ""),
            ]
        ).lower()
        return sum(1 for term in terms if term in haystack)
