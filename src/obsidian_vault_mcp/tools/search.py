"""Search tools for the Obsidian vault MCP server."""

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import frontmatter

from .. import config
from ..vault import resolve_vault_path
from ..utils import sanitize_for_json, SafeJSONEncoder

logger = logging.getLogger(__name__)


def _search_ripgrep(
    query: str,
    search_path: Path,
    file_pattern: str,
    max_results: int,
    context_lines: int,
) -> list[dict]:
    """Search using ripgrep for performance."""
    cmd = [
        "rg",
        "--json",
        f"--max-count={max_results}",
        f"--glob={file_pattern}",
        "-i",
        f"--context={context_lines}",
        query,
        str(search_path),
    ]

    for excluded in config.EXCLUDED_DIRS:
        cmd.insert(-2, f"--glob=!{excluded}/")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    matches = []
    current_match = None

    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "match":
            match_data = data["data"]
            file_path = match_data["path"]["text"]
            try:
                rel_path = str(Path(file_path).relative_to(config.VAULT_PATH))
            except ValueError:
                continue

            line_number = match_data["line_number"]
            line_text = match_data["lines"]["text"].rstrip("\n")

            matches.append({
                "path": rel_path,
                "line_number": line_number,
                "match_context": line_text,
            })

            if len(matches) >= max_results:
                break

    return matches


def _search_python(
    query: str,
    search_path: Path,
    file_pattern: str,
    max_results: int,
    context_lines: int,
) -> list[dict]:
    """Fallback Python-based search."""
    import fnmatch

    query_lower = query.lower()
    matches = []

    for file_path in search_path.rglob("*"):
        if not file_path.is_file():
            continue

        if any(part in config.EXCLUDED_DIRS for part in file_path.parts):
            continue

        if not fnmatch.fnmatch(file_path.name, file_pattern):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        lines = content.splitlines()
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(lines[start:end])

                try:
                    rel_path = str(file_path.relative_to(config.VAULT_PATH))
                except ValueError:
                    continue

                matches.append({
                    "path": rel_path,
                    "line_number": i + 1,
                    "match_context": context,
                })

                if len(matches) >= max_results:
                    return matches

    return matches


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for",
    "on", "at", "by", "it", "its", "this", "that", "from", "with", "as",
    "be", "or", "and", "not", "but", "has", "had", "have", "do", "does",
    "did", "will", "would", "can", "could", "may", "might", "shall",
    "should", "about", "into", "than", "then", "so", "if", "no", "up",
    "out", "my", "your", "our", "his", "her", "their", "we", "you", "he",
    "she", "they", "me", "him", "us", "them",
})


_INTERROGATIVES = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
})

_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
_QUERY_TOKEN_EDGE_CHARS = "?!,;:()[]{}<>\"'`"


def _tokenize_query(query: str) -> list[str]:
    """Extract content tokens from a natural-language query for vault_query's
    keyword leg: quoted phrases kept intact as single tokens, stopwords and
    interrogatives (what/how/where/...) dropped -- vault_query is fed full
    questions rather than the 2-4 bare nouns vault_search expects, so
    interrogatives need stripping here on top of the existing stopword list.

    Only whitespace is ever a delimiter (same principle as the fallback
    tokenizer below), so identifiers, paths, env var names, ports, and error
    codes (BS_BRAIN_API_KEY, ~/.config/supervisor/, 502, feature/vault-tools-v2)
    survive as single atomic tokens -- underscores, slashes, dots, and hyphens
    are never split points. Edge punctuation (trailing '?', ',', etc. left over
    from sentence structure) is stripped so a token searches as the literal
    word it names.
    """
    phrases = [p.strip() for p in _QUOTED_PHRASE_RE.findall(query) if p.strip()]
    remainder = _QUOTED_PHRASE_RE.sub(" ", query)

    tokens: list[str] = list(phrases)
    for word in remainder.split():
        stripped = word.strip(_QUERY_TOKEN_EDGE_CHARS)
        if not stripped or len(stripped) <= 1:
            continue
        if stripped.lower() in _STOPWORDS or stripped.lower() in _INTERROGATIVES:
            continue
        tokens.append(stripped)

    return tokens


def _search_by_tokens(
    keywords: list[str],
    search_path: Path,
    file_pattern: str,
    max_results: int,
    context_lines: int,
    require_all: bool = True,
) -> list[dict]:
    """Single-pass, per-token ranked search: reads each file once, ranks by how
    many distinct tokens it contains, plus a filename boost. Shared by
    _search_keyword_fallback (which supplies its own lowercased, split-on-
    whitespace keyword list, require_all=True, its long-standing behaviour) and
    vault_query's tokenized keyword leg (which supplies richer tokens from
    _tokenize_query with require_all=False).

    require_all=True prefers an AND match (every keyword present) over any
    partial match, falling back to OR ranking only when no file satisfies all
    keywords. This is fine for short queries with 2-4 nouns, but for long,
    natural-language questions (many tokens) it becomes an all-or-nothing
    cliff: a single large, topically broad file that happens to contain every
    token *somewhere* in its body (an append-only changelog, say) wins
    outright and excludes every partial-but-more-relevant match entirely --
    this crowded out the correct answer for several 2026-08-08 diagnosis
    questions once tokenization started producing clean-enough keyword lists
    for an accidental full AND match to actually succeed. require_all=False
    skips that gate and ranks every candidate purely by keyword-match count
    (+ filename boost), so a partial match doesn't get discarded just because
    some unrelated file happened to match everything.
    """
    import fnmatch

    if not keywords:
        return []

    # Single pass: read each file once and record which keywords it contains
    file_data = []  # (file_path, rel_path, content, found_keywords)

    for file_path in search_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in config.EXCLUDED_DIRS for part in file_path.parts):
            continue
        if not fnmatch.fnmatch(file_path.name, file_pattern):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        try:
            rel_path = str(file_path.relative_to(config.VAULT_PATH))
        except ValueError:
            continue
        content_lower = content.lower()
        found = [kw for kw in keywords if kw in content_lower]
        if found:
            file_data.append((file_path, rel_path, content, found))

    if not file_data:
        return []

    if require_all:
        # AND logic: files containing all keywords
        and_matches = [(fp, rp, c, fk) for fp, rp, c, fk in file_data if len(fk) == len(keywords)]
        candidates = and_matches if and_matches else file_data
    else:
        candidates = file_data

    # Rank by keyword count + filename boost
    def _score(item):
        fp, rp, c, fk = item
        name_lower = fp.stem.lower()
        filename_boost = sum(3 for kw in keywords if kw in name_lower)
        return len(fk) + filename_boost

    candidates = sorted(candidates, key=_score, reverse=True)

    # Build result entries (one context block per keyword per file)
    matches = []
    for file_path, rel_path, content, found_keywords in candidates:
        lines = content.splitlines()
        for kw in found_keywords:
            for i, line in enumerate(lines):
                if kw in line.lower():
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    matches.append({
                        "path": rel_path,
                        "line_number": i + 1,
                        "match_context": "\n".join(lines[start:end]),
                        "search_mode": "keyword_fallback",
                    })
                    break  # one context block per keyword per file
            if len(matches) >= max_results:
                return matches

    return matches


def _search_keyword_fallback(
    query: str,
    search_path: Path,
    file_pattern: str,
    max_results: int,
    context_lines: int,
) -> list[dict]:
    """Keyword-based fallback search when ripgrep returns 0 results."""
    words = query.lower().split()
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return _search_by_tokens(keywords, search_path, file_pattern, max_results, context_lines)


# vault_search tokenized-augmentation tuning. A query at or under this many extracted
# content tokens is short enough that it's effectively the "2-4 nouns" case vault_search
# already expects -- and, critically, the case a caller doing exact-string verification
# before vault_str_replace is relying on. Leave it byte-identical to pre-change
# behaviour. Same threshold as vault_query's keyword leg (query.py's
# _SHORT_QUERY_TOKEN_THRESHOLD), kept as a separate constant per-module rather than a
# shared import so the two kill switches (VAULT_SEARCH_TOKENIZE and
# VAULT_QUERY_KEYWORD_TOKENIZE) stay fully decoupled.
_VAULT_SEARCH_TOKEN_THRESHOLD = 4


def _tag_match_type(matches: list[dict], match_type: str) -> list[dict]:
    for match in matches:
        match["match_type"] = match_type
    return matches


def _augment_with_tokenized_matches(
    query: str,
    literal_matches: list[dict],
    search_path: Path,
    file_pattern: str,
    max_results: int,
    context_lines: int,
) -> list[dict]:
    """Augments vault_search's literal (ripgrep/Python) results with tokenized
    matches for long queries, without ever reordering, demoting, or dropping a
    literal match.

    `literal_matches` -- whatever the exact-string search phase found, including
    empty -- always comes first, tagged match_type="literal". For queries with
    more than _VAULT_SEARCH_TOKEN_THRESHOLD content tokens, tokenized matches
    (reusing vault_query's _tokenize_query, not a second tokenizer) are appended
    below them, tagged match_type="tokenized", with paths already present in the
    literal set skipped so no file appears twice.

    This tagging is the exact-string-verification contract: a caller checking
    whether a string exists before vault_str_replace must filter to
    match_type == "literal" and treat zero such matches as "does not exist",
    even if tokenized matches are also present.
    """
    tagged_literal = _tag_match_type(literal_matches, "literal")
    seen_paths = {m["path"] for m in tagged_literal}

    tokens = [t.lower() for t in _tokenize_query(query)]
    tokenized = _search_by_tokens(
        tokens, search_path, file_pattern, max_results, context_lines, require_all=True
    )
    appended = _tag_match_type(
        [m for m in tokenized if m["path"] not in seen_paths], "tokenized"
    )

    return (tagged_literal + appended)[:max_results]


def _get_frontmatter_excerpt(file_path: Path, max_keys: int = 3) -> dict | None:
    """Read frontmatter from a file, returning first N key-value pairs."""
    try:
        content = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(content)
        if not post.metadata:
            return None
        keys = list(post.metadata.keys())[:max_keys]
        return {k: post.metadata[k] for k in keys}
    except Exception:
        return None


def vault_search(
    query: str,
    path_prefix: str | None = None,
    file_pattern: str = "*.md",
    max_results: int = 20,
    context_lines: int = 2,
) -> str:
    """Search for text across vault files."""
    try:
        if path_prefix:
            search_path = resolve_vault_path(path_prefix)
        else:
            search_path = config.VAULT_PATH

        if not search_path.is_dir():
            return json.dumps({"error": f"Search path is not a directory: {path_prefix}"})

        if shutil.which("rg"):
            matches = _search_ripgrep(query, search_path, file_pattern, max_results, context_lines)
        else:
            matches = _search_python(query, search_path, file_pattern, max_results, context_lines)

        if config.VAULT_SEARCH_TOKENIZE and len(_tokenize_query(query)) > _VAULT_SEARCH_TOKEN_THRESHOLD:
            matches = _augment_with_tokenized_matches(
                query, matches, search_path, file_pattern, max_results, context_lines
            )
        elif not matches:
            matches = _search_keyword_fallback(query, search_path, file_pattern, max_results, context_lines)

        for match in matches:
            file_full_path = config.VAULT_PATH / match["path"]
            match["frontmatter_excerpt"] = _get_frontmatter_excerpt(file_full_path)

        truncated = len(matches) >= max_results

        return json.dumps(sanitize_for_json({
            "results": matches,
            "total_matches": len(matches),
            "truncated": truncated,
        }), cls=SafeJSONEncoder)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"vault_search error: {e}")
        return json.dumps({"error": str(e)})


def vault_search_frontmatter(
    field: str,
    value: str = "",
    match_type: str = "exact",
    path_prefix: str | None = None,
    max_results: int = 20,
) -> str:
    """Search vault files by frontmatter field values using the in-memory index."""
    from ..server import frontmatter_index

    try:
        results = frontmatter_index.search_by_field(
            field=field,
            value=value,
            match_type=match_type,
            path_prefix=path_prefix,
        )

        formatted = []
        for item in results[:max_results]:
            path = item["path"]
            fm = item["frontmatter"]
            title = fm.get("title", Path(path).stem)
            formatted.append({
                "path": path,
                "frontmatter": fm,
                "title": title,
            })

        truncated = len(results) > max_results

        return json.dumps(sanitize_for_json({
            "results": formatted,
            "total": len(formatted),
            "truncated": truncated,
        }), cls=SafeJSONEncoder)
    except Exception as e:
        logger.error(f"vault_search_frontmatter error: {e}")
        return json.dumps({"error": str(e)})
