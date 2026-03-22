"""
Knowledge Retriever — Searches the ingested knowledge base for relevant chunks.

Used by pipeline.py to inject expert knowledge into agent prompts.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "knowledge.db")

# How many chunks to return per query
TOP_K = 3
# Max total characters to return (to avoid blowing up token budget)
MAX_CHARS = 2000


def _get_db() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)


def _build_fts_query(text: str) -> str:
    """
    Build an FTS5 query from natural language text.
    Extracts meaningful words and joins with OR for broad matching.
    """
    # Remove very short words and common stopwords
    stopwords = {
        "the", "is", "at", "in", "on", "a", "an", "and", "or", "to", "of",
        "את", "של", "על", "עם", "אני", "הוא", "היא", "זה", "לא", "כן",
        "מה", "אם", "גם", "כי", "אבל", "רק", "כל", "עוד", "יש", "אין",
        "שלי", "שלך", "שלו", "שלה", "היה", "הייתה", "להיות", "יכול",
    }
    words = text.split()
    keywords = [w for w in words if len(w) > 2 and w.lower() not in stopwords]

    if not keywords:
        return None

    # Use OR matching — broader results
    return " OR ".join(f'"{w}"' for w in keywords[:10])


def retrieve(
    user_text: str,
    nlp_keywords: Optional[List[str]] = None,
    top_k: int = TOP_K,
    max_chars: int = MAX_CHARS,
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for chunks relevant to the user's message.

    Args:
        user_text: The user's latest message
        nlp_keywords: Optional keywords from NLP extraction (emotions, beliefs, etc.)
        top_k: Number of chunks to return
        max_chars: Maximum total characters across all returned chunks

    Returns:
        List of dicts: {author, source_name, page_or_time, content, rank}
    """
    conn = _get_db()
    if conn is None:
        return []

    # Combine user text with NLP-derived keywords for better matching
    search_text = user_text
    if nlp_keywords:
        search_text += " " + " ".join(nlp_keywords)

    fts_query = _build_fts_query(search_text)
    if not fts_query:
        return []

    try:
        rows = conn.execute(
            """
            SELECT author, source_name, page_or_time, content,
                   rank
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, top_k * 2)  # fetch extra, then trim by char limit
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()

    results = []
    total_chars = 0
    for author, source_name, page_or_time, content, rank in rows:
        if total_chars + len(content) > max_chars:
            if results:  # At least return one result
                break
            content = content[:max_chars]  # Trim first result if too long

        results.append({
            "author": author,
            "source_name": source_name,
            "page_or_time": page_or_time,
            "content": content,
            "rank": rank,
        })
        total_chars += len(content)

        if len(results) >= top_k:
            break

    return results


def format_for_prompt(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved knowledge chunks for injection into a prompt.
    Returns empty string if no results.
    """
    if not results:
        return ""

    parts = ["KNOWLEDGE_CONTEXT (from expert sources, use the thinking style — do NOT quote or cite):"]
    for r in results:
        source_info = f"[{r['author']} — {r['source_name']}, {r['page_or_time']}]"
        parts.append(f"\n{source_info}\n{r['content']}")

    return "\n".join(parts)


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "ביקורת עצמית דיאלוג פנימי"
    print(f"Searching for: {query}\n")
    results = retrieve(query)
    if results:
        print(format_for_prompt(results))
    else:
        print("No results found. Run ingest.py first.")
