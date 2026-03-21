#!/usr/bin/env python3
"""
Knowledge Ingest — Processes PDFs (and YouTube links) into a searchable SQLite FTS5 database.

Usage:
    # Ingest a folder of PDFs:
    python3 ingest.py --pdf-dir ~/Desktop/"knowledge NLP"

    # Ingest a single PDF:
    python3 ingest.py --pdf ~/Desktop/"knowledge NLP"/Steve\ andreas/2009-Help-with-Negative-Self-Talk.pdf

    # Ingest YouTube video (by transcript):
    python3 ingest.py --youtube "https://youtube.com/watch?v=XXXXX"

    # Ingest all YouTube links from a file:
    python3 ingest.py --youtube-file ~/agents/knowledge/youtube_queue.txt

    # Show stats:
    python3 ingest.py --stats
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "knowledge.db")
CHUNK_SIZE = 500          # words per chunk
CHUNK_OVERLAP = 100       # word overlap between chunks


# ── Database ────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id   TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,          -- 'pdf' | 'youtube' | 'text'
            name        TEXT NOT NULL,
            author      TEXT DEFAULT '',
            path_or_url TEXT NOT NULL,
            ingested_at TEXT DEFAULT (datetime('now')),
            chunk_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            chunk_id,
            source_id,
            author,
            source_name,
            page_or_time,
            content,
            tokenize='unicode61'
        )
    """)
    conn.commit()
    return conn


def source_exists(conn: sqlite3.Connection, source_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    return row is not None


def make_source_id(path_or_url: str) -> str:
    return hashlib.sha256(path_or_url.encode()).hexdigest()[:16]


# ── Chunking ────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if words else []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── PDF Ingestion ───────────────────────────────────────
def ingest_pdf(conn: sqlite3.Connection, pdf_path: str, author: str = "") -> int:
    source_id = make_source_id(pdf_path)
    if source_exists(conn, source_id):
        print(f"  ⏭  Already ingested: {os.path.basename(pdf_path)}")
        return 0

    import pdfplumber

    name = os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ")

    print(f"  📄 Processing: {name}")
    total_chunks = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 50:
                    continue

                for i, chunk in enumerate(chunk_text(text)):
                    chunk_id = f"{source_id}_p{page_num}_{i}"
                    conn.execute(
                        "INSERT INTO chunks(chunk_id, source_id, author, source_name, page_or_time, content) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (chunk_id, source_id, author, name, f"page {page_num}", chunk)
                    )
                    total_chunks += 1

        conn.execute(
            "INSERT INTO sources(source_id, source_type, name, author, path_or_url, chunk_count) "
            "VALUES (?, 'pdf', ?, ?, ?, ?)",
            (source_id, name, author, pdf_path, total_chunks)
        )
        conn.commit()
        print(f"  ✅ {total_chunks} chunks from {name}")
    except Exception as e:
        print(f"  ❌ Error processing {name}: {e}")

    return total_chunks


def ingest_pdf_dir(conn: sqlite3.Connection, dir_path: str) -> int:
    total = 0
    dir_path = os.path.expanduser(dir_path)
    for root, dirs, files in os.walk(dir_path):
        # Use folder name as author
        author = os.path.basename(root)
        if author == os.path.basename(dir_path):
            author = ""

        for f in sorted(files):
            if f.lower().endswith(".pdf") and not f.startswith("."):
                total += ingest_pdf(conn, os.path.join(root, f), author=author)

    return total


# ── YouTube Ingestion ───────────────────────────────────
def ingest_youtube(conn: sqlite3.Connection, url: str) -> int:
    """Ingest YouTube video transcript. Requires youtube-transcript-api."""
    source_id = make_source_id(url)
    if source_exists(conn, source_id):
        print(f"  ⏭  Already ingested: {url}")
        return 0

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("  ⚠️  youtube-transcript-api not installed.")
        print("     Run: pip3 install youtube-transcript-api")
        return 0

    # Extract video ID
    video_id = None
    for pattern in [r'v=([^&]+)', r'youtu\.be/([^?]+)', r'shorts/([^?]+)']:
        m = re.search(pattern, url)
        if m:
            video_id = m.group(1)
            break

    if not video_id:
        print(f"  ❌ Can't extract video ID from: {url}")
        return 0

    print(f"  🎬 Processing YouTube: {video_id}")

    try:
        # Try Hebrew first, then English, then any
        transcript = None
        for lang in [['iw', 'he'], ['en'], None]:
            try:
                if lang:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=lang)
                else:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                break
            except Exception:
                continue

        if not transcript:
            print(f"  ❌ No transcript found for {video_id}")
            return 0

        # Combine transcript entries into text
        full_text = " ".join(entry['text'] for entry in transcript)
        name = f"YouTube: {video_id}"
        total_chunks = 0

        for i, chunk in enumerate(chunk_text(full_text)):
            # Find approximate timestamp
            word_pos = i * (CHUNK_SIZE - CHUNK_OVERLAP)
            approx_time = ""
            cum_words = 0
            for entry in transcript:
                cum_words += len(entry['text'].split())
                if cum_words >= word_pos:
                    mins = int(entry['start'] // 60)
                    secs = int(entry['start'] % 60)
                    approx_time = f"{mins}:{secs:02d}"
                    break

            chunk_id = f"{source_id}_yt_{i}"
            conn.execute(
                "INSERT INTO chunks(chunk_id, source_id, author, source_name, page_or_time, content) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, source_id, "", name, approx_time, chunk)
            )
            total_chunks += 1

        conn.execute(
            "INSERT INTO sources(source_id, source_type, name, author, path_or_url, chunk_count) "
            "VALUES (?, 'youtube', ?, '', ?, ?)",
            (source_id, name, url, total_chunks)
        )
        conn.commit()
        print(f"  ✅ {total_chunks} chunks from YouTube {video_id}")
        return total_chunks

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0


def ingest_youtube_file(conn: sqlite3.Connection, file_path: str) -> int:
    total = 0
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                url = line.split("#")[0].strip()  # Remove inline comments
                if url:
                    total += ingest_youtube(conn, url)
    return total


# ── Stats ───────────────────────────────────────────────
def print_stats(conn: sqlite3.Connection):
    sources = conn.execute("SELECT source_type, name, author, chunk_count FROM sources ORDER BY author, name").fetchall()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    print(f"\n{'═'*60}")
    print(f"  📚 Knowledge Base Stats")
    print(f"{'═'*60}")
    print(f"  Total sources: {len(sources)}")
    print(f"  Total chunks:  {total_chunks}")
    print(f"{'─'*60}")

    by_author = {}
    for stype, name, author, count in sources:
        author = author or "(no author)"
        by_author.setdefault(author, []).append((stype, name, count))

    for author, items in sorted(by_author.items()):
        print(f"\n  📁 {author}")
        for stype, name, count in items:
            icon = "📄" if stype == "pdf" else "🎬" if stype == "youtube" else "📝"
            print(f"     {icon} {name[:50]:50s} [{count} chunks]")

    print(f"\n{'═'*60}")


# ── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Ingest")
    parser.add_argument("--pdf", help="Ingest a single PDF file")
    parser.add_argument("--pdf-dir", help="Ingest all PDFs in a directory (recursive)")
    parser.add_argument("--youtube", help="Ingest a YouTube video transcript")
    parser.add_argument("--youtube-file", help="Ingest YouTube links from a text file")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    args = parser.parse_args()

    conn = get_db()

    if args.stats:
        print_stats(conn)
        return

    if not any([args.pdf, args.pdf_dir, args.youtube, args.youtube_file]):
        parser.print_help()
        return

    total = 0

    if args.pdf:
        total += ingest_pdf(conn, os.path.expanduser(args.pdf))

    if args.pdf_dir:
        total += ingest_pdf_dir(conn, args.pdf_dir)

    if args.youtube:
        total += ingest_youtube(conn, args.youtube)

    if args.youtube_file:
        total += ingest_youtube_file(conn, os.path.expanduser(args.youtube_file))

    print(f"\n{'─'*40}")
    print(f"Total new chunks ingested: {total}")
    print_stats(conn)

    conn.close()


if __name__ == "__main__":
    main()
