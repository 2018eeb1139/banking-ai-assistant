import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    document_id: str
    title: str
    version: str | None
    effective_date: str | None
    owner: str | None
    section: str
    text: str
    is_authoritative: bool


def _metadata_value(header: str, label: str) -> str | None:
    match = re.search(
        rf"(?:^|\|)\s*{re.escape(label)}\s*:?\s*([^|]+)",
        header,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _split_oversized_paragraph(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks


def _chunk_section(
    text: str,
    max_words: int,
    overlap: int,
) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current: list[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if paragraph_word_count > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_word_count = 0
            chunks.extend(
                _split_oversized_paragraph(paragraph, max_words, overlap)
            )
            continue

        if current and current_word_count + paragraph_word_count > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_word_count = 0

        current.append(paragraph)
        current_word_count += paragraph_word_count

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _parse_policy(path: Path, max_words: int, overlap: int) -> list[PolicyChunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next((line for line in lines if "Document ID:" in line), "")

    document_id = _metadata_value(header, "Document ID") or path.stem.split("_")[0]
    version = _metadata_value(header, "Version")
    effective_date = _metadata_value(header, "Effective")
    owner = _metadata_value(header, "Owner")

    title = document_id
    heading_stack: list[str] = []
    sections: list[tuple[str, str]] = []
    body_lines: list[str] = []

    def flush_section() -> None:
        text = "\n".join(body_lines).strip()
        if text:
            section = " > ".join(heading_stack) or title
            sections.append((section, text))
        body_lines.clear()

    for line in lines:
        if line.startswith("#"):
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                level = len(match.group(1))
                heading = match.group(2).strip()

                if level == 1 and title == document_id:
                    title = heading
                    continue

                flush_section()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                continue

        if re.match(r"^\s*Document ID:", line):
            continue

        body_lines.append(line)

    flush_section()

    if not sections:
        return []

    is_authoritative = not (
        (version and "draft" in version.lower())
        or "working notes" in title.lower()
        or "not policy" in "\n".join(text for _, text in sections).lower()
    )

    chunks: list[PolicyChunk] = []
    chunk_number = 0

    for section, text in sections:
        for chunk_text in _chunk_section(text, max_words, overlap):
            if not chunk_text:
                continue

            chunk_number += 1
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{document_id}:{chunk_number}",
                    document_id=document_id,
                    title=title,
                    version=version,
                    effective_date=effective_date,
                    owner=owner,
                    section=section,
                    text=chunk_text,
                    is_authoritative=is_authoritative,
                )
            )

    return chunks


def load_policy_corpus(
    corpus_dir: Path,
    max_words: int = 450,
    overlap: int = 40,
) -> list[PolicyChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be between zero and max_words - 1")
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Policy corpus directory not found: {corpus_dir}")

    policy_files = sorted(corpus_dir.glob("*.md"))
    if not policy_files:
        raise ValueError(f"No Markdown policy files found in {corpus_dir}")

    chunks = [
        chunk
        for path in policy_files
        for chunk in _parse_policy(path, max_words, overlap)
    ]
    if not chunks:
        raise ValueError(f"No policy text could be extracted from {corpus_dir}")

    return chunks