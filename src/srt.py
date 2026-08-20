"""Detect and clean .srt subtitle input before it reaches the screenplay parser.

Subtitle files have none of the structure the pipeline expects (scene
headings, ALL-CAPS speaker lines) and instead carry numbered timestamp
blocks with inline HTML styling — fed in raw, that HTML leaks into
extracted entities and most lines get miscounted as dialogue. This module
strips indices/timestamps/markup down to plain narration/dialogue text,
keeping "SPEAKER: line" attribution where subtitles include it.
"""

import re

TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}"
)
INDEX_LINE_RE = re.compile(r"^\s*\d+\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def looks_like_srt(text: str) -> bool:
    """Heuristic: several HH:MM:SS,mmm --> HH:MM:SS,mmm lines present."""
    if not text:
        return False
    count = sum(1 for line in text.splitlines() if TIMESTAMP_LINE_RE.match(line))
    return count >= 3


def srt_to_text(text: str) -> str:
    """Strip cue indices, timestamps, and inline HTML markup from raw .srt
    content, returning plain lines suitable for the screenplay parser."""
    out_lines: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        for line in block.splitlines():
            line = line.strip()
            if not line or INDEX_LINE_RE.match(line) or TIMESTAMP_LINE_RE.match(line):
                continue
            clean = HTML_TAG_RE.sub("", line).strip()
            clean = clean.lstrip("-").strip()  # dual-speaker "- line" convention
            if clean:
                out_lines.append(clean)
    return "\n".join(out_lines)
