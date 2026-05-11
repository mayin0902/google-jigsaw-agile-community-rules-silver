from __future__ import annotations

import re
from urllib.parse import urlparse


URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")


def clean_url_to_domain_path(text: object) -> str:
    """Replace a URL with a compact domain/path marker for embedding models."""
    if text is None:
        return ""

    def replace_url(match: re.Match[str]) -> str:
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                important_path = "/".join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            return f"<url>: ({domain})"
        except Exception:
            return "<url>: (unknown)"

    return URL_RE.sub(replace_url, str(text))


def url_to_semantics(text: object) -> str:
    """Extract URL domain/path keywords as extra lexical features."""
    if not isinstance(text, str):
        return ""

    urls = re.findall(r"https?://[^\s/$.?#].[^\s]*", text)
    if not urls:
        return ""

    semantics: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url_lower = url.lower()
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            for part in domain_match.group(1).split("."):
                if part and part not in seen and len(part) > 3:
                    semantics.append(f"domain:{part}")
                    seen.add(part)

        path = re.sub(r"^(?:https?://)?[a-z0-9.-]+\.[a-z]{2,}/?", "", url_lower)
        for part in re.split(r"[/_.-]+", path):
            if not part or not part.isalnum():
                continue
            cleaned = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if cleaned and cleaned not in seen and len(cleaned) > 3:
                semantics.append(f"path:{cleaned}")
                seen.add(cleaned)

    return "" if not semantics else "\nURL Keywords: " + " ".join(semantics)
