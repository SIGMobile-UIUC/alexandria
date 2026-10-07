"""Cached HTTP downloads for corpora that are not on the Hugging Face Hub."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from .io import get_logger

log = get_logger("download")
DEFAULT_CACHE = Path(os.environ.get("ALEXANDRIA_CACHE", Path.home() / ".cache" / "alexandria"))
# Some hosts (e.g. manythings.org) reject the default Python user agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; alexandria-models/1.0)"}


def download(url: str, cache_dir: str | Path | None = None, filename: str | None = None) -> Path:
    """Download ``url`` once into the cache directory and return the local path."""
    cache = Path(cache_dir or DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (filename or url.rstrip("/").rsplit("/", 1)[-1])
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        expected = resp.headers.get("Content-Length")
        shutil.copyfileobj(resp, f, length=1 << 20)
    got = tmp.stat().st_size
    if expected is not None and got != int(expected):  # a dropped connection must not be cached
        tmp.unlink(missing_ok=True)
        raise OSError(f"incomplete download of {url}: {got} of {expected} bytes")
    tmp.replace(dest)
    return dest
