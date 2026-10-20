"""Integration layer; keep the upstream solver pinned and unchanged."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "vendor" / "reasoning-bank"
sys.path.insert(0, str(UPSTREAM / "third_party" / "src"))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(ROOT / ".cache" / "mini"))

# Harbor reads task files with Path.read_text() and no encoding. On a
# non-UTF-8 Windows locale that uses the ANSI code page and rejects
# Terminal-Bench instructions. Task sources are UTF-8.
def _read_text_utf8(self, encoding=None, errors=None, newline=None):
    # Path.read_text gained newline in Python 3.13; Path.open supports it on 3.11 too.
    with self.open(encoding=encoding or "utf-8", errors=errors, newline=newline) as fh:
        return fh.read()


Path.read_text = _read_text_utf8
