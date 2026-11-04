"""The generated docs are current and every relative link in the docs resolves."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "jetson" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#L(\d+))?\)")


def test_generated_docs_are_current():
    r = subprocess.run([sys.executable, "scripts/gen_code_reference.py", "--check"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_relative_links_resolve():
    broken = []
    for doc in DOCS:
        for target, line in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (doc.parent / target).resolve()
            if not path.exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
            elif line and path.is_file():
                n = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
                if int(line) > n:
                    broken.append(f"{doc.relative_to(ROOT)} -> {target}#L{line} (file has {n} lines)")
    assert not broken, "\n".join(broken)
