#!/usr/bin/env python3
"""Generate the code-level documentation from the source, so every link points to the right line.

  docs/CODE_REFERENCE.md   every file, class, function, method (with line links), every script's
                           command-line options and every test
  docs/PROJECT_DOSSIER.md  rendered from docs/src/PROJECT_DOSSIER.src.md: ``[[path]]`` becomes a file
                           link and ``[[path:Symbol]]`` / ``[[path:Class.method]]`` a link to that line

    python scripts/gen_code_reference.py            # regenerate both
    python scripts/gen_code_reference.py --check    # exit 1 if out of date or a [[link]] is broken

Re-run after changing code (tests/test_docs.py fails otherwise).
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "CODE_REFERENCE.md"
DOSSIER_SRC = ROOT / "docs" / "src" / "PROJECT_DOSSIER.src.md"
DOSSIER_OUT = ROOT / "docs" / "PROJECT_DOSSIER.md"

SECTIONS = [
    ("gdnav/common", "Common utilities"),
    ("gdnav/sim", "Simulator (vehicle, sensors, estimator, flight controller, world, camera, VIO)"),
    ("gdnav/perception", "Perception (detectors, localizer, temporal filter, floor-map matcher)"),
    ("gdnav/navigation", "Navigation (mission, state machine, local planner, geofence)"),
    ("gdnav/safety", "Safety (health monitor)"),
    ("gdnav/interface", "Vehicle interfaces (simulator, PX4 MAVSDK, Gazebo camera)"),
    ("gdnav/bridge", "Bridges (frames, landmark/VIO fusion, px4_msgs conversions)"),
    ("gdnav/tools", "Tools (report, replay, video)"),
    ("gdnav/dashboard", "Dashboard (server + browser client)"),
    ("gdnav", "Package root (runner = companion main loop)"),
    ("scripts", "Command-line scripts"),
    ("jetson", "Jetson / ROS 2 (Phase 3)"),
    ("tests", "Tests"),
]
OTHER_KINDS = {".yaml": "config", ".yml": "config", ".sdf": "Gazebo", ".config": "Gazebo", ".params": "PX4 params",
               ".sh": "shell", ".service": "systemd", ".js": "JavaScript", ".html": "HTML", ".css": "CSS",
               ".toml": "packaging", ".txt": "requirements", ".md": "docs", ".xml": "ROS package",
               ".svg": "diagram", ".example": "env template", ".cfg": "packaging"}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if "/vendor/" not in f]


def link(path: str, line: int | None = None, text: str | None = None) -> str:
    target = f"../{path}" + (f"#L{line}" if line else "")
    return f"[{text or path}]({target})"


def first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line.replace("|", "\\|")
    return ""


def first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    return " ".join(l.strip() for l in para.splitlines()).replace("|", "\\|")


def comment_hint(lines: list[str], node) -> str:
    """No docstring: use the comment just above the def, or the first comment inside the body."""
    def clean(c: str) -> str:
        c = c.strip().lstrip("#").strip().strip("-= ").strip()
        return c.replace("|", "\\|") if len(c) > 3 else ""

    i = node.lineno - 2
    while i >= 0 and lines[i].strip().startswith("@"):
        i -= 1
    if i >= 0 and lines[i].strip().startswith("#") and clean(lines[i]):
        return clean(lines[i])
    end = getattr(node, "end_lineno", node.lineno + 5)
    for j in range(node.lineno, min(end, node.lineno + 6)):
        t = lines[j].strip() if j < len(lines) else ""
        c = clean(t) if t.startswith("#") else (clean(t.split("#", 1)[1]) if "#" in t and not t.startswith(("'", '"')) else "")
        if c:
            return c
    return ""


def interface_docs() -> dict[str, str]:
    """First docstring line of every VehicleInterface method (inherited by the three backends)."""
    tree = ast.parse((ROOT / "gdnav/interface/vehicle.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VehicleInterface")
    return {m.name: first_line(ast.get_docstring(m)) for m in cls.body
            if isinstance(m, ast.FunctionDef) and ast.get_docstring(m)}


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args if a.arg not in ("self", "cls")]
    if fn.args.vararg:
        args.append("*" + fn.args.vararg.arg)
    args += [a.arg for a in fn.args.kwonlyargs]
    if fn.args.kwarg:
        args.append("**" + fn.args.kwarg.arg)
    s = ", ".join(args)
    return s if len(s) < 70 else s[:67] + "..."


def cli_options(tree: ast.Module) -> list[tuple[str, str]]:
    opts = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            help_ = next((k.value.value for k in node.keywords if k.arg == "help" and isinstance(k.value, ast.Constant)), "")
            opts.append((", ".join(names), str(help_).replace("|", "\\|")))
    return opts


def describe_python(path: str) -> list[str]:
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    srclines = src.splitlines()
    iface = interface_docs()

    def desc(n, cls=None):
        d = first_line(ast.get_docstring(n)) or comment_hint(srclines, n)
        if not d and cls is not None and n.name in iface:  # implementation of a VehicleInterface method
            d = f"VehicleInterface: {iface[n.name]}"
        return d
    lines = [f"#### {link(path)}", ""]
    doc = first_paragraph(ast.get_docstring(tree))
    if doc:
        lines += [doc, ""]
    rows = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            init = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
            sig = f"({signature(init)})" if init else ""
            rows.append(f"| class | {link(path, node.lineno, node.name)}{sig} | {desc(node)} |")
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                    rows.append(f"| method | {link(path, m.lineno, f'{node.name}.{m.name}')}({signature(m)}) | "
                                f"{desc(m, node)} |")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "test" if node.name.startswith("test_") else "function"
            rows.append(f"| {kind} | {link(path, node.lineno, node.name)}({signature(node)}) | "
                        f"{desc(node)} |")
    if rows:
        lines += ["| kind | name | description |", "|---|---|---|", *rows, ""]
    opts = cli_options(tree) if path.startswith("scripts/") else []
    if opts:
        lines += ["Command-line options:", "", "| option | help |", "|---|---|",
                  *[f"| `{o}` | {h} |" for o, h in opts], ""]
    return lines


def describe_other(path: str) -> str:
    p = ROOT / path
    kind = OTHER_KINDS.get(p.suffix, "file") if p.suffix else ("Docker" if "Dockerfile" in p.name else "file")
    desc = ""
    try:
        for line in p.read_text(encoding="utf-8").splitlines()[:12]:
            t = line.strip().lstrip("#").strip().lstrip("<!--").rstrip("-->").strip()
            if t and not t.startswith(("!/", "?xml", "{", "[", "-", "sdf", "<")) and len(t) > 12:
                desc = t.replace("|", "\\|")
                break
    except UnicodeDecodeError:
        desc = "(binary)"
    return f"| {link(path)} | {kind} | {desc[:140]} |"


def build() -> str:
    files = tracked_files()
    py = [f for f in files if f.endswith(".py")]
    out = ["# Code reference", "",
           "Generated by [`scripts/gen_code_reference.py`](../scripts/gen_code_reference.py) from the source "
           "(do not edit by hand; re-run it after changing code). Links point to the exact line.", "",
           f"{len(files)} tracked files, {len(py)} Python modules. Narrative status and design: "
           "[PROJECT_DOSSIER.md](PROJECT_DOSSIER.md).", "", "## Contents", ""]
    used = set()
    blocks = []
    for prefix, title in SECTIONS:
        mods = [f for f in py if f.startswith(prefix + "/") and f not in used
                and (prefix != "gdnav" or f.count("/") == 1)]
        if not mods:
            continue
        used.update(mods)
        anchor = title.lower().split(" (")[0].replace(" ", "-").replace("/", "")
        out.append(f"- [{title}](#{anchor})")
        blocks += [f"## {title.split(' (')[0]}", ""]
        for m in sorted(mods):
            blocks += describe_python(m)
    others = [f for f in files if not f.endswith(".py")]
    out.append("- [Non-Python files](#non-python-files)")
    blocks += ["## Non-Python files", "", "| file | kind | first comment / description |", "|---|---|---|",
               *[describe_other(f) for f in sorted(others)], ""]
    return "\n".join(out + [""] + blocks)


def symbol_line(path: str, symbol: str) -> int:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    parts = symbol.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next((n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == part), None)
        if node is None:
            # module-level constant assignment
            node = next((n for n in body if isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(t, ast.Name) and t.id == part for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))), None)
        if node is None:
            raise KeyError(f"{path}:{symbol}")
        body = getattr(node, "body", [])
    return node.lineno


def render_dossier() -> tuple[str, list[str]]:
    """[[path]] -> file link, [[path:Symbol]] -> line link (relative to docs/)."""
    import re
    errors: list[str] = []
    src = DOSSIER_SRC.read_text(encoding="utf-8")

    def repl(m):
        path, _, sym = m.group(1).partition(":")
        if not (ROOT / path).exists():
            errors.append(f"missing file {path}")
            return m.group(0)
        if not sym:
            return f"[`{path}`](../{path})"
        try:
            return f"[`{sym}`](../{path}#L{symbol_line(path, sym)})"
        except (KeyError, SyntaxError):
            errors.append(f"missing symbol {path}:{sym}")
            return m.group(0)

    out = re.sub(r"\[\[([^\]]+)\]\]", repl, src)
    header = ("<!-- Generated from docs/src/PROJECT_DOSSIER.src.md by scripts/gen_code_reference.py. "
              "Edit the source file, then re-run the script. -->\n")
    return header + out, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/CODE_REFERENCE.md is out of date")
    a = ap.parse_args()
    outputs = {OUT: build()}
    if DOSSIER_SRC.exists():
        dossier, errors = render_dossier()
        if errors:
            print("Broken [[links]] in the dossier source:\n  " + "\n  ".join(errors))
            return 1
        outputs[DOSSIER_OUT] = dossier
    if a.check:
        stale = [p.name for p, t in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        if stale:
            print(f"out of date: {', '.join(stale)} - run python scripts/gen_code_reference.py")
            return 1
        print("up to date")
        return 0
    for p, t in outputs.items():
        p.write_text(t, encoding="utf-8")
        print(f"Wrote {p.relative_to(ROOT)} ({len(t.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
