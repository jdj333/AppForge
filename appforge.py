#!/usr/bin/env python3
"""
AppForge — scaffold & edit project files via natural language prompts (OpenAI API).

This version focuses on build reliability and smooth startup:

- Python & Node baselines:
  - Guarantees Python `requirements.txt` (+ adds `constraints.txt`) with known-good pins
  - Guarantees Node `package.json` with usable scripts (and `.nvmrc`)
- Docker resilience:
  - Adds/repairs a cache-friendly Dockerfile for Python (when applicable)
  - Adds `.dockerignore` and healthcheck helpers
- Startup UX:
  - After PLAN and APPLY, prints concrete "Next steps to run the app (build/start)" with OS-specific
    open-in-browser command and consistent port (default 8000).
- Preflight & smoke:
  - `doctor` command: checks Python/Node/Docker availability, port availability
  - Generates run & smoke helper scripts automatically (executable) for local & Docker flows
- Safety:
  - Writes under ./output by default; safe path checks; no overwrite unless --force

"""

import argparse
import json
import os
import platform
import shutil
import socket
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

# ---- OpenAI client ----
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OUTDIR = "output"
DEFAULT_PORT = int(os.getenv("APPFORGE_PORT", "8000"))

# ---------- Data structures ----------
@dataclass
class FileChange:
    path: str
    contents: str
    executable: bool = False
    overwrite: Optional[bool] = None

@dataclass
class Plan:
    files: List[FileChange]
    notes: Optional[str] = None


# ---------- Utilities ----------
def within(root: Path, target: Path) -> bool:
    """Ensure target is within root (prevent path traversal)."""
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def pretty_table(rows: List[List[str]], header: Optional[List[str]] = None) -> str:
    if not rows and not header:
        return ""
    cols = len(rows[0]) if rows else (len(header) if header else 0)
    widths = [0] * cols
    def measure(row):
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    if header:
        measure(header)
    for r in rows:
        measure(r)
    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
    lines = []
    if header:
        lines.append(fmt(header))
        lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt(r))
    return "\n".join(lines)

def make_tree_preview(root: Path, max_chars: int = 4000) -> str:
    lines, total = [], 0
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if len(str(rel).split(os.sep)) > 6:
            continue
        size = p.stat().st_size
        line = f"{rel} ({size} bytes)"
        total += len(line) + 1
        if total > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
    return "\n".join(lines)

def get_os_open_cmd(url: str) -> str:
    if sys.platform == "darwin":
        return f"open {url}"
    if os.name == "nt":
        return f"start {url}"
    return f"xdg-open {url}"

def find_free_port(preferred: int = DEFAULT_PORT, limit: int = 10) -> int:
    port = preferred
    for _ in range(limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return preferred  # fallback

def read_python_version_major_minor() -> Tuple[int, int]:
    return sys.version_info.major, sys.version_info.minor


# ---------- Model I/O ----------
SYSTEM_PROMPT = """You are AppForge, a precise code scaffolder.

Return ONLY a strict JSON object with this schema, no prose:

{
  "files": [
    {
      "path": "<relative path like 'src/app.py' or 'Dockerfile'>",
      "contents": "<entire file contents>",
      "executable": false,
      "overwrite": false
    }
  ],
  "notes": "Optional short notes for the user (one paragraph max)."
}

Rules:
- Always include requirements.txt for Python projects (Flask 3.x with Werkzeug 3.x is preferred).
- Always include package.json for Node.js projects (include a working 'start' script).
- Use relative paths only.
- Prefer complete, minimal working examples.
- For web apps: include a /health route returning 'ok' and bind host 0.0.0.0; prefer env PORT (default 8000).
- Mark shell scripts executable=true.
- Set overwrite=true if essential.
- Do NOT include markdown code fences.
"""

def build_user_prompt(user_request: str, tree_preview: Optional[str]) -> str:
    base = f"User request:\n{user_request.strip()}\n"
    if tree_preview:
        base += f"\nProject tree preview:\n{tree_preview}\n"
    base += "\nProduce the JSON plan now."
    return base

def call_model(client: OpenAI, model: str, user_prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip().strip("`").strip()
        return json.loads(cleaned)


# ---------- Planning ----------
def parse_plan_json(raw: Dict[str, Any]) -> Plan:
    files: List[FileChange] = []
    for item in raw.get("files", []):
        files.append(FileChange(
            path=item.get("path"),
            contents=item.get("contents"),
            executable=bool(item.get("executable", False)),
            overwrite=bool(item.get("overwrite", False)) if item.get("overwrite") is not None else None
        ))
    return Plan(files=files, notes=raw.get("notes"))

def _get_file(plan: Plan, path: str) -> Optional[FileChange]:
    for fc in plan.files:
        if fc.path == path:
            return fc
    return None

def _set_file(plan: Plan, path: str, contents: str, executable: bool = False, overwrite: Optional[bool] = None):
    existing = _get_file(plan, path)
    if existing:
        existing.contents = contents
        existing.executable = executable
        existing.overwrite = overwrite
    else:
        plan.files.append(FileChange(path=path, contents=contents, executable=executable, overwrite=overwrite))

def _python_known_good_matrix(py_major: int, py_minor: int) -> Dict[str, str]:
    # Conservative pins compatible with Python 3.9–3.12
    return {
        "Flask": "3.0.3",
        "Werkzeug": "3.1.3",
        "Jinja2": "3.1.6",
        "itsdangerous": "2.2.0",
        "click": "8.1.8",
        "Flask-SQLAlchemy": "3.1.1",
        "SQLAlchemy": "2.0.43",
        # extras frequently required by SQLAlchemy 2.x
        "typing-extensions": "4.15.0",
        "greenlet": "3.2.4",
        "importlib-metadata": "8.7.0",  # harmless on newer Pythons (no-op)
        "MarkupSafe": "3.0.3",
        "zipp": "3.23.0",
    }

def _render_requirements_from_matrix(matrix: Dict[str, str]) -> str:
    lines = [f"{k}=={v}" for k, v in matrix.items()]
    return "\n".join(lines) + "\n"

def _render_constraints_from_matrix(matrix: Dict[str, str]) -> str:
    # Constrain transitive deps to avoid sudden breakage
    return _render_requirements_from_matrix(matrix)

def _repair_python_requirements(existing: str, py_major: int, py_minor: int) -> str:
    """
    Heuristic repair for conflicts like Flask 2.2.x + Werkzeug 3.x.
    We override to our known-good matrix if we detect likely problems.
    """
    text = existing or ""
    low = text.lower()
    if "flask==2." in low or ("flask" in low and "werkzeug==3" in low):
        # Force modern matrix
        matrix = _python_known_good_matrix(py_major, py_minor)
        return _render_requirements_from_matrix(matrix)
    # If it's very minimal or unpinned, upgrade to our matrix for reproducibility
    if "flask" in low and "werkzeug" not in low:
        matrix = _python_known_good_matrix(py_major, py_minor)
        return _render_requirements_from_matrix(matrix)
    # Else keep as-is (user/model might have a complete working set)
    return existing

def _python_run_scripts(port: int, entry_candidates: List[str]) -> Dict[str, str]:
    entry = "src/app.py"
    for c in entry_candidates:
        if c:
            entry = c
            break
    run_local = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
python -m venv .venv || true
. .venv/bin/activate
pip install --upgrade pip
if [ -f constraints.txt ]; then
  pip install -r requirements.txt -c constraints.txt
else
  pip install -r requirements.txt
fi
python {entry}
"""
    smoke = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
curl -fsS http://127.0.0.1:${{PORT}}/health || curl -fsS http://127.0.0.1:${{PORT}}/ || (echo "App did not respond" && exit 1)
echo "OK"
"""
    wait_py = """#!/usr/bin/env python3
import os, socket, sys, time
port = int(os.getenv("PORT", "8000"))
deadline = time.time() + 30
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", port))
            sys.exit(0)
        except Exception:
            time.sleep(0.5)
sys.exit(1)
"""
    return {"run_local.sh": run_local, "smoke.sh": smoke, "wait_for_port.py": wait_py}

def _python_dockerfile(port: int, entry_candidates: List[str]) -> str:
    entry = "src/app.py"
    for c in entry_candidates:
        if c:
            entry = c
            break
    return f"""# Auto-generated by AppForge
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PORT={port}
WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/ ./src/
EXPOSE {port}
HEALTHCHECK --interval=10s --timeout=2s --retries=5 CMD python - <<'PY' || exit 1
import os,urllib.request
port=os.getenv("PORT","{port}")
try:
    urllib.request.urlopen(f"http://127.0.0.1:{'{'}port{'}'}/health",timeout=1)
    print("ok")
except Exception:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{'{'}port{'}'}/",timeout=1)
        print("ok")
    except Exception as e:
        raise SystemExit(1)
PY

CMD ["python","{entry}"]
"""

def _dockerignore() -> str:
    return """.venv
__pycache__/
*.pyc
*.pyo
*.pyd
*.log
.git
.gitignore
output/
.env
"""

def _node_package_json() -> str:
    pkg = {
        "name": "appforge-project",
        "version": "1.0.0",
        "private": True,
        "type": "module",
        "scripts": {
            "start": "node index.js"
        },
        "engines": {
            "node": ">=18"
        },
        "dependencies": {}
    }
    return json.dumps(pkg, indent=2) + "\n"

def _node_run_scripts(port: int) -> Dict[str, str]:
    run = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
if command -v npm >/dev/null 2>&1; then
  npm install
  npm start
else
  echo "npm not found"; exit 1
fi
"""
    smoke = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
curl -fsS http://127.0.0.1:${{PORT}}/health || curl -fsS http://127.0.0.1:${{PORT}}/ || (echo "App did not respond" && exit 1)
echo "OK"
"""
    return {"run_local.sh": run, "smoke.sh": smoke}

def ensure_baseline_files(plan: Plan, prompt: str, port: int) -> Plan:
    """
    Guarantee:
      - Python: requirements.txt (pinned), constraints.txt, run scripts, Dockerfile (if mentioned or not present when app code exists), .dockerignore
      - Node: package.json, .nvmrc, run scripts
    Also repairs conflicting Python requirement sets.
    """
    lower = prompt.lower()
    paths = {fc.path for fc in plan.files}

    is_python = ("python" in lower or "flask" in lower or "django" in lower
                 or any(p.endswith(".py") for p in paths) or "requirements.txt" in paths)
    is_node = ("node" in lower or "react" in lower or "express" in lower
               or "package.json" in paths or any(p.endswith(".js") or p.endswith(".jsx") for p in paths))
    mentions_docker = ("docker" in lower) or any(Path(p).name.lower() == "dockerfile" for p in paths)

    # Python baseline
    if is_python:
        py_major, py_minor = read_python_version_major_minor()
        matrix = _python_known_good_matrix(py_major, py_minor)
        req = _render_requirements_from_matrix(matrix)

        if "requirements.txt" in paths:
            existing = _get_file(plan, "requirements.txt").contents
            req = _repair_python_requirements(existing, py_major, py_minor)
        _set_file(plan, "requirements.txt", req)

        # constraints.txt
        _set_file(plan, "constraints.txt", _render_constraints_from_matrix(matrix))

        # Add run/smoke helpers
        entry_candidates = []
        for candidate in ["src/app.py", "app.py"]:
            if candidate in paths:
                entry_candidates.append(candidate)
        helpers = _python_run_scripts(port, entry_candidates)
        for name, content in helpers.items():
            _set_file(plan, name, content, executable=name.endswith(".sh"))

        # Dockerfile (add or repair if python app present)
        if mentions_docker or "Dockerfile" in paths or "src/app.py" in paths or "app.py" in paths:
            dockerfile = _python_dockerfile(port, entry_candidates)
            _set_file(plan, "Dockerfile", dockerfile)
            _set_file(plan, ".dockerignore", _dockerignore())

    # Node baseline
    if is_node:
        if "package.json" not in paths:
            _set_file(plan, "package.json", _node_package_json())
        if ".nvmrc" not in paths:
            _set_file(plan, ".nvmrc", "18\n")
        # run/smoke
        node_helpers = _node_run_scripts(port)
        for name, content in node_helpers.items():
            if _get_file(plan, name) is None:  # don't clobber Python helper if both stacks
                _set_file(plan, name, content, executable=name.endswith(".sh"))

    return plan


def save_plan_to_file(plan: Plan, path: Path) -> None:
    payload = {"files": [fc.__dict__ for fc in plan.files], "notes": plan.notes}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------- Applying ----------
def resolve_output_root(project_root: Path, output_path: Optional[str], inplace: bool) -> Path:
    if inplace:
        return project_root
    out = Path(output_path or DEFAULT_OUTDIR)
    if not out.is_absolute():
        out = (project_root / out).resolve()
    return out

def apply_plan(plan: Plan, output_root: Path, dry_run: bool, force: bool) -> List[Path]:
    written: List[Path] = []
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    for fc in plan.files:
        target = (output_root / fc.path).resolve()
        if not within(output_root, target):
            raise PermissionError(f"Refusing to write outside output root: {fc.path}")
        exists = target.exists()
        if exists and not force:
            print(f"[SKIP] {fc.path} (exists). Use --force to overwrite.")
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as f:
                f.write(fc.contents)
            if fc.executable:
                mode = target.stat().st_mode
                os.chmod(target, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            written.append(target)
        print(f"{'[WRITE]' if not dry_run else '[DRY]'} {target.relative_to(output_root)}")
    return written


# ---------- Detection & Next Steps ----------
def detect_project(plan: Plan, prompt: str) -> Dict[str, bool]:
    paths = {fc.path for fc in plan.files}
    lower = prompt.lower()
    return {
        "django": ("django" in lower) or any(p.endswith("manage.py") or "django" in p for p in paths),
        "python": ("python" in lower or "flask" in lower or any(p.endswith(".py") for p in paths)
                   or "requirements.txt" in paths),
        "node": ("node" in lower or "react" in lower or "express" in lower
                 or "package.json" in paths or any(p.endswith(".js") or p.endswith(".jsx") for p in paths)),
        "docker": ("docker" in lower) or any(Path(p).name.lower() == "dockerfile" for p in paths),
        "has_src_app_py": ("src/app.py" in paths),
        "has_app_py": ("app.py" in paths),
        "has_manage_py": ("manage.py" in paths),
        "has_package_json": ("package.json" in paths),
    }

def next_steps(plan: Plan, prompt: str, after_apply: bool, output_root: Optional[Path] = None, port: int = DEFAULT_PORT) -> None:
    flags = detect_project(plan, prompt)
    url = f"http://localhost:{port}"
    open_cmd = get_os_open_cmd(url)

    print("\nNext steps to run the app (build/start):")
    if after_apply and output_root is not None:
        print(f"  cd {output_root}")

    if flags["django"]:
        print("  python -m venv .venv && . .venv/bin/activate")
        print("  pip install --upgrade pip")
        print("  pip install -r requirements.txt -c constraints.txt")
        print("  python manage.py migrate  # if migrations exist")
        print("  python manage.py runserver 0.0.0.0:{port}".format(port=port))
        print(f"  {open_cmd}")
    elif flags["python"]:
        entry = "src/app.py" if flags["has_src_app_py"] else ("app.py" if flags["has_app_py"] else "src/app.py")
        print("  python -m venv .venv && . .venv/bin/activate")
        print("  pip install --upgrade pip")
        print("  pip install -r requirements.txt -c constraints.txt")
        print(f"  python {entry}")
        print(f"  {open_cmd}")
    elif flags["node"]:
        if flags["has_package_json"]:
            print("  npm install")
            print("  npm start")
            print(f"  {open_cmd}")
        else:
            print("  npm init -y && npm install && npm start")
            print(f"  {open_cmd}")
    else:
        print("  # Review generated files and start the appropriate process")
        print(f"  {open_cmd}")

    if flags["docker"]:
        # Buildx hint for Apple Silicon targeting amd64
        is_arm_mac = (platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"))
        build_cmd = "docker build -t appforge-app ."
        if is_arm_mac:
            build_cmd = "docker buildx build --load --platform=linux/amd64 -t appforge-app ."
        print("\n  # Using Docker instead:")
        print(f"  {build_cmd}")
        print(f"  docker run --rm -p {port}:8000 appforge-app")
        print(f"  {open_cmd}")


# ---------- Doctor ----------
def cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def doctor(port: int) -> None:
    print("\nAppForge Doctor:")
    py = sys.version.replace("\n", " ")
    print(f"- Python: {py}")
    print(f"- Pip: {'found' if cmd_exists('pip') or cmd_exists('pip3') else 'missing'}")
    print(f"- Node: {'found' if cmd_exists('node') else 'missing'}")
    print(f"- NPM: {'found' if cmd_exists('npm') else 'missing'}")
    print(f"- Docker: {'found' if cmd_exists('docker') else 'missing'}")
    # Port check
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        print(f"- Port {port}: available")
    except OSError:
        print(f"- Port {port}: BUSY (AppForge will suggest a different port)")
    finally:
        try:
            s.close()
        except Exception:
            pass


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(prog="AppForge", description="Scaffold & edit code from natural language prompts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global
    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")

    # doctor
    p_doc = sub.add_parser("doctor", help="Run environment checks (Python/Node/Docker, port availability).")
    p_doc.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to check (default: 8000)")

    # plan
    p_plan = sub.add_parser("plan", help="Generate a plan and print to console (no files written).")
    p_plan.add_argument("prompt", nargs="+", help="Natural language instructions")
    p_plan.add_argument("--save-plan", default=None, help="Optional: save plan JSON to this path")
    p_plan.add_argument("--no-tree", action="store_true", help="Skip project tree preview")
    p_plan.add_argument("--port", type=int, default=None, help="Preferred app port (default: 8000)")

    # apply
    p_apply = sub.add_parser("apply", help="Apply a plan to disk (from a plan file or from a fresh prompt).")
    p_apply.add_argument("prompt", nargs="*", help="Optional: prompt to generate + apply immediately")
    p_apply.add_argument("--plan", default=None, help="Plan JSON file to apply (ignored if prompt is given)")
    p_apply.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    p_apply.add_argument("--force", action="store_true", help="Allow overwriting existing files")
    p_apply.add_argument("--output-path", default=None, help=f"Where to write files (default: ./{DEFAULT_OUTDIR})")
    p_apply.add_argument("--inplace", action="store_true", help="Write directly into --root (disables output dir)")
    p_apply.add_argument("--save-plan", default=None, help="Optional: save generated/loaded plan JSON to this path")
    p_apply.add_argument("--port", type=int, default=None, help="Preferred host port for local & Docker (default: 8000)")

    args = parser.parse_args()
    project_root = Path(args.root).resolve()

    if args.cmd == "doctor":
        doctor(args.port or DEFAULT_PORT)
        return

    # API key (needed for plan/apply)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    client = OpenAI(api_key=api_key)

    if args.cmd == "plan":
        user_prompt = " ".join(args.prompt)
        preferred_port = args.port or DEFAULT_PORT
        port = find_free_port(preferred_port)

        tree_preview = None if args.no_tree else make_tree_preview(project_root)
        raw = call_model(client, args.model, build_user_prompt(user_prompt, tree_preview))
        plan = ensure_baseline_files(parse_plan_json(raw), user_prompt, port)

        if args.save_plan:
            save_plan_to_file(plan, Path(args.save_plan))
            print(f"Plan saved to {args.save_plan}")

        print("\nPlanned files:")
        print(pretty_table([[fc.path, f"{len(fc.contents)} chars"] for fc in plan.files],
                           header=["path", "size"]))

        # Print concrete next steps
        next_steps(plan, user_prompt, after_apply=False, output_root=None, port=port)

    elif args.cmd == "apply":
        preferred_port = args.port or DEFAULT_PORT
        port = find_free_port(preferred_port)

        # Source the plan
        if args.prompt:
            user_prompt = " ".join(args.prompt)
            tree_preview = make_tree_preview(project_root)
            raw = call_model(client, args.model, build_user_prompt(user_prompt, tree_preview))
            plan = ensure_baseline_files(parse_plan_json(raw), user_prompt, port)
        else:
            if not args.plan:
                print("Error: --plan required if no prompt.", file=sys.stderr)
                sys.exit(2)
            with Path(args.plan).open("r", encoding="utf-8") as f:
                plan = parse_plan_json(json.load(f))
            user_prompt = "applied plan"
            # Even when loading an existing plan, enforce baselines/repairs & helpers
            plan = ensure_baseline_files(plan, user_prompt, port)

        # Resolve output root (default ./output)
        output_root = resolve_output_root(project_root, args.output_path or DEFAULT_OUTDIR, args.inplace)

        # Show intended actions (summary)
        rows = []
        for fc in plan.files:
            tgt = (output_root / fc.path).resolve()
            action = "write"
            if tgt.exists():
                action = "overwrite" if args.force else "skip"
            rows.append([fc.path, action, "exec" if fc.executable else "", f"{len(fc.contents)} chars"])
        print("\nApplying plan to:", output_root)
        print(pretty_table(rows, header=["path", "action (if exists)", "mode", "size"]))

        written = apply_plan(plan, output_root, dry_run=args.dry_run, force=args.force)
        if args.dry_run:
            print("\n(DRY RUN) No files were written.")
        else:
            print(f"\nDone. Wrote {len(written)} file(s) to {output_root}.")

        # Print concrete next steps (with cd to output dir)
        next_steps(plan, user_prompt, after_apply=True, output_root=output_root, port=port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
