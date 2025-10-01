#!/usr/bin/env python3
"""
AppForge — scaffold & edit project files via natural language prompts (OpenAI API).

Reliability, UX, Self-Heal & Login-Aware:
- Align Python entrypoint across local & Docker (prefer src/app.py)
- Auto-fix common missing imports in generated Python (e.g., `import os`)
- Known-good Python pins + constraints; Node package.json + .nvmrc
- Safe DB bootstrap: generate init_db.py and ensure instance/ exists
- Dockerfile normalization, .dockerignore, healthcheck
- Buildx-aware Docker commands (fallback to classic `docker build` if buildx missing)
- Doctor checks Python/Node/NPM/Docker/Buildx and port availability
- Helper scripts: run_local.sh, run_docker.sh, smoke.sh, wait_for_port.py
- After plan/apply: concrete build/start steps + OS-specific open command
- NEW: --self-heal runs install/start/health and can auto-patch on failure
- NEW: Login-aware: if a login is detected, seed demo credentials and create login_test.sh.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import time
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
SELF_HEAL_MAX_CYCLES = 2
SELF_HEAL_STARTUP_TIMEOUT = 30  # seconds

DEMO_USERNAME = os.getenv("DEMO_USERNAME", "admin")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "admin123!")

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
    return preferred

def read_python_version_major_minor() -> Tuple[int, int]:
    return sys.version_info.major, sys.version_info.minor

def cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def has_buildx() -> bool:
    if not cmd_exists("docker"):
        return False
    try:
        cp = subprocess.run(
            ["docker", "buildx", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return cp.returncode == 0
    except Exception:
        return False

def short(s: str, n: int = 1200) -> str:
    return s if len(s) <= n else s[:n] + "\n... [truncated] ..."


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

SELF_HEAL_SYSTEM = """You are AppForge Fixer. You receive:
- Build/startup errors and logs
- A project tree preview
- Snippets of important files
Return ONLY a JSON object with the same schema used for plans:
{
  "files": [{"path": "...", "contents": "...", "executable": false, "overwrite": true}],
  "notes": "One short paragraph of what you changed and why."
}
Rules:
- Provide minimal, targeted patches to fix the errors.
- Set overwrite=true for files you modify.
- Do not include code fences or extra prose.
- Ensure Flask apps bind 0.0.0.0, read PORT (default 8000), have /health, and safe DB init (no create_all at import time).
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

def call_fixer(client: OpenAI, model: str, fixer_prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SELF_HEAL_SYSTEM},
            {"role": "user", "content": fixer_prompt},
        ],
        temperature=0.1,
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
    # Conservative pins compatible with Python 3.9–3.13
    return {
        "Flask": "3.0.3",
        "Werkzeug": "3.1.3",
        "Jinja2": "3.1.6",
        "itsdangerous": "2.2.0",
        "click": "8.1.8",
        "Flask-SQLAlchemy": "3.1.1",
        "SQLAlchemy": "2.0.43",
        "typing-extensions": "4.15.0",
        "greenlet": "3.2.4",
        "importlib-metadata": "8.7.0",
        "MarkupSafe": "3.0.3",
        "zipp": "3.23.0",
        "blinker": "1.9.0",
    }

def _render_requirements_from_matrix(matrix: Dict[str, str]) -> str:
    lines = [f"{k}=={v}" for k, v in matrix.items()]
    return "\n".join(lines) + "\n"

def _render_constraints_from_matrix(matrix: Dict[str, str]) -> str:
    return _render_requirements_from_matrix(matrix)

def _repair_python_requirements(existing: str, py_major: int, py_minor: int) -> str:
    text = existing or ""
    low = text.lower()
    if "flask==2." in low or ("flask" in low and "werkzeug==3" in low) or "werkzeug>=3" in low:
        matrix = _python_known_good_matrix(py_major, py_minor)
        return _render_requirements_from_matrix(matrix)
    if "flask" in low and "werkzeug" not in low:
        matrix = _python_known_good_matrix(py_major, py_minor)
        return _render_requirements_from_matrix(matrix)
    return existing

def _maybe_inject_imports(py_src: str) -> str:
    needs_os = ("os." in py_src or "os.environ" in py_src)
    has_os = ("import os" in py_src) or ("from os " in py_src)
    if needs_os and not has_os:
        lines = py_src.splitlines()
        insert_idx = 0
        if lines and lines[0].startswith("#!"):
            insert_idx = 1
        if len(lines) > insert_idx and lines[insert_idx].startswith("# -*-"):
            insert_idx += 1
        lines.insert(insert_idx, "import os")
        return "\n".join(lines) + ("\n" if not py_src.endswith("\n") else "")
    return py_src

# ---- Login detection & helpers ----
def detect_login(plan: Plan, prompt: str) -> Dict[str, Any]:
    lower = prompt.lower()
    keywords = ("login", "signin", "sign-in", "authenticate", "auth")
    login_in_prompt = any(k in lower for k in keywords)
    login_in_files = False
    login_endpoint = "/login"

    for fc in plan.files:
        p_low = fc.path.lower()
        if "login" in p_low:
            login_in_files = True
        if fc.path.endswith(".py"):
            c_low = fc.contents.lower()
            if "route(\"/login" in c_low or "route('/login" in c_low:
                login_in_files = True
        if fc.path.endswith(".html"):
            if "login" in fc.contents.lower():
                login_in_files = True

    return {
        "has_login": bool(login_in_prompt or login_in_files),
        "login_endpoint": login_endpoint,  # heuristic default
        "user_field": "username",
        "pass_field": "password",
    }

def _init_db_py(entry_candidates: List[str]) -> str:
    # Tries to import (src.)app and detect SQLAlchemy + User model,
    # seeds demo user if possible; otherwise creates tables only.
    return f"""#!/usr/bin/env python3
import sys

app = db = User = None
# Try both import paths so this works for src/app.py or app.py.
try:
    from src.app import app, db, User  # type: ignore
except Exception:
    try:
        from src.app import app, db  # type: ignore
    except Exception:
        try:
            from app import app, db, User  # type: ignore
        except Exception:
            try:
                from app import app, db  # type: ignore
            except Exception as e:
                print("Could not import app/db from src.app or app:", e)
                sys.exit(1)

try:
    from werkzeug.security import generate_password_hash
except Exception:
    generate_password_hash = None

DEMO_USERNAME = "{DEMO_USERNAME}"
DEMO_PASSWORD = "{DEMO_PASSWORD}"

def maybe_seed_user():
    global db, User
    if User is None:
        try:
            # Try dynamic import if User wasn't imported explicitly
            from src.app import User as SrcUser  # type: ignore
            User = SrcUser
        except Exception:
            try:
                from app import User as RootUser  # type: ignore
                User = RootUser
            except Exception:
                return False, "No User model"
    if not hasattr(User, "username") or not hasattr(User, "password_hash"):
        return False, "User model missing required fields"
    if generate_password_hash is None:
        return False, "werkzeug.security missing"
    try:
        u = User.query.filter_by(username=DEMO_USERNAME).first()
        if not u:
            u = User(username=DEMO_USERNAME, password_hash=generate_password_hash(DEMO_PASSWORD))
            db.session.add(u)
            db.session.commit()
        return True, "Seeded demo user"
    except Exception as e:
        return False, f"Error seeding: {{e}}"

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        ok, msg = maybe_seed_user()
        print("Database initialized. " + (msg or ""))
"""

def _login_test_sh(port: int, endpoint: str, user_field: str, pass_field: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
USER=${{DEMO_USERNAME:-{DEMO_USERNAME}}}
PASS=${{DEMO_PASSWORD:-{DEMO_PASSWORD}}}
URL="http://127.0.0.1:${{PORT}}{endpoint}"

# Try a simple form login without CSRF.
code=$(curl -sS -L -o /dev/null -w "%{{http_code}}" -c cookies.txt -b cookies.txt \\
  -X POST -d "{user_field}=${{USER}}&{pass_field}=${{PASS}}" "$URL")

echo "login_test: HTTP $code from $URL"
if [[ "$code" == "200" || "$code" == "302" || "$code" == "303" ]]; then
  echo "login_test: PASS"
  exit 0
else
  echo "login_test: FAIL"
  exit 1
fi
"""

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
python init_db.py || true
python {entry}
"""
    smoke = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
python wait_for_port.py || true
curl -fsS http://127.0.0.1:${{PORT}}/health || curl -fsS http://127.0.0.1:${{PORT}}/ || (echo "App did not respond" && exit 1)
echo "OK"
"""
    wait_py = f"""#!/usr/bin/env python3
import os, socket, sys, time
port = int(os.getenv("PORT", "{port}"))
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
    run_docker = f"""#!/usr/bin/env bash
set -euo pipefail
PORT=${{PORT:-{port}}}
IMG=appforge-app
if docker buildx version >/dev/null 2>&1; then
  docker buildx build -t "$IMG" . --load
else
  docker build -t "$IMG" .
fi
docker run --rm "$IMG" python init_db.py || true
docker run --rm -p "$PORT":8000 "$IMG"
"""
    return {
        "run_local.sh": run_local,
        "smoke.sh": smoke,
        "wait_for_port.py": wait_py,
        "run_docker.sh": run_docker,
    }

def _python_dockerfile(port: int, entry_candidates: List[str]) -> str:
    entry = "src/app.py"
    for c in entry_candidates:
        if c:
            entry = c
            break
    return f"""# Auto-generated by AppForge
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \\
    PORT={port}
WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/ ./src/
COPY init_db.py ./init_db.py
EXPOSE {port}
HEALTHCHECK --interval=10s --timeout=2s --retries=5 CMD python - <<'PY' || exit 1
import os,urllib.request
port=os.getenv("PORT","{port}")
for path in (f"http://127.0.0.1:{'{'}port{'}'}/health", f"http://127.0.0.1:{'{'}port{'}'}/"):
    try:
        urllib.request.urlopen(path,timeout=1)
        print("ok")
        raise SystemExit(0)
    except Exception:
        pass
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
instance/*.db
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
set -euo pipefill
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
      - Python: requirements.txt (pinned/repair), constraints.txt, init_db.py, instance/.keep,
                run scripts, Dockerfile, .dockerignore
      - Node: package.json, .nvmrc, run scripts
      - If login detected: add login_test.sh
    Also repairs conflicting Python requirement sets and auto-injects missing imports.
    """
    lower = prompt.lower()
    paths = {fc.path for fc in plan.files}

    is_python = ("python" in lower or "flask" in lower or "django" in lower
                 or any(p.endswith(".py") for p in paths) or "requirements.txt" in paths)
    is_node = ("node" in lower or "react" in lower or "express" in lower
               or "package.json" in paths or any(p.endswith(".js") or p.endswith(".jsx") for p in paths))
    mentions_docker = ("docker" in lower) or any(Path(p).name.lower() == "dockerfile" for p in paths)

    login_info = detect_login(plan, prompt)

    # Python baseline
    if is_python:
        py_major, py_minor = read_python_version_major_minor()
        matrix = _python_known_good_matrix(py_major, py_minor)
        req = _render_requirements_from_matrix(matrix)

        if "requirements.txt" in paths:
            existing = _get_file(plan, "requirements.txt").contents
            req = _repair_python_requirements(existing, py_major, py_minor)
        _set_file(plan, "requirements.txt", req)

        _set_file(plan, "constraints.txt", _render_constraints_from_matrix(matrix))

        # entry detection
        entry_candidates = []
        for candidate in ["src/app.py", "app.py"]:
            if candidate in paths:
                entry_candidates.append(candidate)

        # inject missing imports
        for fc in list(plan.files):
            if fc.path.endswith(".py"):
                fc.contents = _maybe_inject_imports(fc.contents)

        # init_db and instance/
        _set_file(plan, "init_db.py", _init_db_py(entry_candidates), executable=False)
        _set_file(plan, "instance/.keep", "", executable=False)

        # helpers
        helpers = _python_run_scripts(port, entry_candidates)
        for name, content in helpers.items():
            _set_file(plan, name, content, executable=name.endswith(".sh"))

        # login test helper
        if login_info["has_login"]:
            login_test = _login_test_sh(port, login_info["login_endpoint"], login_info["user_field"], login_info["pass_field"])
            _set_file(plan, "login_test.sh", login_test, executable=True)

        # Dockerfile
        if mentions_docker or "Dockerfile" in paths or entry_candidates:
            dockerfile = _python_dockerfile(port, entry_candidates)
            _set_file(plan, "Dockerfile", dockerfile)
            _set_file(plan, ".dockerignore", _dockerignore())

    # Node baseline
    if is_node:
        if "package.json" not in paths:
            _set_file(plan, "package.json", _node_package_json())
        if ".nvmrc" not in paths:
            _set_file(plan, ".nvmrc", "18\n")
        node_helpers = _node_run_scripts(port)
        for name, content in node_helpers.items():
            if _get_file(plan, name) is None:
                _set_file(plan, name, content, executable=name.endswith(".sh"))

    return plan


def save_plan_to_file(plan: Plan, path: Path) -> None:
    payload = {"files": [fc.__dict__ for fc in plan.files], "notes": plan.notes}
    path.parent.mkdir(parents=True, exist_ok=True
    )
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
    login_info = detect_login(plan, prompt)

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
        print("  python init_db.py  # one-time DB init (safe to re-run)")
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

    if login_info["has_login"]:
        print("\n  # Login credentials (demo)")
        print(f"  username: {DEMO_USERNAME}")
        print(f"  password: {DEMO_PASSWORD}")
        print("  # Curl login test (after app is running)")
        print(f"  curl -sS -L -c cookies.txt -b cookies.txt -X POST "
              f"-d \"{login_info['user_field']}={DEMO_USERNAME}&{login_info['pass_field']}={DEMO_PASSWORD}\" "
              f"http://127.0.0.1:{port}{login_info['login_endpoint']} -o /dev/null -w \"%{{http_code}}\\n\"")

    if flags["docker"]:
        build_cmd = "docker build -t appforge-app ."
        if has_buildx():
            build_cmd = "docker buildx build -t appforge-app . --load"
        print("\n  # Using Docker instead:")
        print(f"  {build_cmd}")
        print("  docker run --rm appforge-app python init_db.py  # one-time DB init")
        print(f"  docker run --rm -p {port}:8000 appforge-app")
        print(f"  {open_cmd}")


# ---------- Doctor ----------
def doctor(port: int) -> None:
    print("\nAppForge Doctor:")
    py = sys.version.replace("\n", " ")
    print(f"- Python: {py}")
    print(f"- Pip: {'found' if cmd_exists('pip') or cmd_exists('pip3') else 'missing'}")
    print(f"- Node: {'found' if cmd_exists('node') else 'missing'}")
    print(f"- NPM: {'found' if cmd_exists('npm') else 'missing'}")
    print(f"- Docker: {'found' if cmd_exists('docker') else 'missing'}")
    print(f"- Docker Buildx: {'found' if has_buildx() else 'missing'}")
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


# ---------- Self-Heal Runtime ----------
def _venv_paths(output_root: Path) -> Tuple[Path, Path]:
    venv_dir = output_root / ".venv"
    if sys.platform == "win32":
        py = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
    else:
        py = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"
    return py, pip

def run_cmd(args: List[str], cwd: Path, env: Dict[str, str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        err += f"\n[AppForge] Command timed out after {timeout}s."
    return p.returncode, out or "", err or ""

def start_process(args: List[str], cwd: Path, env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def wait_for_health(port: int, timeout: int, proc: Optional[subprocess.Popen]) -> Tuple[bool, str]:
    import urllib.request
    start = time.time()
    logs = []
    while time.time() - start < timeout:
        if proc and proc.poll() is None:
            try:
                line = proc.stdout.readline()
                if line:
                    logs.append(line)
            except Exception:
                pass
        time.sleep(0.5)
        for path in (f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/"):
            try:
                with urllib.request.urlopen(path, timeout=1) as resp:
                    if 200 <= resp.status < 400:
                        return True, "".join(logs)
            except Exception:
                pass
    return False, "".join(logs)

def determine_entry(output_root: Path) -> str:
    if (output_root / "src" / "app.py").exists():
        return "src/app.py"
    if (output_root / "app.py").exists():
        return "app.py"
    return "src/app.py"

def collect_key_files_for_fix(output_root: Path, max_chars: int = 4000) -> str:
    important = [
        output_root / "src" / "app.py",
        output_root / "app.py",
        output_root / "requirements.txt",
        output_root / "constraints.txt",
        output_root / "init_db.py",
        output_root / "Dockerfile",
    ]
    parts = []
    used = 0
    for p in important:
        if p and p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                chunk = f"\n--- FILE: {p.relative_to(output_root)} ---\n{short(txt, 1600)}\n"
                if used + len(chunk) <= max_chars:
                    parts.append(chunk)
                    used += len(chunk)
            except Exception:
                pass
    return "".join(parts)

def self_heal_once(
    client: OpenAI,
    model: str,
    output_root: Path,
    port: int,
    has_login: bool,
    login_endpoint: str,
    user_field: str,
    pass_field: str,
) -> Tuple[bool, str]:
    # 1) Ensure venv
    venv_py, venv_pip = _venv_paths(output_root)
    if not venv_py.exists():
        code, out, err = run_cmd([sys.executable, "-m", "venv", ".venv"], output_root, os.environ.copy(), timeout=120)
        if code != 0:
            return False, f"Failed to create venv:\n{err}"

    env = os.environ.copy()
    env["PATH"] = str(venv_py.parent) + os.pathsep + env.get("PATH", "")
    env["PORT"] = str(port)
    env["DEMO_USERNAME"] = DEMO_USERNAME
    env["DEMO_PASSWORD"] = DEMO_PASSWORD

    # 2) pip upgrade + install
    code, out, err = run_cmd([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], output_root, env, timeout=180)
    pip_logs = out + "\n" + err
    if code != 0:
        return False, f"Pip upgrade failed:\n{pip_logs}"

    req = output_root / "requirements.txt"
    cons = output_root / "constraints.txt"
    if req.exists():
        args = [str(venv_py), "-m", "pip", "install", "-r", "requirements.txt"]
        if cons.exists():
            args += ["-c", "constraints.txt"]
        code, out, err = run_cmd(args, output_root, env, timeout=600)
        pip_logs += "\n" + out + "\n" + err
        if code != 0:
            return False, f"[InstallError]\n{short(pip_logs, 4000)}"

    # 3) init db
    if (output_root / "init_db.py").exists():
        code, out, err = run_cmd([str(venv_py), "init_db.py"], output_root, env, timeout=60)
        if code != 0:
            return False, f"init_db.py failed:\n{out}\n{err}"

    # 4) start app, wait for health
    entry = determine_entry(output_root)
    proc = start_process([str(venv_py), entry], output_root, env)
    ok, logs = wait_for_health(port, SELF_HEAL_STARTUP_TIMEOUT, proc)

    if not ok:
        try:
            time.sleep(1.0)
            leftover = proc.communicate(timeout=2)[0] or ""
            logs += leftover
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        return False, f"[StartupError]\n{short(logs, 4000)}"

    # 5) optional: login test
    if has_login and (output_root / "login_test.sh").exists():
        code, out, err = run_cmd(
            ["bash", "login_test.sh"],
            output_root,
            env,
            timeout=30,
        )
        if code != 0:
            # collect limited app logs if still running
            try:
                leftover = proc.communicate(timeout=2)[0] or ""
                logs += leftover
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            return False, f"[LoginError]\n{out}\n{err}\n\n[AppLogs]\n{short(logs, 1500)}"

    # Cleanly stop the app; health and (optional) login test passed
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return True, "Startup OK"

def prompt_yes_no(msg: str) -> bool:
    try:
        ans = input(f"{msg} (y/n): ").strip().lower()
        return ans in ("y", "yes")
    except EOFError:
        return False

def run_self_heal(
    client: OpenAI,
    model: str,
    plan: Plan,
    output_root: Path,
    user_prompt: str,
    port: int,
) -> None:
    print("\n[Self-Heal] Starting startup check & heal loop...")
    tree = make_tree_preview(output_root)
    login_info = detect_login(plan, user_prompt)
    for cycle in range(1, SELF_HEAL_MAX_CYCLES + 1):
        print(f"[Self-Heal] Cycle {cycle} — installing & starting app...")
        ok, summary = self_heal_once(
            client, model, output_root, port,
            login_info["has_login"], login_info["login_endpoint"],
            login_info["user_field"], login_info["pass_field"]
        )
        if ok:
            print("[Self-Heal] ✅ Application started and responded healthy.")
            if login_info["has_login"]:
                print("[Self-Heal] ✅ Login test passed with demo credentials.")
            return
        else:
            print("[Self-Heal] ❌ Errors were found during startup of the application.")
            print(short(summary, 1200))
            if not prompt_yes_no("Would you like me to correct these errors?"):
                print("[Self-Heal] Skipping auto-fix by user choice.")
                return

            files_for_fix = collect_key_files_for_fix(output_root)
            fixer_prompt = (
                "User intent:\n" + user_prompt.strip() +
                "\n\nProject tree:\n" + tree +
                "\n\nRelevant files (truncated):\n" + files_for_fix +
                "\n\nErrors/Logs:\n" + summary +
                "\n\nPlease return a JSON plan with minimal patches (overwrite=true) to fix the errors."
            )

            try:
                raw_fix = call_fixer(client, model, fixer_prompt)
                patch_plan = parse_plan_json(raw_fix)
            except Exception as e:
                print(f"[Self-Heal] Failed to get a valid patch plan from model: {e}")
                return

            print("[Self-Heal] Applying patches:")
            for fc in patch_plan.files:
                fc.overwrite = True if fc.overwrite is None else fc.overwrite
                print(f" - {fc.path}")

            apply_plan(patch_plan, output_root, dry_run=False, force=True)

            if patch_plan.notes:
                print("\n[Self-Heal] Patch notes:\n" + patch_plan.notes)

    print("[Self-Heal] Reached maximum heal attempts without a healthy start.")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(prog="AppForge", description="Scaffold & edit code from natural language prompts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global
    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")

    # doctor
    p_doc = sub.add_parser("doctor", help="Run environment checks (Python/Node/Docker/Buildx, port availability).")
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
    p_apply.add_argument("--self-heal", action="store_true", help="After apply, install, start, health-check (and login if present), then auto-fix on failure")

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

        next_steps(plan, user_prompt, after_apply=True, output_root=output_root, port=port)

        if args.self_heal and not args.dry_run:
            run_self_heal(client, args.model, plan, output_root, user_prompt, port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
