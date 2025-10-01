#!/usr/bin/env python3
"""
AppForge — scaffold & edit project files via natural language prompts (OpenAI API).

Enhancements in this version:
- Ensures Python projects always include requirements.txt
- Ensures Node.js projects always include package.json
- After each plan/apply, prints concrete "Next steps to run the app (build/start)" commands
"""

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict

# ---- OpenAI client ----
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OUTDIR = "output"

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
- Always include requirements.txt for Python projects.
- Always include package.json for Node.js projects.
- Use relative paths only.
- Prefer complete, minimal working examples.
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

def ensure_baseline_files(plan: Plan, prompt: str) -> Plan:
    """Guarantee requirements.txt for Python, package.json for Node.js (if missing)."""
    lower = prompt.lower()
    paths = {fc.path for fc in plan.files}

    is_python = ("python" in lower or "flask" in lower or "django" in lower
                 or any(p.endswith(".py") for p in paths) or "requirements.txt" in paths)
    is_node = ("node" in lower or "react" in lower or "express" in lower
               or "package.json" in paths or any(p.endswith(".js") or p.endswith(".jsx") for p in paths))

    if is_python and "requirements.txt" not in paths:
        plan.files.append(FileChange(
            path="requirements.txt",
            contents="flask>=3.0.0\n# django>=5.0.0  # uncomment if using Django\n"
        ))
    if is_node and "package.json" not in paths:
        plan.files.append(FileChange(
            path="package.json",
            contents=json.dumps({
                "name": "appforge-project",
                "version": "1.0.0",
                "private": True,
                "scripts": {"start": "node index.js"},
                "dependencies": {}
            }, indent=2)
        ))
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


# ---------- Next steps helper ----------
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
        "has_src_app_py": ("app.py" in paths or "src/app.py" in paths),
        "has_manage_py": ("manage.py" in paths),
        "has_package_json": ("package.json" in paths),
    }

def next_steps(plan: Plan, prompt: str, after_apply: bool, output_root: Optional[Path] = None) -> None:
    flags = detect_project(plan, prompt)
    print("\nNext steps to run the app (build/start):")
    if after_apply and output_root is not None:
        print(f"  cd {output_root}")

    if flags["django"]:
        print("  pip install -r requirements.txt")
        print("  python manage.py migrate  # if migrations exist")
        print("  python manage.py runserver")
    elif flags["python"]:
        print("  pip install -r requirements.txt")
        # Prefer src/app.py if present
        entry = "src/app.py" if flags["has_src_app_py"] else "app.py"
        print(f"  python {entry}")
    elif flags["node"]:
        if flags["has_package_json"]:
            print("  npm install")
            print("  npm start  # or: node index.js")
        else:
            print("  # Initialize node project if needed:")
            print("  npm init -y && npm install && npm start")
    else:
        print("  # Review generated files and run using the appropriate toolchain.")

    if flags["docker"]:
        print("\n  # Using Docker instead:")
        print("  docker build -t appforge-app .")
        print("  docker run --rm -p 8000:8000 appforge-app")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(prog="AppForge", description="Scaffold & edit code from natural language prompts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global
    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")

    # plan
    p_plan = sub.add_parser("plan", help="Generate a plan and print to console (no files written).")
    p_plan.add_argument("prompt", nargs="+", help="Natural language instructions")
    p_plan.add_argument("--save-plan", default=None, help="Optional: save plan JSON to this path")
    p_plan.add_argument("--no-tree", action="store_true", help="Skip project tree preview")

    # apply
    p_apply = sub.add_parser("apply", help="Apply a plan to disk (from a plan file or from a fresh prompt).")
    p_apply.add_argument("prompt", nargs="*", help="Optional: prompt to generate + apply immediately")
    p_apply.add_argument("--plan", default=None, help="Plan JSON file to apply (ignored if prompt is given)")
    p_apply.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    p_apply.add_argument("--force", action="store_true", help="Allow overwriting existing files")
    p_apply.add_argument("--output-path", default=None, help=f"Where to write files (default: ./{DEFAULT_OUTDIR})")
    p_apply.add_argument("--inplace", action="store_true", help="Write directly into --root (disables output dir)")
    p_apply.add_argument("--save-plan", default=None, help="Optional: save generated/loaded plan JSON to this path")

    args = parser.parse_args()
    project_root = Path(args.root).resolve()

    # API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    client = OpenAI(api_key=api_key)

    if args.cmd == "plan":
        user_prompt = " ".join(args.prompt)
        tree_preview = None if args.no_tree else make_tree_preview(project_root)
        model_prompt = build_user_prompt(user_prompt, tree_preview)
        raw = call_model(client, args.model, model_prompt)
        plan = ensure_baseline_files(parse_plan_json(raw), user_prompt)

        if args.save_plan:
            save_plan_to_file(plan, Path(args.save_plan))
            print(f"Plan saved to {args.save_plan}")

        print("\nPlanned files:")
        print(pretty_table([[fc.path, f"{len(fc.contents)} chars"] for fc in plan.files],
                           header=["path", "size"]))

        # Print concrete next steps
        next_steps(plan, user_prompt, after_apply=False, output_root=None)

    elif args.cmd == "apply":
        # Source the plan
        if args.prompt:
            user_prompt = " ".join(args.prompt)
            tree_preview = make_tree_preview(project_root)
            raw = call_model(client, args.model, build_user_prompt(user_prompt, tree_preview))
            plan = ensure_baseline_files(parse_plan_json(raw), user_prompt)
        else:
            if not args.plan:
                print("Error: --plan required if no prompt.", file=sys.stderr)
                sys.exit(2)
            with Path(args.plan).open("r", encoding="utf-8") as f:
                plan = parse_plan_json(json.load(f))
            user_prompt = "applied plan"

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
        next_steps(plan, user_prompt, after_apply=True, output_root=output_root)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
