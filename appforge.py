#!/usr/bin/env python3
"""
AppForge — scaffold & edit project files via natural language prompts (OpenAI API).

Enhancements in this version:
- Ensures Python projects always include requirements.txt
- Ensures Node.js projects always include package.json
- After each plan/apply, suggests commands to build/run the app
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
    """Guarantee requirements.txt for Python, package.json for Node.js."""
    lower = prompt.lower()
    paths = {fc.path for fc in plan.files}
    if "python" in lower or "flask" in lower or "django" in lower:
        if "requirements.txt" not in paths:
            plan.files.append(FileChange(
                path="requirements.txt",
                contents="flask\ndjango\n# add other deps here\n"
            ))
    if "node" in lower or "react" in lower or "express" in lower:
        if "package.json" not in paths:
            plan.files.append(FileChange(
                path="package.json",
                contents=json.dumps({
                    "name": "appforge-project",
                    "version": "1.0.0",
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


# ---------- CLI ----------
def suggest_commands(prompt: str):
    lower = prompt.lower()
    if "python" in lower or "flask" in lower or "django" in lower:
        print("\n👉 Suggested next steps:")
        print("   pip install -r requirements.txt")
        print("   python app.py   # or manage.py runserver (for Django)")
    elif "node" in lower or "react" in lower or "express" in lower:
        print("\n👉 Suggested next steps:")
        print("   npm install")
        print("   npm start")

def main():
    parser = argparse.ArgumentParser(prog="AppForge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser.add_argument("--root", default=".", help="Project root (default: .)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("prompt", nargs="+")
    p_plan.add_argument("--save-plan", default=None)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("prompt", nargs="*")
    p_apply.add_argument("--plan", default=None)
    p_apply.add_argument("--dry-run", action="store_true")
    p_apply.add_argument("--force", action="store_true")
    p_apply.add_argument("--output-path", default=None)
    p_apply.add_argument("--inplace", action="store_true")
    p_apply.add_argument("--save-plan", default=None)

    args = parser.parse_args()
    project_root = Path(args.root).resolve()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    client = OpenAI(api_key=api_key)

    if args.cmd == "plan":
        user_prompt = " ".join(args.prompt)
        model_prompt = build_user_prompt(user_prompt, make_tree_preview(project_root))
        raw = call_model(client, args.model, model_prompt)
        plan = ensure_baseline_files(parse_plan_json(raw), user_prompt)
        if args.save_plan:
            save_plan_to_file(plan, Path(args.save_plan))
            print(f"Plan saved to {args.save_plan}")
        print("\nPlanned files:")
        print(pretty_table([[fc.path, str(len(fc.contents))+" chars"] for fc in plan.files],
                          header=["path","size"]))
        suggest_commands(user_prompt)

    elif args.cmd == "apply":
        if args.prompt:
            user_prompt = " ".join(args.prompt)
            raw = call_model(client, args.model, build_user_prompt(user_prompt, make_tree_preview(project_root)))
            plan = ensure_baseline_files(parse_plan_json(raw), user_prompt)
        else:
            if not args.plan:
                print("Error: --plan required if no prompt.", file=sys.stderr)
                sys.exit(2)
            with Path(args.plan).open("r") as f:
                plan = parse_plan_json(json.load(f))
            user_prompt = "applied plan"

        output_root = resolve_output_root(project_root, args.output_path, args.inplace)
        written = apply_plan(plan, output_root, args.dry_run, args.force)
        if args.dry_run:
            print("\n(DRY RUN) No files written.")
        else:
            print(f"\nDone. Wrote {len(written)} files to {output_root}")
        suggest_commands(user_prompt)

if __name__ == "__main__":
    main()
