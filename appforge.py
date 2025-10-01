#!/usr/bin/env python3
"""
AppForge — scaffold & edit project files via natural language prompts (OpenAI API).

Commands
--------
  plan   : Ask the model for a JSON "file plan" and print it to the console only.
           (No files are written. Optionally save the JSON with --save-plan.)
  apply  : Write files to disk from a plan file or from a fresh prompt.
           By default, writes under ./output (relative to --root).
           Override with --output-path=PATH or use --inplace to write into --root.

Examples
--------
  # 1) PLAN ONLY (no writes)
  python3 appforge.py plan "Create README.md and .gitignore for a Python app"

  # 2) Save the plan to a file (still no writes)
  python3 appforge.py plan "Flask app with /health and Dockerfile" --save-plan plan.json

  # 3) APPLY an existing plan file to ./output
  python3 appforge.py apply --plan plan.json

  # 4) One-shot: generate + apply immediately to ./output
  python3 appforge.py apply "Add a Makefile and LICENSE"

  # 5) Choose a different output directory
  python3 appforge.py apply --plan plan.json --output-path build_out

  # 6) Old behavior (write into project root directly)
  python3 appforge.py apply --plan plan.json --inplace

Requirements
------------
  pip install openai python-dotenv (optional)

Env
----
  OPENAI_API_KEY   : your API key
  OPENAI_MODEL     : override default model (optional)

"""

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict

# ---- OpenAI client (requires `openai>=1.40.0`) ----
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
    lines = []
    total = 0
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
- Use relative paths only (no absolute paths).
- Keep the plan focused and minimal but complete (no diffs).
- If a file should be executable (e.g., shell script), set executable=true.
- If overwriting an existing file is essential, set overwrite=true (user must pass --force to allow).
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
        try:
            return json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Model did not return valid JSON. Raw:\n{content}") from e


# ---------- Planning ----------
def parse_plan_json(raw: Dict[str, Any]) -> Plan:
    if not isinstance(raw, dict) or "files" not in raw:
        raise ValueError("Plan JSON missing 'files' array.")
    files: List[FileChange] = []
    for i, item in enumerate(raw["files"]):
        if not isinstance(item, dict):
            raise ValueError(f"files[{i}] is not an object")
        path = item.get("path")
        contents = item.get("contents")
        if not path or not isinstance(path, str):
            raise ValueError(f"files[{i}].path missing/invalid")
        if contents is None or not isinstance(contents, str):
            raise ValueError(f"files[{i}].contents missing/invalid")
        executable = bool(item.get("executable", False))
        overwrite = item.get("overwrite", None)
        if overwrite is not None:
            overwrite = bool(overwrite)
        files.append(FileChange(path=path, contents=contents, executable=executable, overwrite=overwrite))
    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        notes = None
    return Plan(files=files, notes=notes)

def save_plan_to_file(plan: Plan, path: Path) -> None:
    payload = {
        "files": [{"path": fc.path, "contents": fc.contents, "executable": fc.executable, "overwrite": fc.overwrite}
                  for fc in plan.files],
        "notes": plan.notes,
    }
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
            if fc.overwrite:
                print(f"[BLOCKED] Would overwrite '{fc.path}'. Re-run with --force to allow.")
                continue
            else:
                print(f"[SKIP] '{fc.path}' already exists in {output_root}. Use --force to replace.")
                continue

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as f:
                f.write(fc.contents)
            if fc.executable:
                mode = target.stat().st_mode
                os.chmod(target, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            written.append(target)
        print(f"{'[WRITE]' if not dry_run else '[DRY]'} {target.relative_to(output_root)}"
              f"{' (exec)' if fc.executable else ''}"
              f"{' (overwrite)' if exists and force else ''}")
    return written


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(prog="AppForge", description="Scaffold & edit code from natural language prompts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global flags
    parser.add_argument("--root", default=".", help="Project root for context (default: .)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")

    # plan
    p_plan = sub.add_parser("plan", help="Generate a plan and print to console (no files written).")
    p_plan.add_argument("prompt", nargs="+", help="Natural language instructions")
    p_plan.add_argument("--no-tree", action="store_true", help="Skip project tree preview")
    p_plan.add_argument("--save-plan", default=None, help="Optional: save plan JSON to this path")

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

    # Ensure API key for any model call
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
        plan = parse_plan_json(raw)

        # Print summary to console (no writes)
        if plan.notes:
            print("\nNotes:", plan.notes)

        rows = [[fc.path,
                 "yes" if fc.executable else "no",
                 "yes" if (fc.overwrite or False) else "no",
                 f"{len(fc.contents)} chars"]
                for fc in plan.files]
        print("\nPlanned files:")
        print(pretty_table(rows, header=["path", "executable", "overwrite?", "size"]))

        if args.save_plan:
            save_plan_to_file(plan, Path(args.save_plan))
            print(f"\nPlan saved to {args.save_plan}")

        print("\nNext:")
        print("  - To write files to ./output:      python3 appforge.py apply --plan <plan.json>")
        print("  - Or one-shot apply from prompt:   python3 appforge.py apply \"<your prompt>\"")

    elif args.cmd == "apply":
        # Source the plan
        if args.prompt:
            user_prompt = " ".join(args.prompt)
            tree_preview = make_tree_preview(project_root)
            raw = call_model(client, args.model, build_user_prompt(user_prompt, tree_preview))
            plan = parse_plan_json(raw)
            if args.save_plan:
                save_plan_to_file(plan, Path(args.save_plan))
                print(f"Generated plan -> {args.save_plan}")
        else:
            if not args.plan:
                print("Error: --plan is required when no prompt is provided.", file=sys.stderr)
                sys.exit(2)
            with Path(args.plan).open("r", encoding="utf-8") as f:
                plan = parse_plan_json(json.load(f))
            if args.save_plan:
                save_plan_to_file(plan, Path(args.save_plan))
                print(f"Re-saved plan -> {args.save_plan}")

        # Resolve output root (default ./output)
        output_root = resolve_output_root(project_root, args.output_path or DEFAULT_OUTDIR, args.inplace)

        # Show intended actions
        rows = []
        for fc in plan.files:
            tgt = (output_root / fc.path).resolve()
            action = "write"
            if tgt.exists():
                action = "overwrite" if args.force else "skip"
            rows.append([fc.path,
                         action,
                         "exec" if fc.executable else "",
                         f"{len(fc.contents)} chars",
                         str(tgt.parent.relative_to(output_root)) if within(output_root, tgt) else ""])
        print("\nApplying plan to:", output_root)
        print(pretty_table(rows, header=["path", "action (if exists)", "mode", "size", "dir"]))

        written = apply_plan(plan, output_root, dry_run=args.dry_run, force=args.force)
        if args.dry_run:
            print("\n(DRY RUN) No files were written.")
        else:
            print(f"\nDone. Wrote {len(written)} file(s) to {output_root}.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
