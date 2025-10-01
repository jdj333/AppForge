# AppForge

AppForge is a command-line tool that lets you use **natural language prompts** to scaffold and edit project files.  
It leverages the OpenAI API to generate project structures, config files, and boilerplate code — then safely writes them to disk.

By default, files are written under `./output/`, so your main project tree stays untouched until you’re ready.  
You can override with `--output-path` or use `--inplace` to write directly into your project root.

---

## Features

- 🔑 Uses your `OPENAI_API_KEY` (never hardcoded).
- 📝 **Plan / Apply workflow**:
  - `plan` → generate and preview a JSON plan (no writes).
  - `apply` → write files to disk (from a saved plan or a prompt).
- 🛡️ Safety:
  - Defaults to `./output/`.
  - Prevents path traversal.
  - Skips overwrites unless you pass `--force`.
- 📂 Optional `--save-plan` to persist JSON plans.
- 📑 Automatic summary tables for planned changes.

---

## 📦 Installation

Clone the script or copy `appforge.py` into your repo:

```bash
chmod +x appforge.py
````

Install dependencies with `requirements.txt`:

```txt
openai>=1.40.0
python-dotenv>=1.0.1
rich>=13.7.0
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

(Windows PowerShell)

```powershell
setx OPENAI_API_KEY "sk-..."
```

---

## 🚀 Usage

### 1. Plan (no writes)

```bash
python3 appforge.py plan "Scaffold a Flask web app with app.py, requirements.txt, and Dockerfile" --save-plan plan.json
```

This prints a table of proposed files and saves the JSON plan to `plan.json`.

### 2. Apply

Write the plan’s files to `./output`:

```bash
python3 appforge.py apply --plan plan.json
```

Apply directly from a fresh prompt:

```bash
python3 appforge.py apply "Create a Django app with models.py, views.py, urls.py, and settings.py"
```

---

## 🔧 Options

* `--root` : project root (default `.`) for tree context
* `--output-path` : custom output dir (default `./output/`)
* `--inplace` : write directly into project root
* `--dry-run` : preview without writing
* `--force` : allow overwriting existing files
* `--save-plan <file>` : save generated/loaded plan JSON

Run `python3 appforge.py --help` for full details.

---

## 🌱 Example Prompts

### Start a Flask Web Project

```bash
python3 appforge.py apply "Create a Flask app with:
- app.py with a /health endpoint
- requirements.txt
- Dockerfile
- README.md with run instructions"
```

### Start a Django Web Project

```bash
python3 appforge.py apply "Scaffold a Django project called farmapp with:
- manage.py
- farmapp/settings.py, urls.py, wsgi.py
- Dockerfile
- requirements.txt
- a 'fields' app with models.py for Field(name, acres)"
```

### Add Frontend (React + Vite)

```bash
python3 appforge.py apply "Add a React + Vite frontend with:
- package.json
- src/App.jsx
- vite.config.js
- README.md for setup"
```

### Add CI/CD

```bash
python3 appforge.py apply "Add a GitHub Actions workflow at .github/workflows/ci.yml
that runs pytest, builds Docker image, and pushes to GHCR"
```

---

## 🛡️ Notes

* Always **review plans** with `plan` before applying.
* Plans are saved as JSON and can be version-controlled.
* By default, all files go to `./output/`.
* To overwrite existing files, add `--force`.

---

## 📄 License

MIT

