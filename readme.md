# AppForge

AppForge is a command-line tool that lets you use **natural language prompts** to scaffold and edit project files.  
It leverages the OpenAI API to generate project structures, config files, and boilerplate code — then safely writes them to disk.

By default, files are written under `./output/`, so your main project tree stays untouched until you’re ready.  
You can override with `--output-path` or use `--inplace` to write directly into your project root.

---

## ✨ What’s new (reliability & DX)

- ✅ **Build reliability for Python & Node**
  - Auto-generate **`requirements.txt`** with **known-good pins** and **`constraints.txt`** for reproducible installs
  - Auto-generate **`package.json`** (+ `.nvmrc`) with usable `start` script and Node ≥ 18
- 🐍 **Python web defaults**
  - Adds `/health` route, binds to `0.0.0.0`, honors `PORT` (default **8000**)
  - Creates `run_local.sh`, `smoke.sh`, and `wait_for_port.py`
- 🐳 **Docker resilience**
  - Emits a cache-friendly **Dockerfile** (Python), **.dockerignore**, and container **HEALTHCHECK**
  - Prints Buildx & platform hints on Apple Silicon
- 🩺 **Preflight**
  - `doctor` command: checks Python/Node/NPM/Docker availability and port status
- 🧭 **Run UX**
  - After **plan** and **apply**, prints **Next steps to run the app (build/start)** +
    **OS-specific open command** (`open` / `start` / `xdg-open`) to `http://localhost:<port>`
  - Detects port conflicts and picks the next free port (8001…​)

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
- 📦 Dependency hygiene:
  - Python: `requirements.txt` **and** `constraints.txt`
  - Node: `package.json` + `.nvmrc`
- 🐳 Docker ergonomics:
  - Dockerfile with `pip` upgrade, pinned installs, app healthcheck, and expose 8000.
- 🔍 Preflight:
  - `doctor` validates environment & ports.
- 📂 Optional `--save-plan` to persist JSON plans.
- 📑 Automatic summary tables for planned changes.

---

## 📦 Installation

Copy `appforge.py` into your repo and make it executable:

```bash
chmod +x appforge.py
````

Install dependencies with your project’s `requirements.txt` (for AppForge itself):

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

## 🩺 Preflight

Check your environment and port availability (default 8000):

```bash
python3 appforge.py doctor
# or specify a port
python3 appforge.py doctor --port 8001
```

---

## 🚀 Usage

### 1) Plan (no writes)

```bash
python3 appforge.py plan "Scaffold a Flask web app with app.py, requirements.txt, and Dockerfile. Add /health and bind 0.0.0.0 on PORT." --save-plan plan.json
```

* Prints a table of proposed files (including `requirements.txt`, `constraints.txt`, helpers, Dockerfile).
* Saves JSON plan to `plan.json`.
* Prints **Next steps to run the app (build/start)** including an **open** command to your localhost URL.

### 2) Apply

Write the plan’s files to `./output`:

```bash
python3 appforge.py apply --plan plan.json
```

Apply directly from a fresh prompt:

```bash
python3 appforge.py apply "Create a Django app with manage.py, settings, urls, wsgi, and a /health endpoint"
```

After apply, AppForge prints:

* `cd ./output`
* Exact install & start commands (Python/Node)
* Docker build/run alternatives
* An **open** command to launch your browser (`http://localhost:<port>`)

---

## 🔧 Options

| Option               | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `--root`             | Project root for context (default `.`)                         |
| `--output-path`      | Custom output dir (default `./output/`)                        |
| `--inplace`          | Write directly into project root                               |
| `--dry-run`          | Preview without writing                                        |
| `--force`            | Allow overwriting existing files                               |
| `--save-plan <file>` | Save generated/loaded plan JSON                                |
| `--port <n>`         | Preferred port (AppForge finds a free one starting from `<n>`) |

Run `python3 appforge.py --help` for full details.

---

## 🌱 Example Prompts

### Start a Flask Web Project (with healthcheck + Docker)

```bash
python3 appforge.py apply "Create a Flask app with:
- src/app.py exposing / and /health, bind 0.0.0.0 and PORT env (default 8000)
- requirements.txt and constraints.txt using Flask 3 + Werkzeug 3 pins
- Dockerfile with healthcheck and .dockerignore
- templates/login.html and templates/dashboard.html
- README.md with run instructions"
```

**Then run (shown by AppForge after apply):**

```bash
cd output
python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
python src/app.py
# macOS: open http://localhost:8000
# Linux: xdg-open http://localhost:8000
# WinPS: start http://localhost:8000
```

**Docker alternative (also shown after apply):**

```bash
docker build -t appforge-app .
docker run --rm -p 8000:8000 appforge-app
# open http://localhost:8000
```

### Start a Django Web Project

```bash
python3 appforge.py apply "Scaffold a Django project called farmapp with:
- manage.py
- farmapp/settings.py, urls.py (add /health), wsgi.py
- requirements.txt + constraints.txt with pinned versions
- Dockerfile (+ .dockerignore)
- README with run commands"
```

**Run (AppForge prints exact commands):**

```bash
cd output
python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
# open http://localhost:8000
```

### Add Frontend (React + Vite)

```bash
python3 appforge.py apply "Add a React + Vite frontend with:
- package.json (start script), vite.config.js
- src/App.jsx with a health route call
- .nvmrc set to Node 18
- README with npm commands"
```

**Run (AppForge prints exact commands):**

```bash
cd output
npm install
npm start
# open http://localhost:8000
```

### Add CI/CD

```bash
python3 appforge.py apply "Add a GitHub Actions workflow at .github/workflows/ci.yml
that installs dependencies, runs lint/test, builds Docker image, and pushes to GHCR"
```

---

## 🧪 Helpers Generated by AppForge

| File               | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `requirements.txt` | Pinned, known-good versions for Python apps              |
| `constraints.txt`  | Locks transitive deps for reproducible installs          |
| `package.json`     | Node app scaffold with `start` script and engines        |
| `.nvmrc`           | Sets Node version (18) for consistency                   |
| `run_local.sh`     | One-command local startup (Python or Node)               |
| `smoke.sh`         | Simple curl-based health check                           |
| `wait_for_port.py` | Small helper to wait for a local port to open (Python)   |
| `Dockerfile`       | Cache-friendly Dockerfile with healthcheck (Python apps) |
| `.dockerignore`    | Sensible defaults to shrink Docker build context         |

> All shell helpers are created **executable**.

---

## 🛡️ Notes

* Always **review plans** with `plan` before applying.
* Plans are saved as JSON and can be version-controlled.
* By default, all files go to `./output/`.
* To overwrite existing files, add `--force`.
* On Apple Silicon, AppForge may print a `buildx` command with `--platform=linux/amd64` when appropriate.

---