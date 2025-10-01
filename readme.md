# AppForge

AppForge is a command-line tool that lets you use **natural language prompts** to scaffold and edit project files.  
It leverages the OpenAI API to generate project structures, config files, and boilerplate code — then safely writes them to disk.  
Now with **self-healing startup checks** and **login verification** for generated apps.

By default, files are written under `./output/`, so your main project tree stays untouched until you’re ready.  
You can override with `--output-path` or use `--inplace` to write directly into your project root.

---

## ✨ What’s new (reliability, UX & self-heal)

- ✅ **Build reliability for Python & Node**
  - Auto-generate **`requirements.txt`** with **known-good pins** and **`constraints.txt`** for reproducible installs
  - Auto-generate **`package.json`** (+ `.nvmrc`) with usable `start` script and Node ≥ 18
- 🐍 **Python web defaults**
  - Adds `/health` route, binds to `0.0.0.0`, honors `PORT` (default **8000**)
  - Creates `run_local.sh`, `smoke.sh`, `wait_for_port.py`, and safe `init_db.py`
- 🔑 **Login detection**
  - If a prompt requests login/auth, AppForge scaffolds a **default login** (HTML form + backend)
  - Generates **default test credentials** (e.g., `admin` / `password123`)
  - Self-heal runs a **login test via curl** after startup to verify it works
- 🐳 **Docker resilience**
  - Emits a cache-friendly **Dockerfile** (Python), **.dockerignore**, and container **HEALTHCHECK**
  - Prints Buildx & platform hints on Apple Silicon
- 🩺 **Preflight**
  - `doctor` command: checks Python/Node/NPM/Docker availability and port status
- 🔁 **Self-heal**
  - New `--self-heal` flag: after apply, AppForge:
    1. Installs dependencies  
    2. Starts the app  
    3. Runs `/health` check (and `/login` if present)  
    4. If errors are found, prompts you to auto-patch  
    5. Applies patches, retries up to 2 cycles
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
- 🔁 Self-healing startup check:
  - Installs, runs, and tests health/login endpoints automatically.
  - Proposes auto-fixes for errors.
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
python3 appforge.py plan "Scaffold a Flask web app with login and dashboard, /health, and Docker support" --save-plan plan.json
```

* Prints a table of proposed files.
* Saves JSON plan to `plan.json`.
* Prints **Next steps to run the app (build/start)**.

### 2) Apply

Apply a plan’s files to `./output`:

```bash
python3 appforge.py apply --plan plan.json
```

Apply directly from a fresh prompt:

```bash
python3 appforge.py apply "Create a Flask app with login page, /health endpoint, and dashboard route"
```

Apply with **self-heal**:

```bash
python3 appforge.py apply "Create a Flask app with login page and dashboard" --self-heal
```

AppForge will then:

* Create a virtualenv
* Install dependencies
* Start the app
* Check `/health`
* If a login is present:

  * Print generated **username/password**
  * Run a test `curl -X POST` against `/login`
* If startup or login fails, it offers to auto-fix and retry.

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
| `--self-heal`        | Run install/start/health/login tests and auto-fix errors       |

Run `python3 appforge.py --help` for full details.

---

## 🌱 Example Prompts

### Flask App with Login

```bash
python3 appforge.py apply "Create a Flask app with:
- src/app.py with /health, /login, /dashboard
- login.html template (POST form with username/password)
- default credentials admin/password123
- requirements.txt + constraints.txt
- Dockerfile with healthcheck
- README with run instructions"
--self-heal
```

AppForge will generate the app, install dependencies, start it,
print the **login credentials**, and run a **curl login test** automatically.

### Django App with Auth

```bash
python3 appforge.py apply "Scaffold a Django project with:
- manage.py
- settings.py, urls.py with /health
- built-in Django auth (username/password)
- requirements.txt + constraints.txt
- Dockerfile"
--self-heal
```

Self-heal runs `/health` and a login test using the default superuser.

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
| `init_db.py`       | Safe DB initialization script (for Flask/Django apps)    |
| `Dockerfile`       | Cache-friendly Dockerfile with healthcheck (Python apps) |
| `.dockerignore`    | Sensible defaults to shrink Docker build context         |

---

## 🛡️ Notes

* Always **review plans** with `plan` before applying.
* Plans are saved as JSON and can be version-controlled.
* By default, all files go to `./output/`.
* To overwrite existing files, add `--force`.
* On Apple Silicon, AppForge may print a `buildx` command with `--platform=linux/amd64` when appropriate.
* If login is included, AppForge prints generated **credentials** and verifies login with a `curl` POST.

