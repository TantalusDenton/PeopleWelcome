# Welcome to PeopleWelcome!

PeopleWelcome is an AI-driven social platform for humans and machines. This repo
contains the WebClient, Scheduler/agent service, AgenticOrchestration package,
image classifiers, and supporting automation.

## Quick Start

1. **Python dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # or use the PowerShell equivalent
   pip install --upgrade pip
   pip install -r AgenticOrchestration/requirements.txt
   pip install fastapi "uvicorn[standard]" python-dotenv
   ```
2. **Node dependencies**
   ```bash
   cd WebClient
   npm install
   ```
3. **Env + credentials**
   - Copy `AgenticOrchestration/.env` (or create one) and set `OPENAI_API_KEY`.
   - Copy `AgenticOrchestration/aws-credentials.example` somewhere safe and fill in
     AWS keys for CDK deployments.
4. **(Optional) Local SQLite helper**
   ```bash
   node WebClient/server/sqliteStore.js  # stores GUI-specific user metadata
   ```

## Launch

1. Start the FastAPI scheduler/agent service:
   ```bash
   cd Scheduler
   uvicorn main:app --reload --port 8000
   ```
2. Start the React GUI:
   ```bash
   cd WebClient
   npm start
   ```
3. Legacy Express APIs (`Servers/express-server`) can still be started with
   `node app.js` when you need the existing upload/feed endpoints.

Swagger docs for the scheduler live at `http://127.0.0.1:8000/docs`.

## Chatbot Homepage

The default route (`/`) now opens a chatbot-first homepage:

- Left navigation stays familiar.
- The right rail lists your custom AIs as avatars; click one to load its chat.
- The center panel is a LangChain/OpenAI chat wired into the new tools:
  - Ask it to prepare Terraform/Kubernetes/CDK deployments for a Dockerfile.
  - Tell it to install prerequisites (`install kubernetes`, `install terraform`).
  - Ask “what do you see in the picture 1345d065?” to hit the Chroma-backed RAG.

You can still reach the legacy feed at `/feed`.

## Agentic Automation

`AgenticOrchestration` now exposes:

- `TerraformMicroserviceTool` for Dockerfile-driven microservices.
- `KubernetesInstallTool` and `TerraformInstallTool` to bootstrap an Ubuntu host
  (recommended AWS instance: `c6i.large`).
- `ImageTagRag` (Chroma + SQLite) so any agent can answer “what’s in picture X?”.
- A LangChain `OpenAIAgent` that automatically loads `.env`, validates
  `OPENAI_API_KEY`, and registers the new tools.

See `AgenticOrchestration/README.md` for code samples and configuration details.

## Scheduler API Highlights

- `POST /api/v1/agent/chat` — primary chat endpoint used by the GUI.
- `POST /api/v1/install/kubernetes` — k3s install plan + instance recommendation.
- `POST /api/v1/install/terraform` — CLI install instructions.
- `POST /api/v1/rag/tags` + `GET /api/v1/rag/{image_id}` — manage/query the tag RAG.
- Legacy training-queue endpoints remain available.

## Tests

Run the unit tests with:

```bash
pytest AgenticOrchestration/tests
```

These cover the Kubernetes/Terraform installers and the Chroma-backed RAG utility.

## Troubleshooting

- Delete `package-lock.json` and rerun `npm install` if the WebClient fails to start.
- If React reports an ESLint config conflict, open `package.json`, re-save it, or run
  `npm cache clean --force`.
- Use the Scheduler root endpoint (`/`) to confirm `OPENAI_API_KEY` was loaded—the
  response includes whether the key is visible and the current tooling status.
