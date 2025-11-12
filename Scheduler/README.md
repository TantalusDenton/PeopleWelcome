# Scheduler Service

The Scheduler module now fronts the LangChain/OpenAI agent, Kubernetes/Terraform
install helpers, training queue, and the new Chroma RAG endpoint.

## Prerequisites

```bash
sudo apt update
sudo apt install python3-pip
pip install --upgrade pip
pip install -r AgenticOrchestration/requirements.txt
pip install fastapi "uvicorn[standard]" python-dotenv
```

1. Copy `AgenticOrchestration/.env` (or create a new file) that contains
   `OPENAI_API_KEY=...`.
2. (Optional) Copy `AgenticOrchestration/aws-credentials.example` to a safe location
   and populate it for AWS CDK deployments.
3. (Optional) Preload the RAG store by calling
   `ImageTagRag.sync_from_csv(Path("ImageClassifier/ai-data/ai-dataset.csv"))`.

## Launch

```bash
cd Scheduler
uvicorn main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

## Key Endpoints

- `POST /api/v1/agent/chat` – routes prompts to the LangChain/OpenAI agent. Body:
  `{ "user_id": "friendly.henry", "ai_name": "Finn The Human", "message": "..." }`.
- `GET /api/v1/agent/history?user=...&ai=...` – returns stored chat history.
- `POST /api/v1/install/kubernetes` – returns a k3s installation plan (includes the
  recommended AWS instance type).
- `POST /api/v1/install/terraform` – returns CLI installation steps.
- `POST /api/v1/rag/tags` and `GET /api/v1/rag/{image_id}` – manage/query the image
  tag knowledge base stored inside Chroma/SQLite.
- Existing training queue endpoints remain (`/api/v1/addToTrainingQueue/...` etc.).
