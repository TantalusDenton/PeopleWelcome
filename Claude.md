# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PeopleWelcome is an AI-driven social platform for humans and machines consisting of:

- **Scheduler** (port 8000) - FastAPI service orchestrating the LangChain/OpenAI agent, chat sessions, and install/RAG endpoints
- **WebClient** (port 3000) - React frontend with chatbot-first homepage and Firebase auth
- **AgenticOrchestration** - Python package with LangChain agent, tools (Terraform, Kubernetes, RAG), and orchestration logic
- **ImageClassifier** (port 3001) - TensorFlow-based multi-label image classification
- **GoogleVisionTree** (port 3002) - Google Vision API with decision tree logic
- **App-logic** (port 5000) - Legacy Express backend for file uploads

## Build and Run Commands

### Python Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r AgenticOrchestration/requirements.txt
pip install fastapi "uvicorn[standard]" python-dotenv
```

### Node Dependencies
```bash
cd WebClient && npm install
```

### Launch Services
```bash
# Scheduler (main agent service)
cd Scheduler && uvicorn main:app --reload --port 8000

# React frontend
cd WebClient && npm start

# Optional services
cd ImageClassifier && uvicorn main:app --reload --port 3001
cd GoogleVisionTree && uvicorn main:app --reload --port 3002
cd App-logic && node app.js
```

### Run Tests
```bash
pytest AgenticOrchestration/tests
```

## Architecture

### Data Flow
1. **Chat**: WebClient → `POST /api/v1/agent/chat` → Scheduler → AgenticOrchestrator → OpenAIAgent (with tools)
2. **Image Analysis**: Upload → App-logic → ImageClassifier/GoogleVisionTree → results cached in Chroma RAG → queryable via `/api/v1/rag/{image_id}`
3. **Infrastructure Deployment**: Agent calls TerraformMicroserviceTool → generates configs in `infra/microservices` → `terraform plan/apply`

### Key Components

**Scheduler/main.py**: `AgentSessionRegistry` maintains per-user/AI conversation state. Endpoints:
- `POST /api/v1/agent/chat` - Primary chat interface
- `GET /api/v1/agent/history` - Chat history
- `POST /api/v1/install/kubernetes` - k3s installation planning
- `POST /api/v1/install/terraform` - Terraform CLI setup
- `POST /api/v1/rag/tags` and `GET /api/v1/rag/{image_id}` - Image tag RAG

**AgenticOrchestration/**:
- `openai_agent.py` - LangChain agent wrapping ChatOpenAI, auto-loads `.env`
- `orchestrator.py` - Maintains conversation history across calls
- `toolkit.py` - Tools: `TerraformMicroserviceTool`, `KubernetesInstallTool`, `TerraformInstallTool`, `ImageTagRag` (Chroma-backed), `SeleniumTool`

**WebClient/src/**:
- `App.js` - Main router with AWS Amplify auth
- `pages/ChatHome.jsx` - Primary chat interface
- `context/AuthContext.js` and `context/ChatContext.js` - State management
- `firebase/firebase.js` - Firebase config

### Adding New Agent Tools
1. Create tool class in `AgenticOrchestration/toolkit.py`
2. Define Pydantic input schema
3. Register in `OpenAIAgent._build_tools()`
4. Add test in `AgenticOrchestration/tests/`

## Environment Configuration

Required in `AgenticOrchestration/.env`:
```
OPENAI_API_KEY=sk-...
```

AWS credentials for CDK: copy `AgenticOrchestration/aws-credentials.example` and populate.

Firebase config: `WebClient/src/firebase/firebase.js`

## Conventions

### File Naming
- React components: PascalCase (`ImageForm.jsx`)
- JS services/utilities: camelCase (`s3Service.js`)
- Python modules: snake_case (`decision_tree.py`)

### API Response Format
```javascript
{ status: 400|500, message: "Error description" }
```

### State Management
- React Context for global state (`AuthContext`, `ChatContext`)
- Local component state for UI-specific data
- AWS services for persistent storage

## API Documentation
- Scheduler: http://127.0.0.1:8000/docs
- ImageClassifier: http://127.0.0.1:3001/docs
- GoogleVisionTree: http://127.0.0.1:3002/docs
