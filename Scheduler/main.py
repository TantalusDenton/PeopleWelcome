from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from AgenticOrchestration import (
    AgentTooling,
    AgenticOrchestrator,
    ImageTagRag,
    KubernetesInstallTool,
    OpenAIAgent,
    TerraformInstallTool,
    TerraformMicroserviceTool,
)
import trainingQueue

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATHS = [
    BASE_DIR / ".env",
    BASE_DIR / "AgenticOrchestration" / ".env",
]
for env_path in ENV_PATHS:
    load_dotenv(env_path, override=False)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not configured. Add it to AgenticOrchestration/.env "
        "or export it before starting Scheduler."
    )

INFRA_ROOT = BASE_DIR / "infra"
MICROSERVICE_ROOT = INFRA_ROOT / "microservices"
RAG_STORE_DIR = INFRA_ROOT / "rag-store"
MICROSERVICE_ROOT.mkdir(parents=True, exist_ok=True)
RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)

tooling = AgentTooling(
    microservice=TerraformMicroserviceTool(MICROSERVICE_ROOT, kubeconfig=Path("~/.kube/config").expanduser()),
    kubernetes_installer=KubernetesInstallTool(),
    terraform_installer=TerraformInstallTool(),
    rag=ImageTagRag(RAG_STORE_DIR),
)


class AgentSessionRegistry:
    """Keeps AgenticOrchestrator instances keyed by (user, ai)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, AgenticOrchestrator] = {}

    @staticmethod
    def _key(user: str, ai: str) -> str:
        user_key = (user or "anonymous").strip().lower() or "anonymous"
        ai_key = (ai or "default").strip().lower() or "default"
        return f"{user_key}::{ai_key}"

    def _create_session(self) -> AgenticOrchestrator:
        return AgenticOrchestrator(OpenAIAgent(tooling))

    def get_orchestrator(self, user: str, ai: str) -> AgenticOrchestrator:
        key = self._key(user, ai)
        if key not in self._sessions:
            self._sessions[key] = self._create_session()
        return self._sessions[key]

    def get_history(self, user: str, ai: str) -> List[Dict[str, Any]]:
        orchestrator = self._sessions.get(self._key(user, ai))
        if not orchestrator:
            return []
        formatted: List[Dict[str, Any]] = []
        for message in orchestrator.history:
            formatted.append(
                {
                    "role": self._role_for_message(message),
                    "content": message.content,
                }
            )
        return formatted

    @staticmethod
    def _role_for_message(message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        return getattr(message, "type", message.__class__.__name__)


SESSION_REGISTRY = AgentSessionRegistry()


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Identifier for the human that owns the AI.")
    ai_name: str = Field(..., description="Name of the AI persona.")
    message: str = Field(..., description="Prompt sent to the LangChain-powered agent.")


class InstallKubernetesRequest(BaseModel):
    cluster_name: str
    channel: str = "stable"
    runtime: str = "containerd"
    hostname: str = "localhost"


class InstallTerraformRequest(BaseModel):
    version: Optional[str] = None
    destination: Optional[str] = None


class RagUpsertRequest(BaseModel):
    image_id: str
    tags: List[str]
    owner: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagSearchRequest(BaseModel):
    question: str
    top_k: int = Field(3, ge=1, le=10)


app = FastAPI(title="Scheduler", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "server": "Scheduler",
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "tooling": json.loads(tooling.to_status_report()),
    }


def fileify(file_path: Path) -> FileResponse:
    return FileResponse(file_path)


@app.post("/api/v1/addToTrainingQueue/{user}/{ai}")
async def add_to_training_queue(user: str, ai: str) -> Dict[str, str]:
    trainingQueue.queueForRetraining(user, ai)
    return {"status": "ok", "queued": f"{user}:{ai}"}


@app.get("/api/v1/getTrainingQueue/")
async def get_training_queue() -> Dict[str, Any]:
    queue_snapshot = list(trainingQueue.trainingQueue.queue)
    return {"queue": queue_snapshot}


@app.get("/api/v1/getFirstElementInQueue/")
async def get_first_element() -> Dict[str, Any]:
    queue_element = trainingQueue.getFirstElementInQueue()
    return {"next": queue_element}


@app.post("/api/v1/agent/chat")
async def agent_chat(payload: ChatRequest) -> Dict[str, Any]:
    orchestrator = SESSION_REGISTRY.get_orchestrator(payload.user_id, payload.ai_name)
    result = orchestrator.invoke(payload.message)
    messages = [
        {
            "role": SESSION_REGISTRY._role_for_message(msg),
            "content": msg.content,
        }
        for msg in result["messages"]
    ]
    reply = messages[-1]["content"] if messages else ""
    return {
        "reply": reply,
        "messages": messages,
        "intermediate_steps": result.get("intermediate_steps", []),
        "context": result.get("context", {}),
    }


@app.get("/api/v1/agent/history/")
async def agent_history(user: str, ai: str) -> Dict[str, Any]:
    return {"messages": SESSION_REGISTRY.get_history(user, ai)}


@app.post("/api/v1/install/kubernetes")
async def install_kubernetes(payload: InstallKubernetesRequest) -> Dict[str, Any]:
    if tooling.kubernetes_installer is None:
        raise HTTPException(status_code=503, detail="Kubernetes installer is not configured.")
    plan = tooling.kubernetes_installer.install(
        cluster_name=payload.cluster_name,
        channel=payload.channel,
        runtime=payload.runtime,
        hostname=payload.hostname,
    )
    return json.loads(plan)


@app.post("/api/v1/install/terraform")
async def install_terraform(payload: InstallTerraformRequest) -> Dict[str, Any]:
    if tooling.terraform_installer is None:
        raise HTTPException(status_code=503, detail="Terraform installer is not configured.")
    plan = tooling.terraform_installer.install(
        version=payload.version,
        destination=payload.destination,
    )
    return json.loads(plan)


@app.post("/api/v1/rag/tags")
async def rag_upsert(payload: RagUpsertRequest) -> Dict[str, Any]:
    if tooling.rag is None:
        raise HTTPException(status_code=503, detail="RAG store is not configured.")
    metadata = {**payload.metadata}
    if payload.owner:
        metadata.setdefault("owner", payload.owner)
    record = tooling.rag.upsert(payload.image_id, payload.tags, metadata=metadata)
    return {"status": "ok", "record": record}


@app.get("/api/v1/rag/{image_id}")
async def rag_describe(image_id: str) -> Dict[str, Any]:
    if tooling.rag is None:
        raise HTTPException(status_code=503, detail="RAG store is not configured.")
    return tooling.rag.describe(image_id)


@app.post("/api/v1/rag/search")
async def rag_search(payload: RagSearchRequest) -> Dict[str, Any]:
    if tooling.rag is None:
        raise HTTPException(status_code=503, detail="RAG store is not configured.")
    return {"results": tooling.rag.query(payload.question, top_k=payload.top_k)}
