from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
from AgenticOrchestration.multi_agent_graph import MultiAgentOrchestrator
from . import trainingQueue

# Add App-logic to path for database imports
APP_LOGIC_DIR = Path(__file__).resolve().parents[1] / "App-logic"
sys.path.insert(0, str(APP_LOGIC_DIR))

# Import database and S3 services
import db_service
import s3_service
from database import init_database

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


# New request models for platform upgrade
class CreateUserRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Display username")


class CreateAIRequest(BaseModel):
    owner_id: str = Field(..., description="User ID of the AI owner")
    name: str = Field(..., description="Name of the AI")
    is_public: bool = Field(False, description="Whether the AI is publicly visible")
    system_prompt: Optional[str] = Field(None, description="System prompt/personality for the AI")


class UpdateSystemPromptRequest(BaseModel):
    system_prompt: str = Field(..., description="New system prompt for the AI")


class UpdateAIVisibilityRequest(BaseModel):
    is_public: bool = Field(..., description="Whether the AI is publicly visible")


class AddTagRequest(BaseModel):
    image_id: str = Field(..., description="ID of the image to tag")
    ai_id: str = Field(..., description="ID of the AI adding the tag")
    tag: str = Field(..., description="Tag text")


class RemoveTagRequest(BaseModel):
    image_id: str = Field(..., description="ID of the image")
    ai_id: str = Field(..., description="ID of the AI")
    tag: str = Field(..., description="Tag to remove")


class MultiAgentChatRequest(BaseModel):
    user_id: str = Field(..., description="User ID initiating the chat")
    ai_ids: List[str] = Field(..., description="List of AI IDs to participate")
    message: str = Field(..., description="User message to send")


class ConversationMessageRequest(BaseModel):
    ai_id: str = Field(..., description="AI ID for the conversation")
    user_id: str = Field(..., description="User ID")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class PresignedUploadRequest(BaseModel):
    user_id: str = Field(..., description="User ID for the upload")
    ai_id: str = Field(..., description="AI ID to associate the image with")
    filename: str = Field(..., description="Original filename (used for extension)")
    content_type: str = Field("image/jpeg", description="MIME type of the file")
    expiration: int = Field(3600, description="URL expiration time in seconds")


app = FastAPI(title="Scheduler", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import training service
from . import training_service


@app.on_event("startup")
async def startup_event():
    """Initialize database and start background services on startup."""
    # Initialize database tables
    init_database()

    # Start the background training service
    training_service.start_training_service()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background services on shutdown."""
    training_service.stop_training_service()


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


# ============================================================================
# User Management Endpoints
# ============================================================================

@app.post("/api/v1/users")
async def create_user(payload: CreateUserRequest) -> Dict[str, Any]:
    """Create a new user or get existing one."""
    try:
        user = db_service.get_or_create_user(payload.user_id, payload.username)
        return {"status": "ok", "user": user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str) -> Dict[str, Any]:
    """Get a user by ID."""
    user = db_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": user}


# ============================================================================
# AI Management Endpoints
# ============================================================================

@app.post("/api/v1/ais")
async def create_ai(payload: CreateAIRequest) -> Dict[str, Any]:
    """Create a new AI for a user."""
    try:
        # Ensure user exists
        db_service.get_or_create_user(payload.owner_id, payload.owner_id)
        ai = db_service.create_ai(
            owner_id=payload.owner_id,
            name=payload.name,
            is_public=payload.is_public,
            system_prompt=payload.system_prompt
        )
        return {"status": "ok", "ai": ai}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/ais/{ai_id}")
async def get_ai(ai_id: str) -> Dict[str, Any]:
    """Get an AI by ID."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")
    return {"ai": ai}


@app.get("/api/v1/ais/owner/{owner_id}")
async def get_ais_by_owner(owner_id: str) -> Dict[str, Any]:
    """Get all AIs owned by a user."""
    ais = db_service.get_ais_by_owner(owner_id)
    return {"ais": ais}


@app.get("/api/v1/ais/public")
async def get_public_ais() -> Dict[str, Any]:
    """Get all public AIs."""
    ais = db_service.get_public_ais()
    return {"ais": ais}


@app.put("/api/v1/ais/{ai_id}/system-prompt")
async def update_ai_system_prompt(ai_id: str, payload: UpdateSystemPromptRequest) -> Dict[str, Any]:
    """Update an AI's system prompt (instruction mode)."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    success = db_service.update_ai_system_prompt(ai_id, payload.system_prompt)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update system prompt")

    return {"status": "ok", "ai_id": ai_id, "system_prompt": payload.system_prompt}


@app.put("/api/v1/ais/{ai_id}/visibility")
async def update_ai_visibility(ai_id: str, payload: UpdateAIVisibilityRequest) -> Dict[str, Any]:
    """Update an AI's public visibility."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    success = db_service.update_ai_visibility(ai_id, payload.is_public)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update visibility")

    return {"status": "ok", "ai_id": ai_id, "is_public": payload.is_public}


@app.delete("/api/v1/ais/{ai_id}")
async def delete_ai(ai_id: str) -> Dict[str, Any]:
    """Delete an AI and all related data."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    success = db_service.delete_ai(ai_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete AI")

    return {"status": "ok", "deleted_ai_id": ai_id}


# ============================================================================
# Image Management Endpoints
# ============================================================================

@app.post("/api/v1/images/presigned-upload")
async def get_presigned_upload(payload: PresignedUploadRequest) -> Dict[str, Any]:
    """Get a presigned URL for client-side direct upload to S3."""
    # Validate AI exists
    ai = db_service.get_ai(payload.ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    # Ensure user exists
    db_service.get_or_create_user(payload.user_id, payload.user_id)

    try:
        # Generate S3 key
        s3_key = s3_service.generate_s3_key(
            payload.user_id,
            payload.ai_id,
            payload.filename
        )

        # Get presigned upload URL
        presigned_data = s3_service.get_presigned_upload_url(
            s3_key,
            content_type=payload.content_type,
            expiration=payload.expiration
        )

        return {
            "status": "ok",
            "s3_key": s3_key,
            "presigned_url": presigned_data["url"],
            "fields": presigned_data["fields"],
            "expires_in": payload.expiration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")


@app.post("/api/v1/images/confirm-upload")
async def confirm_upload(
    s3_key: str = Form(...),
    user_id: str = Form(...),
    ai_id: str = Form(...),
    is_public: bool = Form(False)
) -> Dict[str, Any]:
    """Confirm a successful client-side upload and create database record."""
    # Validate AI exists
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    # Verify the image exists in S3
    if not s3_service.check_image_exists(s3_key):
        raise HTTPException(status_code=400, detail="Image not found in S3")

    try:
        # Create database record
        image = db_service.create_image(
            owner_id=user_id,
            ai_id=ai_id,
            s3_key=s3_key,
            is_public=is_public
        )

        return {
            "status": "ok",
            "image": image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to confirm upload: {str(e)}")


@app.post("/api/v1/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    ai_id: str = Form(...),
    is_public: bool = Form(False)
) -> Dict[str, Any]:
    """Upload an image with AI assignment."""
    # Validate AI exists
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    # Ensure user exists
    db_service.get_or_create_user(user_id, user_id)

    try:
        # Upload to S3
        s3_result = s3_service.upload_image(
            file.file,
            user_id,
            ai_id,
            file.filename,
            file.content_type
        )

        # Create database record
        image = db_service.create_image(
            owner_id=user_id,
            ai_id=ai_id,
            s3_key=s3_result["s3_key"],
            is_public=is_public
        )

        return {
            "status": "ok",
            "image": image,
            "s3_key": s3_result["s3_key"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@app.get("/api/v1/images/{ai_id}")
async def get_images_by_ai(
    ai_id: str,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """Get paginated images for an AI."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    images = db_service.get_images_by_ai(ai_id, limit, offset)
    return {"images": images, "count": len(images), "limit": limit, "offset": offset}


@app.get("/api/v1/images/{image_id}/url")
async def get_image_url(image_id: str, expiration: int = 3600) -> Dict[str, Any]:
    """Get presigned S3 URL for an image."""
    image = db_service.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        url = s3_service.get_presigned_url(image["s3_key"], expiration)
        return {"url": url, "expires_in": expiration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")


@app.get("/api/v1/images/{image_id}/details")
async def get_image_details(image_id: str) -> Dict[str, Any]:
    """Get image details including tags."""
    image = db_service.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    tags = db_service.get_tags_for_image(image_id)
    return {"image": image, "tags": tags}


@app.delete("/api/v1/images/{image_id}")
async def delete_image(image_id: str) -> Dict[str, Any]:
    """Delete an image from S3 and database."""
    image = db_service.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Delete from S3
        s3_service.delete_image(image["s3_key"])
    except Exception:
        pass  # Continue even if S3 deletion fails

    # Delete from database
    success = db_service.delete_image(image_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete image")

    return {"status": "ok", "deleted_image_id": image_id}


# ============================================================================
# Tag Management Endpoints
# ============================================================================

@app.post("/api/v1/tags")
async def add_tag(payload: AddTagRequest) -> Dict[str, Any]:
    """Add a tag to an image (triggers training queue)."""
    # Validate image and AI exist
    image = db_service.get_image(payload.image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    ai = db_service.get_ai(payload.ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    tag_result = db_service.add_tag(payload.image_id, payload.ai_id, payload.tag)

    # Queue for training
    db_service.create_training_job(payload.ai_id)

    return {"status": "ok", "tag": tag_result}


@app.delete("/api/v1/tags")
async def remove_tag(payload: RemoveTagRequest) -> Dict[str, Any]:
    """Remove a tag from an image."""
    success = db_service.remove_tag(payload.image_id, payload.ai_id, payload.tag)
    if not success:
        raise HTTPException(status_code=404, detail="Tag not found")

    return {"status": "ok", "removed": True}


@app.get("/api/v1/tags/image/{image_id}")
async def get_tags_for_image(image_id: str) -> Dict[str, Any]:
    """Get all tags for an image across all AIs."""
    image = db_service.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    tags = db_service.get_tags_for_image(image_id)
    return {"tags": tags}


@app.get("/api/v1/tags/image/{image_id}/ai/{ai_id}")
async def get_tags_for_image_by_ai(image_id: str, ai_id: str) -> Dict[str, Any]:
    """Get tags for an image by a specific AI."""
    tags = db_service.get_tags_for_image_by_ai(image_id, ai_id)
    return {"tags": tags}


@app.get("/api/v1/tags/ai/{ai_id}")
async def get_all_tags_by_ai(ai_id: str) -> Dict[str, Any]:
    """Get all unique tags used by an AI with counts."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    tags = db_service.get_all_tags_by_ai(ai_id)
    return {"tags": tags}


@app.get("/api/v1/tags/inference/{ai_id}/{image_id}")
async def run_inference(ai_id: str, image_id: str) -> Dict[str, Any]:
    """Run tag inference on an image using an AI's trained model."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    image = db_service.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if AI has a trained model
    if not ai.get("model_path"):
        raise HTTPException(status_code=400, detail="AI has no trained model")

    # Run inference using the training service
    result = training_service.run_inference_for_ai(ai_id, image_id)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Inference failed"))

    return {
        "status": "success",
        "ai_id": ai_id,
        "image_id": image_id,
        "predictions": result.get("predictions", [])
    }


# ============================================================================
# Conversation Management Endpoints
# ============================================================================

@app.post("/api/v1/conversations/message")
async def add_conversation_message(payload: ConversationMessageRequest) -> Dict[str, Any]:
    """Add a message to an AI conversation history."""
    ai = db_service.get_ai(payload.ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    message = db_service.add_conversation_message(
        payload.ai_id,
        payload.user_id,
        payload.role,
        payload.content
    )
    return {"status": "ok", "message": message}


@app.get("/api/v1/conversations/{ai_id}/{user_id}")
async def get_conversation_history(
    ai_id: str,
    user_id: str,
    limit: int = 100
) -> Dict[str, Any]:
    """Get conversation history between a user and AI."""
    history = db_service.get_conversation_history(ai_id, user_id, limit)
    return {"messages": history}


@app.delete("/api/v1/conversations/{ai_id}/{user_id}")
async def clear_conversation(ai_id: str, user_id: str) -> Dict[str, Any]:
    """Clear conversation history between a user and AI."""
    count = db_service.clear_conversation_history(ai_id, user_id)
    return {"status": "ok", "deleted_count": count}


# ============================================================================
# Training Job Endpoints
# ============================================================================

@app.get("/api/v1/training/pending")
async def get_pending_training_jobs() -> Dict[str, Any]:
    """Get all pending training jobs."""
    jobs = db_service.get_pending_training_jobs()
    return {"jobs": jobs}


@app.get("/api/v1/training/ai/{ai_id}")
async def get_ai_training_status(ai_id: str) -> Dict[str, Any]:
    """Get the latest training job status for an AI."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    job = db_service.get_latest_training_job(ai_id)
    return {"ai_id": ai_id, "latest_job": job}


# ============================================================================
# Spawned Agent Endpoints
# ============================================================================

@app.get("/api/v1/agents/spawned")
async def get_all_spawned_agents() -> Dict[str, Any]:
    """Get all spawned agents."""
    agents = db_service.get_all_spawned_agents()
    return {"agents": agents}


@app.get("/api/v1/agents/spawned/{agent_id}")
async def get_spawned_agent(agent_id: str) -> Dict[str, Any]:
    """Get a spawned agent by ID."""
    agent = db_service.get_spawned_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Spawned agent not found")
    return {"agent": agent}


@app.delete("/api/v1/agents/spawned/{agent_id}")
async def delete_spawned_agent(agent_id: str) -> Dict[str, Any]:
    """Delete a spawned agent."""
    success = db_service.delete_spawned_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Spawned agent not found")
    return {"status": "ok", "deleted_agent_id": agent_id}


# ============================================================================
# Custom Integration Endpoints
# ============================================================================

@app.get("/api/v1/integrations/ai/{ai_id}")
async def get_integrations_by_ai(ai_id: str) -> Dict[str, Any]:
    """Get all custom integrations for an AI."""
    ai = db_service.get_ai(ai_id)
    if not ai:
        raise HTTPException(status_code=404, detail="AI not found")

    integrations = db_service.get_integrations_by_ai(ai_id)
    return {"integrations": integrations}


@app.delete("/api/v1/integrations/{integration_id}")
async def delete_custom_integration(integration_id: str) -> Dict[str, Any]:
    """Delete a custom integration."""
    success = db_service.delete_custom_integration(integration_id)
    if not success:
        raise HTTPException(status_code=404, detail="Integration not found")
    return {"status": "ok", "deleted_integration_id": integration_id}


# ============================================================================
# Multi-AI Chat Endpoints
# ============================================================================

# Singleton orchestrator instance
_multi_agent_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the multi-agent orchestrator."""
    global _multi_agent_orchestrator
    if _multi_agent_orchestrator is None:
        _multi_agent_orchestrator = MultiAgentOrchestrator()
    return _multi_agent_orchestrator


@app.post("/api/v1/agent/multi-chat")
async def multi_ai_chat(payload: MultiAgentChatRequest) -> Dict[str, Any]:
    """
    Run a multi-AI roundtable discussion.

    This endpoint orchestrates multiple AIs to collaboratively answer a user's question.
    The AIs first respond independently, then discuss their responses, and finally
    a synthesized answer is produced.
    """
    # Validate all AIs exist
    ai_configs = []
    for ai_id in payload.ai_ids:
        ai = db_service.get_ai(ai_id)
        if not ai:
            raise HTTPException(status_code=404, detail=f"AI not found: {ai_id}")
        ai_configs.append({
            "id": ai["id"],
            "name": ai["name"],
            "owner_id": ai["owner_id"],
            "system_prompt": ai.get("system_prompt"),
        })

    if len(ai_configs) < 2:
        raise HTTPException(
            status_code=400,
            detail="Multi-AI chat requires at least 2 AIs"
        )

    # Run the multi-agent workflow
    orchestrator = get_multi_agent_orchestrator()
    result = await orchestrator.run(
        user_prompt=payload.message,
        selected_ais=ai_configs,
        user_id=payload.user_id
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Multi-AI chat failed")
        )

    return {
        "status": "success",
        "session_id": result.get("session_id"),
        "reply": result.get("winning_response"),
        "responses": result.get("responses", {}),
        "discussion_log": result.get("discussion_log", []),
        "spawned_agents": result.get("spawned_agents", [])
    }
