"""PeopleWelcome's authenticated, persona-based chat API."""
from __future__ import annotations

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Literal

import httpx
import stripe
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from .auth import authenticator, optional_user, require_user
from .billing import BillingConfigurationError, BillingService
from .storage import Store


BASE_DIR = Path(__file__).resolve().parents[1]
logger = logging.getLogger("peoplewelcome.scheduler")
load_dotenv(BASE_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://peoplewelcome:peoplewelcome-local@localhost:5432/peoplewelcome")
store = Store(DATABASE_URL)
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "data" / "media"))
IMAGE_GENERATOR_URL = os.getenv("IMAGE_GENERATOR_URL", "http://image-generator:8001")
billing = BillingService(store)

app = FastAPI(title="PeopleWelcome Chat API", version="4.0")
app.state.store = store
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:13000").split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PersonaRequest(BaseModel):
    persona: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    ai_id: str
    message: str = Field(min_length=1, max_length=12000)


class ImageSceneRequest(BaseModel):
    ai_ids: list[str] = Field(min_length=1, max_length=4)
    prompt: str = Field(default="", max_length=4000)


def configured_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def trim_messages(messages: list[dict], max_chars: int) -> list[dict]:
    kept, used = [], 0
    for message in reversed(messages):
        size = len(message.get("content", ""))
        if kept and used + size > max_chars:
            break
        kept.append(message)
        used += size
    return list(reversed(kept))


@app.on_event("startup")
async def startup() -> None:
    await run_in_threadpool(store.initialize)
    migration_path = os.getenv("SQLITE_MIGRATION_PATH")
    if migration_path:
        await run_in_threadpool(store.migrate_sqlite, Path(migration_path))


@app.on_event("shutdown")
async def shutdown() -> None:
    store.close()


@app.get("/")
async def root():
    return {
        "server": "PeopleWelcome",
        "storage": "postgresql",
        "models": ["openai", "unstoppable"],
        "authentication": "zitadel",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "zitadel_configured": authenticator.configured,
        "stripe_configured": billing.configured,
    }


@app.get("/api/v1/me")
async def me(user: dict = Depends(require_user)):
    return {"user": user}


@app.post("/api/v1/legacy/import")
async def import_legacy_characters(user: dict = Depends(require_user)):
    if os.getenv("LEGACY_IMPORT_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(404, "Local legacy import is disabled.")
    legacy_owner_ids = [
        owner_id.strip()
        for owner_id in os.getenv("LEGACY_OWNER_IDS", "").split(",")
        if owner_id.strip()
    ]
    if not legacy_owner_ids:
        raise HTTPException(404, "No local legacy owners are configured for import.")
    try:
        imported = store.import_legacy_owner_data(user["id"], legacy_owner_ids)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"imported": imported, "ais": store.list_ais(owner_id=user["id"])}


def media_url(path: Path) -> str:
    return f"/media/{path.relative_to(MEDIA_ROOT).as_posix()}"


async def generate_character_views(ai_id: str, source_path: Path) -> None:
    ai = store.get_ai(ai_id)
    if not ai:
        return
    store.update_generation(ai_id, "generating")
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("IMAGE_GENERATION_TIMEOUT", "1200"))) as client:
            response = await client.post(
                f"{IMAGE_GENERATOR_URL.rstrip('/')}/v1/generate-character-views",
                json={
                    "character_id": ai_id,
                    "reference_image_path": str(source_path),
                    "name": ai["name"],
                    "persona": ai["persona"],
                    "output_dir": str(MEDIA_ROOT / "generated" / ai_id),
                },
            )
        if not response.is_success:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text.strip()
            raise RuntimeError(detail or f"Image generator returned HTTP {response.status_code}.")
        payload = response.json()
        image_urls = {key: media_url(Path(value)) for key, value in payload["images"].items()}
        store.update_generation(
            ai_id,
            "ready",
            profile_image_url=image_urls["avatar_front_face_url"],
            **image_urls,
        )
    except Exception as error:
        logger.exception("Character-view generation failed for %s", ai_id)
        store.update_generation(ai_id, "failed", error=str(error)[:1000])


def source_path_for_ai(ai: dict) -> Path | None:
    source_url = ai.get("original_avatar_url")
    if not source_url or not source_url.startswith("/media/"):
        return None
    path = (MEDIA_ROOT / source_url.removeprefix("/media/")).resolve()
    return path if MEDIA_ROOT.resolve() in path.parents and path.is_file() else None


def character_description(ai: dict) -> str:
    return f"{ai['name']}: {ai['persona']}"


def accessible_ai(ai_id: str, user: dict | None, owner_only: bool = False) -> dict:
    ai = store.get_ai(ai_id)
    if not ai:
        raise HTTPException(404, "AI not found")
    owned = bool(user and ai["owner_id"] == user["id"])
    if not owned and (owner_only or not ai["is_public"]):
        raise HTTPException(404, "AI not found")
    return ai


@app.post("/api/v1/ais")
async def create_ai(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    persona: str = Form(...),
    model: Literal["openai", "unstoppable"] = Form("openai"),
    is_public: bool = Form(False),
    avatar: UploadFile | None = File(None),
    user: dict = Depends(require_user),
):
    name, persona = name.strip(), persona.strip()
    if not 1 <= len(name) <= 120 or not 1 <= len(persona) <= 4000:
        raise HTTPException(422, "Name and persona are required and must fit their length limits.")
    if model == "unstoppable" and not user["is_premium"]:
        raise HTTPException(403, "Unstoppable is a Premium feature. Start Premium in Settings first.")
    avatar_url, source_path, status = None, None, "not_requested"
    if avatar:
        if not (avatar.content_type or "").startswith("image/"):
            raise HTTPException(422, "Avatar must be an image.")
        suffix = Path(avatar.filename or "avatar.png").suffix.lower() or ".png"
        source_path = MEDIA_ROOT / "uploads" / f"{uuid.uuid4()}{suffix}"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        avatar_url, status = media_url(source_path), "pending"
    try:
        ai = store.create_ai(user["id"], name, persona, model, is_public, avatar_url, status)
    except Exception as error:
        if "unique" in str(error).lower():
            raise HTTPException(409, "You already have an AI character with that name.") from error
        raise
    if source_path:
        background_tasks.add_task(generate_character_views, ai["id"], source_path)
    return {"ai": ai}


@app.get("/api/v1/ais/mine")
async def my_ais(user: dict = Depends(require_user)):
    return {"ais": store.list_ais(owner_id=user["id"])}


@app.get("/api/v1/ais/public")
async def public_ais():
    return {"ais": store.list_ais(public_only=True)}


@app.get("/api/v1/ais/{ai_id}")
async def get_ai(ai_id: str, user: dict | None = Depends(optional_user)):
    return {"ai": accessible_ai(ai_id, user)}


@app.get("/api/v1/ais/{ai_id}/images")
async def ai_images(ai_id: str, user: dict | None = Depends(optional_user)):
    ai = accessible_ai(ai_id, user)
    fields = (
        "original_avatar_url", "avatar_front_face_url", "avatar_side_face_url",
        "full_body_front_url", "full_body_side_url", "full_body_back_url", "profile_image_url",
    )
    return {
        "status": ai["image_generation_status"],
        "error": ai["image_generation_error"],
        "images": {field: ai[field] for field in fields},
    }


@app.post("/api/v1/ais/{ai_id}/images/regenerate")
async def regenerate_ai_images(
    ai_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_user),
):
    ai = accessible_ai(ai_id, user, owner_only=True)
    source_path = source_path_for_ai(ai)
    if not source_path:
        raise HTTPException(422, "This character has no stored avatar reference image.")
    store.update_generation(ai_id, "pending")
    background_tasks.add_task(generate_character_views, ai_id, source_path)
    return {"ai": store.get_ai(ai_id)}


@app.get("/media/{media_path:path}")
async def get_media(media_path: str):
    path = (MEDIA_ROOT / media_path).resolve()
    if MEDIA_ROOT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Media not found")
    return FileResponse(path)


@app.post("/api/v1/images/generate")
async def generate_image_scene(request: ImageSceneRequest, user: dict = Depends(require_user)):
    characters = [accessible_ai(ai_id, user) for ai_id in dict.fromkeys(request.ai_ids)]
    references = [path for ai in characters if (path := source_path_for_ai(ai))]
    if not references:
        raise HTTPException(422, "Select a character with an uploaded avatar reference before generating an image.")
    scene_id = str(uuid.uuid4())
    destination = MEDIA_ROOT / "scenes" / f"{scene_id}.png"
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("IMAGE_GENERATION_TIMEOUT", "1200"))) as client:
            response = await client.post(
                f"{IMAGE_GENERATOR_URL.rstrip('/')}/v1/generate-scene",
                json={
                    "scene_id": scene_id,
                    "reference_image_paths": [str(path) for path in references],
                    "character_descriptions": [character_description(ai) for ai in characters],
                    "user_prompt": request.prompt,
                    "output_path": str(destination),
                },
            )
        if not response.is_success:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text.strip()
            raise HTTPException(response.status_code, detail or "Image generation failed.")
        return {
            "image_url": media_url(destination),
            "characters": [ai["name"] for ai in characters],
            "prompt": response.json().get("prompt"),
        }
    except httpx.HTTPStatusError as error:
        raise HTTPException(error.response.status_code, error.response.text) from error
    except httpx.HTTPError as error:
        raise HTTPException(503, f"Image generator is unavailable: {error}") from error


@app.put("/api/v1/ais/{ai_id}/persona")
async def update_persona(ai_id: str, payload: PersonaRequest, user: dict = Depends(require_user)):
    accessible_ai(ai_id, user, owner_only=True)
    store.update_persona(ai_id, payload.persona)
    return {"ai": store.get_ai(ai_id)}


async def generate(ai: dict, messages: list[dict]) -> dict:
    if ai["model"] == "unstoppable":
        url, key = os.getenv("UNSTOPPABLE_LLM_URL"), os.getenv("UNSTOPPABLE_LLM_API_KEY")
        if not url or not key:
            raise HTTPException(503, "Unstoppable model is not configured.")
        max_tokens = configured_int("UNSTOPPABLE_MAX_TOKENS", 2048, 128, 8192)
        history_chars = configured_int("UNSTOPPABLE_HISTORY_MAX_CHARS", 32000, 4000, 200000)
        timeout = configured_int("UNSTOPPABLE_TIMEOUT_SECONDS", 300, 30, 1800)
        request_body = {
            "persona_prompt": ai["persona"],
            "messages": trim_messages(messages, history_chars),
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        payload, reply = None, None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.post(
                        f"{url.rstrip('/')}/v1/unstoppable/chat",
                        headers={"X-API-Key": key},
                        json=request_body,
                    )
            except httpx.TimeoutException as error:
                raise HTTPException(504, f"Unstoppable timed out after {timeout} seconds.") from error
            except httpx.HTTPError as error:
                raise HTTPException(502, f"Unstoppable request failed: {error}") from error
            if response.status_code == 401:
                raise HTTPException(503, "Unstoppable model authentication failed.")
            if response.is_error:
                raise HTTPException(502, "Unstoppable model could not complete the request.")
            try:
                payload = response.json()
            except ValueError as error:
                if attempt == 0:
                    continue
                raise HTTPException(502, "Unstoppable returned an invalid response after retrying.") from error
            reply = payload.get("response")
            if isinstance(reply, str) and reply.strip():
                break
            if attempt == 1:
                raise HTTPException(502, "Unstoppable returned an empty response after retrying.")
        usage = payload.get("usage") or {}
        finish_reason = payload.get("finish_reason") or (
            "length" if usage.get("completion_tokens", 0) >= max_tokens - 1 else "stop"
        )
        return {
            "content": reply,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "usage": usage,
        }
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(503, "OpenAI is not configured.")
    response = await run_in_threadpool(
        lambda: OpenAI(api_key=key).chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": ai["persona"]}, *messages],
        )
    )
    choice = response.choices[0]
    return {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "truncated": choice.finish_reason == "length",
        "usage": response.usage.model_dump() if response.usage else {},
    }


@app.post("/api/v1/agent/chat")
async def chat(payload: ChatRequest, user: dict = Depends(require_user)):
    ai = accessible_ai(payload.ai_id, user)
    if ai["model"] == "unstoppable" and not user["is_premium"]:
        raise HTTPException(403, "An active Premium subscription is required for Unstoppable chat.")
    history = store.history(payload.ai_id, user["id"])
    messages = [{"role": row["role"], "content": row["content"]} for row in history]
    messages.append({"role": "user", "content": payload.message})
    generation = await generate(ai, messages)
    user_message = store.add_message(payload.ai_id, user["id"], "user", payload.message)
    assistant_message = store.add_message(payload.ai_id, user["id"], "assistant", generation["content"])
    return {
        "reply": generation["content"],
        "messages": [user_message, assistant_message],
        "model": ai["model"],
        **{key: generation[key] for key in ("finish_reason", "truncated", "usage")},
    }


@app.get("/api/v1/conversations/{ai_id}")
async def conversation(ai_id: str, user: dict = Depends(require_user)):
    accessible_ai(ai_id, user)
    return {"messages": store.history(ai_id, user["id"])}


@app.get("/api/v1/billing/status")
async def billing_status(user: dict = Depends(require_user)):
    return {
        "premium": user["is_premium"],
        "status": user["subscription_status"],
        "current_period_end": user["subscription_current_period_end"],
        "cancel_at_period_end": user["cancel_at_period_end"],
        "has_customer": bool(user["stripe_customer_id"]),
        "price": {"amount": 20, "currency": "USD", "interval": "month"},
    }


@app.post("/api/v1/billing/checkout")
async def create_checkout(user: dict = Depends(require_user)):
    if user["is_premium"]:
        raise HTTPException(409, "Premium is already active. Use Manage subscription instead.")
    try:
        url = await run_in_threadpool(billing.create_checkout, user)
        return {"url": url}
    except BillingConfigurationError as error:
        raise HTTPException(503, str(error)) from error
    except stripe.StripeError as error:
        raise HTTPException(502, "Stripe could not create the checkout session.") from error


@app.post("/api/v1/billing/portal")
async def create_billing_portal(user: dict = Depends(require_user)):
    try:
        url = await run_in_threadpool(billing.create_portal, user)
        return {"url": url}
    except (BillingConfigurationError, ValueError) as error:
        raise HTTPException(409 if isinstance(error, ValueError) else 503, str(error)) from error
    except stripe.StripeError as error:
        raise HTTPException(502, "Stripe could not open the billing portal.") from error


@app.post("/api/v1/billing/webhook")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(None, alias="Stripe-Signature")):
    payload = await request.body()
    try:
        await run_in_threadpool(billing.process_webhook, payload, stripe_signature)
    except BillingConfigurationError as error:
        raise HTTPException(503, str(error)) from error
    except (stripe.SignatureVerificationError, TypeError, ValueError) as error:
        raise HTTPException(400, "Invalid Stripe webhook signature or payload.") from error
    return {"received": True}
