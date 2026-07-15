"""PeopleWelcome's local, persona-based chat API."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from .storage import Store

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
store = Store(Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "peoplewelcome.db")))
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "data" / "media"))
IMAGE_GENERATOR_URL = os.getenv("IMAGE_GENERATOR_URL", "http://image-generator:8001")

app = FastAPI(title="PeopleWelcome Chat API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","), allow_methods=["*"], allow_headers=["*"])

class UserRequest(BaseModel):
    user_id: str
    username: str

class SubscriptionRequest(BaseModel):
    premium: bool = True

class AIRequest(BaseModel):
    owner_id: str
    name: str = Field(min_length=1, max_length=120)
    persona: str = Field(min_length=1, max_length=4000)
    model: Literal["openai", "unstoppable"] = "openai"
    is_public: bool = False

class PersonaRequest(BaseModel):
    persona: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    user_id: str
    ai_id: str
    message: str = Field(min_length=1, max_length=12000)

class ImageSceneRequest(BaseModel):
    ai_ids: list[str] = Field(min_length=1, max_length=4)
    prompt: str = Field(default="", max_length=4000)

def configured_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try: value = int(os.getenv(name, str(default)))
    except ValueError: value = default
    return max(minimum, min(value, maximum))

def trim_messages(messages: list[dict], max_chars: int) -> list[dict]:
    kept, used = [], 0
    for message in reversed(messages):
        size = len(message.get("content", ""))
        if kept and used + size > max_chars: break
        kept.append(message); used += size
    return list(reversed(kept))

@app.on_event("startup")
async def startup() -> None:
    store.initialize()

@app.get("/")
async def root():
    return {"server": "PeopleWelcome", "storage": "sqlite", "models": ["openai", "unstoppable"]}

@app.post("/api/v1/users")
async def create_user(payload: UserRequest):
    return {"user": store.upsert_user(payload.user_id, payload.username)}

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str):
    user = store.get_user(user_id)
    if not user: raise HTTPException(404, "User not found")
    return {"user": user}

@app.put("/api/v1/users/{user_id}/subscription")
async def set_subscription(user_id: str, payload: SubscriptionRequest):
    return {"user": store.set_premium(user_id, payload.premium)}

def media_url(path: Path) -> str:
    return f"/media/{path.relative_to(MEDIA_ROOT).as_posix()}"

async def generate_character_views(ai_id: str, source_path: Path) -> None:
    ai = store.get_ai(ai_id)
    if not ai: return
    store.update_generation(ai_id, "generating")
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("IMAGE_GENERATION_TIMEOUT", "900"))) as client:
            response = await client.post(f"{IMAGE_GENERATOR_URL.rstrip('/')}/v1/generate-character-views", json={
                "character_id": ai_id, "reference_image_path": str(source_path), "name": ai["name"],
                "persona": ai["persona"], "output_dir": str(MEDIA_ROOT / "generated" / ai_id),
            })
        response.raise_for_status()
        payload = response.json()
        image_urls = {key: media_url(Path(value)) for key, value in payload["images"].items()}
        store.update_generation(ai_id, "ready", profile_image_url=image_urls["avatar_front_face_url"], **image_urls)
    except Exception as error:
        store.update_generation(ai_id, "failed", error=str(error)[:1000])

def source_path_for_ai(ai: dict) -> Path | None:
    source_url = ai.get("original_avatar_url")
    if not source_url or not source_url.startswith("/media/"): return None
    path = (MEDIA_ROOT / source_url.removeprefix("/media/")).resolve()
    return path if MEDIA_ROOT.resolve() in path.parents and path.is_file() else None

def character_description(ai: dict) -> str:
    return f"{ai['name']}: {ai['persona']}"

@app.post("/api/v1/ais")
async def create_ai(
    background_tasks: BackgroundTasks,
    owner_id: str = Form(...), name: str = Form(...), persona: str = Form(...),
    model: Literal["openai", "unstoppable"] = Form("openai"), is_public: bool = Form(False),
    avatar: UploadFile | None = File(None),
):
    store.upsert_user(owner_id, owner_id)
    if model == "unstoppable" and not store.get_user(owner_id)["is_premium"]:
        raise HTTPException(403, "Unstoppable is a Premium feature. Complete subscription first.")
    avatar_url, source_path, status = None, None, "not_requested"
    if avatar:
        if not (avatar.content_type or "").startswith("image/"): raise HTTPException(422, "Avatar must be an image.")
        suffix = Path(avatar.filename or "avatar.png").suffix.lower() or ".png"
        source_path = MEDIA_ROOT / "uploads" / f"{uuid.uuid4()}{suffix}"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("wb") as buffer: shutil.copyfileobj(avatar.file, buffer)
        avatar_url, status = media_url(source_path), "pending"
    ai = store.create_ai(owner_id, name, persona, model, is_public, avatar_url, status)
    if source_path: background_tasks.add_task(generate_character_views, ai["id"], source_path)
    return {"ai": ai}

@app.get("/api/v1/ais/owner/{owner_id}")
async def owner_ais(owner_id: str):
    return {"ais": store.list_ais(owner_id=owner_id)}

@app.get("/api/v1/ais/public")
async def public_ais():
    return {"ais": store.list_ais(public_only=True)}

@app.get("/api/v1/ais/{ai_id}")
async def get_ai(ai_id: str):
    ai = store.get_ai(ai_id)
    if not ai: raise HTTPException(404, "AI not found")
    return {"ai": ai}

@app.get("/api/v1/ais/{ai_id}/images")
async def ai_images(ai_id: str):
    ai = store.get_ai(ai_id)
    if not ai: raise HTTPException(404, "AI not found")
    fields = ("original_avatar_url", "avatar_front_face_url", "avatar_side_face_url", "full_body_front_url", "full_body_side_url", "full_body_back_url", "profile_image_url")
    return {"status": ai["image_generation_status"], "error": ai["image_generation_error"], "images": {field: ai[field] for field in fields}}

@app.post("/api/v1/ais/{ai_id}/images/regenerate")
async def regenerate_ai_images(ai_id: str, background_tasks: BackgroundTasks):
    ai = store.get_ai(ai_id)
    if not ai: raise HTTPException(404, "AI not found")
    source_path = source_path_for_ai(ai)
    if not source_path: raise HTTPException(422, "This character has no stored avatar reference image.")
    store.update_generation(ai_id, "pending")
    background_tasks.add_task(generate_character_views, ai_id, source_path)
    return {"ai": store.get_ai(ai_id)}

@app.get("/media/{media_path:path}")
async def get_media(media_path: str):
    path = (MEDIA_ROOT / media_path).resolve()
    if MEDIA_ROOT.resolve() not in path.parents or not path.is_file(): raise HTTPException(404, "Media not found")
    return FileResponse(path)

@app.post("/api/v1/images/generate")
async def generate_image_scene(request: ImageSceneRequest):
    characters = []
    for ai_id in request.ai_ids:
        ai = store.get_ai(ai_id)
        if not ai: raise HTTPException(404, f"Character {ai_id} was not found")
        characters.append(ai)
    reference = next((source_path_for_ai(ai) for ai in characters if source_path_for_ai(ai)), None)
    if not reference: raise HTTPException(422, "Select a character with an uploaded avatar reference before generating an image.")
    scene_id = str(uuid.uuid4())
    destination = MEDIA_ROOT / "scenes" / f"{scene_id}.png"
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("IMAGE_GENERATION_TIMEOUT", "900"))) as client:
            response = await client.post(f"{IMAGE_GENERATOR_URL.rstrip('/')}/v1/generate-scene", json={
                "scene_id": scene_id, "reference_image_path": str(reference),
                "character_descriptions": [character_description(ai) for ai in characters],
                "user_prompt": request.prompt, "output_path": str(destination),
            })
        response.raise_for_status()
        return {"image_url": media_url(destination), "characters": [ai["name"] for ai in characters], "prompt": response.json().get("prompt")}
    except httpx.HTTPStatusError as error:
        raise HTTPException(error.response.status_code, error.response.text) from error
    except httpx.HTTPError as error:
        raise HTTPException(503, f"Image generator is unavailable: {error}") from error

@app.put("/api/v1/ais/{ai_id}/persona")
async def update_persona(ai_id: str, payload: PersonaRequest):
    if not store.update_persona(ai_id, payload.persona): raise HTTPException(404, "AI not found")
    return {"ai": store.get_ai(ai_id)}

async def generate(ai: dict, messages: list[dict]) -> dict:
    if ai["model"] == "unstoppable":
        url, key = os.getenv("UNSTOPPABLE_LLM_URL"), os.getenv("UNSTOPPABLE_LLM_API_KEY")
        if not url or not key: raise HTTPException(503, "Unstoppable model is not configured.")
        max_tokens = configured_int("UNSTOPPABLE_MAX_TOKENS", 2048, 128, 8192)
        history_chars = configured_int("UNSTOPPABLE_HISTORY_MAX_CHARS", 32000, 4000, 200000)
        timeout = configured_int("UNSTOPPABLE_TIMEOUT_SECONDS", 300, 30, 1800)
        request_body = {"persona_prompt": ai["persona"], "messages": trim_messages(messages, history_chars), "max_tokens": max_tokens, "temperature": 0.7}
        payload, reply = None, None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.post(f"{url.rstrip('/')}/v1/unstoppable/chat", headers={"X-API-Key": key}, json=request_body)
            except httpx.TimeoutException as error:
                raise HTTPException(504, f"Unstoppable timed out after {timeout} seconds.") from error
            except httpx.HTTPError as error:
                raise HTTPException(502, f"Unstoppable request failed: {error}") from error
            if response.status_code == 401: raise HTTPException(503, "Unstoppable model authentication failed.")
            if response.is_redirect: raise HTTPException(504, "Unstoppable deferred the response, but its result redirect could not be followed.")
            if response.is_error: raise HTTPException(502, "Unstoppable model could not complete the request.")
            try:
                payload = response.json()
            except ValueError as error:
                if attempt == 0: continue
                raise HTTPException(502, "Unstoppable returned an empty or invalid response after retrying.") from error
            reply = payload.get("response")
            if isinstance(reply, str) and reply.strip(): break
            if attempt == 1: raise HTTPException(502, "Unstoppable returned an empty response after retrying.")
        usage = payload.get("usage") or {}
        finish_reason = payload.get("finish_reason")
        if not finish_reason: finish_reason = "length" if usage.get("completion_tokens", 0) >= max_tokens - 1 else "stop"
        return {"content": reply, "finish_reason": finish_reason, "truncated": finish_reason == "length", "usage": usage}
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise HTTPException(503, "OpenAI is not configured.")
    response = OpenAI(api_key=key).chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "system", "content": ai["persona"]}, *messages])
    choice = response.choices[0]
    return {"content": choice.message.content or "", "finish_reason": choice.finish_reason, "truncated": choice.finish_reason == "length", "usage": response.usage.model_dump() if response.usage else {}}

@app.post("/api/v1/agent/chat")
async def chat(payload: ChatRequest):
    ai = store.get_ai(payload.ai_id)
    if not ai: raise HTTPException(404, "AI not found")
    history = store.history(payload.ai_id, payload.user_id)
    messages = [{"role": row["role"], "content": row["content"]} for row in history] + [{"role": "user", "content": payload.message}]
    generation = await generate(ai, messages); reply = generation["content"]
    user_message = store.add_message(payload.ai_id, payload.user_id, "user", payload.message)
    assistant_message = store.add_message(payload.ai_id, payload.user_id, "assistant", reply)
    return {"reply": reply, "messages": [user_message, assistant_message], "model": ai["model"], **{key: generation[key] for key in ("finish_reason", "truncated", "usage")}}

@app.get("/api/v1/conversations/{ai_id}/{user_id}")
async def conversation(ai_id: str, user_id: str):
    return {"messages": store.history(ai_id, user_id)}
