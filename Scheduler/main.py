"""PeopleWelcome's local, persona-based chat API."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from .storage import Store

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
store = Store(Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "peoplewelcome.db")))

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

@app.post("/api/v1/ais")
async def create_ai(payload: AIRequest):
    store.upsert_user(payload.owner_id, payload.owner_id)
    if payload.model == "unstoppable" and not store.get_user(payload.owner_id)["is_premium"]:
        raise HTTPException(403, "Unstoppable is a Premium feature. Complete subscription first.")
    return {"ai": store.create_ai(**payload.model_dump())}

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

@app.put("/api/v1/ais/{ai_id}/persona")
async def update_persona(ai_id: str, payload: PersonaRequest):
    if not store.update_persona(ai_id, payload.persona): raise HTTPException(404, "AI not found")
    return {"ai": store.get_ai(ai_id)}

async def generate(ai: dict, messages: list[dict]) -> str:
    if ai["model"] == "unstoppable":
        url, key = os.getenv("UNSTOPPABLE_LLM_URL"), os.getenv("UNSTOPPABLE_LLM_API_KEY")
        if not url or not key: raise HTTPException(503, "Unstoppable model is not configured.")
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{url.rstrip('/')}/v1/unstoppable/chat", headers={"X-API-Key": key}, json={"persona_prompt": ai["persona"], "messages": messages, "max_tokens": 512, "temperature": 0.7})
        if response.status_code == 401: raise HTTPException(503, "Unstoppable model authentication failed.")
        if response.is_error: raise HTTPException(502, "Unstoppable model could not complete the request.")
        return response.json()["response"]
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise HTTPException(503, "OpenAI is not configured.")
    response = OpenAI(api_key=key).chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "system", "content": ai["persona"]}, *messages])
    return response.choices[0].message.content or ""

@app.post("/api/v1/agent/chat")
async def chat(payload: ChatRequest):
    ai = store.get_ai(payload.ai_id)
    if not ai: raise HTTPException(404, "AI not found")
    history = store.history(payload.ai_id, payload.user_id)
    messages = [{"role": row["role"], "content": row["content"]} for row in history] + [{"role": "user", "content": payload.message}]
    reply = await generate(ai, messages)
    user_message = store.add_message(payload.ai_id, payload.user_id, "user", payload.message)
    assistant_message = store.add_message(payload.ai_id, payload.user_id, "assistant", reply)
    return {"reply": reply, "messages": [user_message, assistant_message], "model": ai["model"]}

@app.get("/api/v1/conversations/{ai_id}/{user_id}")
async def conversation(ai_id: str, user_id: str):
    return {"messages": store.history(ai_id, user_id)}
