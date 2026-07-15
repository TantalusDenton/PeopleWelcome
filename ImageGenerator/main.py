"""Diffusers-powered, reference-image character view generation service."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="PeopleWelcome Image Generator", version="1.0")
pipeline = None
IMAGE_SIZE = int(os.getenv("IMAGE_GENERATION_SIZE", "512"))

class CharacterViewsRequest(BaseModel):
    character_id: str
    reference_image_path: str
    name: str
    persona: str
    output_dir: str

class SceneRequest(BaseModel):
    scene_id: str
    reference_image_path: str
    character_descriptions: list[str]
    user_prompt: str = ""
    output_path: str

SHOT_PROMPTS = {
    "avatar_front_face_url": "clean close-up portrait, front-facing face, direct eye contact, centered composition",
    "avatar_side_face_url": "clean close-up portrait, side profile facing right, clear facial silhouette",
    "full_body_front_url": "full body character sheet, front view, standing neutral pose, centered",
    "full_body_side_url": "full body character sheet, side view facing right, standing neutral pose, centered",
    "full_body_back_url": "full body character sheet, back view, standing neutral pose, centered",
}

def character_identity_prompt(name: str, persona: str) -> str:
    return (
        f"{name}. Persona and appearance: {persona}. "
        "Same face, hair, skin tone, and identity as the reference image."
    )

def get_pipeline():
    global pipeline
    if pipeline is None:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required by the image generation service.")
        model = os.getenv("DIFFUSERS_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline = AutoPipelineForImage2Image.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=True)
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    return pipeline

def prepare_reference(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    edge = min(width, height)
    left, top = (width - edge) // 2, (height - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

@app.get("/health")
async def health():
    return {"status": "ok", "cuda_available": torch.cuda.is_available(), "model": os.getenv("DIFFUSERS_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"), "image_size": IMAGE_SIZE}

@app.post("/v1/generate-character-views")
async def generate_character_views(request: CharacterViewsRequest):
    source = Path(request.reference_image_path)
    if not source.is_file(): raise HTTPException(404, "Reference image does not exist in shared media storage.")
    try:
        generator = get_pipeline()
        reference = prepare_reference(source)
        output_dir = Path(request.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        identity = character_identity_prompt(request.name, request.persona)
        images = {}
        for field, shot in SHOT_PROMPTS.items():
            prompt = f"{shot}. {identity} Polished game character art, neutral background."
            image = generator(prompt=prompt, image=reference, strength=0.35, guidance_scale=7.0, num_inference_steps=35).images[0]
            destination = output_dir / f"{field}.png"
            image.save(destination)
            images[field] = str(destination)
        return {"images": images}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        raise HTTPException(500, f"Generation failed: {error}") from error

@app.post("/v1/generate-scene")
async def generate_scene(request: SceneRequest):
    source = Path(request.reference_image_path)
    if not source.is_file(): raise HTTPException(404, "A selected character has no usable reference image.")
    try:
        generator = get_pipeline()
        reference = prepare_reference(source)
        details = "\n".join(request.character_descriptions)
        user_prompt = request.user_prompt.strip() or "Create a polished character scene with all selected characters together."
        prompt = (
            f"Scene direction: {user_prompt}. Include every selected character together; do not omit anyone. "
            f"Selected character personas and appearance:\n{details}\n"
            "Preserve identity from the reference image. Polished cohesive character art, no text, no watermark."
        )
        output = Path(request.output_path); output.parent.mkdir(parents=True, exist_ok=True)
        image = generator(prompt=prompt, image=reference, strength=0.42, guidance_scale=7.5, num_inference_steps=35).images[0]
        image.save(output)
        return {"image_path": str(output), "prompt": prompt}
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        raise HTTPException(500, f"Scene generation failed: {error}") from error
