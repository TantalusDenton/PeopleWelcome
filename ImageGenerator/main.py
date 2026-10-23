"""Persona- and pose-controlled image generation for PeopleWelcome."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field

app = FastAPI(title="PeopleWelcome Image Generator", version="2.0")
logger = logging.getLogger("peoplewelcome.image_generator")
pipeline: StableDiffusionControlNetPipeline | None = None
generation_lock = threading.Lock()

BASE_MODEL_ID = os.getenv("DIFFUSERS_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "lllyasviel/control_v11p_sd15_openpose")
REFERENCE_SIZE = int(os.getenv("IMAGE_GENERATION_SIZE", "512"))
FULL_BODY_WIDTH = int(os.getenv("CHARACTER_VIEW_WIDTH", "512"))
FULL_BODY_HEIGHT = int(os.getenv("CHARACTER_VIEW_HEIGHT", "768"))
SCENE_WIDTH = int(os.getenv("SCENE_IMAGE_WIDTH", "768"))
SCENE_HEIGHT = int(os.getenv("SCENE_IMAGE_HEIGHT", "512"))
INFERENCE_STEPS = int(os.getenv("IMAGE_INFERENCE_STEPS", "18"))
MINIMUM_FREE_VRAM_MIB = int(os.getenv("MINIMUM_FREE_VRAM_MIB", "1800"))


class CharacterViewsRequest(BaseModel):
    character_id: str
    reference_image_path: str
    name: str
    persona: str
    output_dir: str


class SceneRequest(BaseModel):
    scene_id: str
    reference_image_paths: list[str] = Field(default_factory=list)
    reference_image_path: str | None = None
    character_descriptions: list[str]
    user_prompt: str = ""
    output_path: str


@dataclass(frozen=True)
class Shot:
    field: str
    pose: str
    prompt: str
    negative: str
    width: int
    height: int
    control_scale: float


SHOTS = (
    Shot(
        "full_body_front_url", "body_front",
        "single character, complete head-to-toe full body, straight front view, standing neutral A-pose, both feet visible, centered character turnaround",
        "close-up, cropped feet, side view, rear view, back view, profile, seated, multiple people",
        FULL_BODY_WIDTH, FULL_BODY_HEIGHT, 0.88,
    ),
    Shot(
        "full_body_side_url", "body_side",
        "single character, complete head-to-toe full body, exact right-side orthographic profile, standing neutral pose, both feet visible, character turnaround",
        "close-up, cropped feet, front view, rear view, three-quarter view, looking at camera, multiple people",
        FULL_BODY_WIDTH, FULL_BODY_HEIGHT, 0.92,
    ),
    Shot(
        "full_body_back_url", "body_back",
        "single character, complete head-to-toe full body, exact rear view from behind, back of head visible, standing neutral A-pose, both feet visible, character turnaround",
        "close-up, cropped feet, front view, side view, three-quarter view, face visible, looking at camera, multiple people",
        FULL_BODY_WIDTH, FULL_BODY_HEIGHT, 0.94,
    ),
    Shot(
        "avatar_front_face_url", "face_front",
        "single character close-up head-and-shoulders portrait, exact front view, direct eye contact, centered face",
        "full body, side profile, rear view, three-quarter view, cropped head, multiple people",
        REFERENCE_SIZE, REFERENCE_SIZE, 0.72,
    ),
    Shot(
        "avatar_side_face_url", "face_side",
        "single character close-up head-and-shoulders portrait, exact right-side facial profile, clear nose and jaw silhouette, looking right",
        "full body, front-facing, rear view, three-quarter view, looking at camera, cropped head, multiple people",
        REFERENCE_SIZE, REFERENCE_SIZE, 0.88,
    ),
)

COMMON_NEGATIVE = (
    "collage, contact sheet, split screen, duplicate character, extra person, extra limbs, missing limbs, "
    "bad anatomy, malformed hands, distorted face, low resolution, blurry, black image, silhouette, text, watermark"
)

# OpenPose COCO body layout: nose, neck, right arm, left arm, right leg,
# left leg, eyes and ears. Colors follow the OpenPose conditioning convention.
LIMBS = ((1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9),
         (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17))
COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
          (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
          (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
          (255, 0, 255), (255, 0, 170), (255, 0, 85))


def get_pipeline() -> StableDiffusionControlNetPipeline:
    global pipeline
    if pipeline is not None:
        return pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the image generation service.")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # Whole-model CPU offload fits SD1.5 + one ControlNet in 6 GB while avoiding
    # the per-layer transfers that made five-view sheets take excessively long.
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing("max")
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline


def prepare_reference(path: Path, size: int = REFERENCE_SIZE) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(source.convert("RGB"), (size, size), Image.Resampling.LANCZOS)


def compose_references(paths: list[Path]) -> Image.Image:
    images = [prepare_reference(path, REFERENCE_SIZE // 2) for path in paths[:4]]
    if len(images) == 1:
        return prepare_reference(paths[0])
    canvas = Image.new("RGB", (REFERENCE_SIZE, REFERENCE_SIZE), "white")
    positions = ((0, 0), (REFERENCE_SIZE // 2, 0), (0, REFERENCE_SIZE // 2),
                 (REFERENCE_SIZE // 2, REFERENCE_SIZE // 2))
    for image, position in zip(images, positions):
        canvas.paste(image, position)
    return canvas


def _body_points(kind: str) -> dict[int, tuple[float, float]]:
    if kind == "body_side":
        return {
            0: (.55, .11), 1: (.50, .22), 2: (.49, .25), 3: (.48, .43), 4: (.48, .62),
            5: (.53, .25), 6: (.55, .43), 7: (.56, .61), 8: (.49, .50), 9: (.47, .72),
            10: (.45, .94), 11: (.53, .50), 12: (.55, .72), 13: (.57, .94),
            14: (.57, .10), 15: (.53, .10), 16: (.59, .12), 17: (.50, .12),
        }
    points = {
        0: (.50, .11), 1: (.50, .22), 2: (.36, .25), 3: (.31, .43), 4: (.29, .62),
        5: (.64, .25), 6: (.69, .43), 7: (.71, .62), 8: (.43, .50), 9: (.42, .72),
        10: (.40, .94), 11: (.57, .50), 12: (.58, .72), 13: (.60, .94),
    }
    if kind != "body_back":
        points.update({14: (.47, .10), 15: (.53, .10), 16: (.43, .12), 17: (.57, .12)})
    return points


def _face_points(kind: str) -> dict[int, tuple[float, float]]:
    if kind == "face_side":
        return {
            0: (.61, .34), 1: (.49, .70), 2: (.25, .78), 3: (.16, .95),
            5: (.73, .78), 6: (.84, .95), 14: (.56, .30), 16: (.43, .37),
        }
    return {
        0: (.50, .34), 1: (.50, .70), 2: (.24, .78), 3: (.14, .96),
        5: (.76, .78), 6: (.86, .96), 14: (.43, .30), 15: (.57, .30),
        16: (.33, .35), 17: (.67, .35),
    }


def _draw_pose(canvas: Image.Image, points: dict[int, tuple[float, float]], box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    box_width, box_height = right - left, bottom - top
    pixels = {index: (int(left + x * box_width), int(top + y * box_height)) for index, (x, y) in points.items()}
    draw = ImageDraw.Draw(canvas)
    line_width = max(4, min(canvas.size) // 85)
    radius = max(4, min(canvas.size) // 110)
    for limb_index, (start, end) in enumerate(LIMBS):
        if start in pixels and end in pixels:
            draw.line((pixels[start], pixels[end]), fill=COLORS[limb_index], width=line_width)
    for index, point in pixels.items():
        color = COLORS[index % len(COLORS)]
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)


def render_pose(kind: str, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "black")
    points = _face_points(kind) if kind.startswith("face_") else _body_points(kind)
    _draw_pose(canvas, points, (0, 0, width, height))
    return canvas


def render_group_pose(count: int, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "black")
    slot_width = width / max(count, 1)
    for index in range(count):
        left = int(index * slot_width)
        right = int((index + 1) * slot_width)
        inset = int(slot_width * .08)
        _draw_pose(canvas, _body_points("body_front"), (left + inset, 0, right - inset, height))
    return canvas


def character_identity_prompt(name: str, persona: str) -> str:
    # Stable Diffusion 1.5 has a 77-token text window. Keep persona details near
    # the start so they influence every generated viewpoint.
    compact_persona = " ".join(persona.split())[:240]
    return f"{name}. Character appearance and persona: {compact_persona}"


def deterministic_seed(identifier: str, view: str) -> int:
    digest = hashlib.sha256(f"{identifier}:{view}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def require_free_vram() -> tuple[int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mib, total_mib = free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)
    if free_mib < MINIMUM_FREE_VRAM_MIB:
        raise RuntimeError(
            f"Only {free_mib} MiB of {total_mib} MiB GPU memory is free. "
            f"Free at least {MINIMUM_FREE_VRAM_MIB} MiB by stopping other GPU workloads, then retry."
        )
    return free_mib, total_mib


def generate_controlled_image(
    generator: StableDiffusionControlNetPipeline,
    *,
    prompt: str,
    negative_prompt: str,
    pose: Image.Image,
    seed: int,
    control_scale: float = 1.0,
) -> Image.Image:
    return generator(
        prompt=prompt,
        negative_prompt=f"{COMMON_NEGATIVE}, {negative_prompt}",
        image=pose,
        width=pose.width,
        height=pose.height,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=7.0,
        controlnet_conditioning_scale=control_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "model": BASE_MODEL_ID,
        "controlnet": CONTROLNET_MODEL_ID,
        "generation_method": "persona+openpose-controlnet",
        "safety_checker_enabled": False,
        "view_size": [FULL_BODY_WIDTH, FULL_BODY_HEIGHT],
        "free_vram_mib": torch.cuda.mem_get_info()[0] // (1024 * 1024) if torch.cuda.is_available() else 0,
        "minimum_free_vram_mib": MINIMUM_FREE_VRAM_MIB,
    }


@app.post("/v1/generate-character-views")
async def generate_character_views(request: CharacterViewsRequest):
    source = Path(request.reference_image_path)
    if not source.is_file():
        raise HTTPException(404, "Reference image does not exist in shared media storage.")
    try:
        with generation_lock:
            require_free_vram()
            generator = get_pipeline()
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            identity = character_identity_prompt(request.name, request.persona)
            images = {}
            for shot in SHOTS:
                pose = render_pose(shot.pose, shot.width, shot.height)
                image = generate_controlled_image(
                    generator,
                    prompt=f"{shot.prompt}. {identity}. detailed polished character design, plain studio background",
                    negative_prompt=shot.negative,
                    pose=pose,
                    seed=deterministic_seed(request.character_id, shot.field),
                    control_scale=shot.control_scale,
                )
                destination = output_dir / f"{shot.field}.png"
                image.save(destination)
                images[shot.field] = str(destination)
        return {"images": images, "method": "persona+openpose-controlnet"}
    except RuntimeError as error:
        logger.exception("Character-view generation failed")
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        logger.exception("Character-view generation failed")
        raise HTTPException(500, f"Generation failed: {error}") from error


@app.post("/v1/generate-scene")
async def generate_scene(request: SceneRequest):
    raw_paths = request.reference_image_paths or ([request.reference_image_path] if request.reference_image_path else [])
    paths = [Path(path) for path in raw_paths]
    if not paths or any(not path.is_file() for path in paths):
        raise HTTPException(404, "Every selected character needs a usable reference image.")
    try:
        with generation_lock:
            require_free_vram()
            generator = get_pipeline()
            pose = render_group_pose(len(request.character_descriptions), SCENE_WIDTH, SCENE_HEIGHT)
            details = " ".join(request.character_descriptions)
            user_prompt = request.user_prompt.strip() or "A polished scene with all selected characters together"
            prompt = (
                f"{user_prompt}. Exactly {len(request.character_descriptions)} distinct characters, all visible. "
                f"Character appearances: {details}. cohesive detailed character art, no text"
            )
            output = Path(request.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            image = generate_controlled_image(
                generator,
                prompt=prompt,
                negative_prompt="missing character, duplicate character, close-up, cropped body",
                pose=pose,
                seed=deterministic_seed(request.scene_id, "scene"),
                control_scale=0.72,
            )
            image.save(output)
        return {"image_path": str(output), "prompt": prompt, "method": "persona+openpose-controlnet"}
    except RuntimeError as error:
        logger.exception("Scene generation failed")
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        logger.exception("Scene generation failed")
        raise HTTPException(500, f"Scene generation failed: {error}") from error
