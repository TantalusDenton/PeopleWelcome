# PeopleWelcome

PeopleWelcome is a local, persona-based AI chat platform. Each character stores
its persona, model choice, and conversation history in local SQLite storage.

OpenAI characters are available to all users. The Unstoppable model
(`huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated`) is a simulated Premium
feature backed by the supplied Modal endpoint.

## Run locally with Docker

1. Copy `.env.example` to `.env` and add the credentials for each model you use.
2. Run `docker compose up --build`.
3. Open http://localhost:13000. The API is available at http://localhost:18000.

The frontend never receives model API keys. SQLite data persists in the
`peoplewelcome-data` Docker volume.

## Premium flow

Open **Settings**, confirm the simulated subscription, then create an AI. Premium
users see a choice between OpenAI and Unstoppable. Every new character requires
a persona, which is sent with each model request.

## Character view sheets

The `image-generator` service is a GPU-only Python Diffusers API. Uploading a
reference avatar when creating a character stores the source image locally,
creates the character as `pending`, then starts a background generation job.
The job uses Stable Diffusion 1.5 img2img at 512x512 with the same reference for five shot-specific views:
front/side face plus front/side/back full body. The generated front-face image
becomes the character's `profile_image_url`.

Run the full stack with `docker compose up --build`. Docker needs NVIDIA GPU
support enabled for `image-generator`; it downloads `DIFFUSERS_MODEL_ID` on its
first request. The service exposes `GET /health` and
`POST /v1/generate-character-views` internally. The initial implementation uses
maintainable reference-guided img2img. The default model and resolution are sized
for a 6 GB NVIDIA GPU; an IP-Adapter can be added later if tighter identity locking
is needed.
