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
