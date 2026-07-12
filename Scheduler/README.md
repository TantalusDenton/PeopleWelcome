# Scheduler

The Scheduler is PeopleWelcome's local persona-chat API. It uses SQLite and
proxies model requests server-side so API keys remain private.

Run it with `uvicorn Scheduler.main:app --port 8000`, or use Docker Compose from
the repository root. Configure `OPENAI_API_KEY`, `UNSTOPPABLE_LLM_URL`, and
`UNSTOPPABLE_LLM_API_KEY` in `.env` as needed.
