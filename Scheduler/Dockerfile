FROM python:3.12-slim
WORKDIR /app
COPY Scheduler/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY Scheduler /app/Scheduler
CMD ["uvicorn", "Scheduler.main:app", "--host", "0.0.0.0", "--port", "8000"]
