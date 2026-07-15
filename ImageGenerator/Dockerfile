FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY ImageGenerator/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY ImageGenerator /app/ImageGenerator
CMD ["python3", "-m", "uvicorn", "ImageGenerator.main:app", "--host", "0.0.0.0", "--port", "8001"]
