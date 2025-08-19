# GPU-enabled base
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Keep apt quiet & Python stdout unbuffered
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
 && python -m pip install --upgrade pip

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest
COPY . .

# Start training by default
CMD ["python", "train.py"]
