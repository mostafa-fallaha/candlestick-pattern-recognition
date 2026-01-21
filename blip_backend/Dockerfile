FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app
RUN chmod +x /app/entrypoint.sh

# Environment defaults
ENV PORT=8000
ENV YOLO_PATH=/weights/yolo.pt
ENV BLIP_PATH=/weights/blip_model

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
