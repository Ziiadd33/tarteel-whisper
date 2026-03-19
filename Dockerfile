FROM python:3.11-slim

WORKDIR /app

# Install torch >= 2.6 with CUDA 12.4 (required by transformers for CVE-2025-32434)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

# Remove torchvision/torchaudio if present (not needed)
RUN pip uninstall -y torchvision torchaudio 2>/dev/null; true

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    "transformers>=4.40,<5" \
    librosa \
    soundfile

COPY handler.py .

# Pre-download whisper-large-v3 during build (~3GB, slower build but faster cold starts)
RUN python -c "from transformers import pipeline; \
    pipeline('automatic-speech-recognition', model='openai/whisper-large-v3', device='cpu')"

CMD ["python", "-u", "handler.py"]
