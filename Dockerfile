FROM python:3.11-slim

WORKDIR /app

# Install torch CPU-first (lighter), then upgrade to CUDA version
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# Remove torchvision/torchaudio if present (not needed, causes conflicts)
RUN pip uninstall -y torchvision torchaudio 2>/dev/null; true

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    "transformers>=4.40,<5" \
    librosa \
    soundfile

COPY handler.py .

# Pre-download model during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran')"

CMD ["python", "-u", "handler.py"]
