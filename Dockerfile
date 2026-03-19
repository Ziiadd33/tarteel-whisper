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

# Pre-download models during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    GenerationConfig.from_pretrained('openai/whisper-base')"

CMD ["python", "-u", "handler.py"]
