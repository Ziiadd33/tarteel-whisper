FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Upgrade torch to >= 2.6 (required by transformers >= 5.x for CVE-2025-32434)
RUN pip install --no-cache-dir --upgrade torch

RUN pip install --no-cache-dir \
    runpod \
    transformers \
    librosa \
    soundfile \
    whisperx

COPY handler.py .

# Pre-download Whisper model during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran')"

# Pre-download wav2vec2 Arabic alignment model during build
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ar', device='cpu')"

CMD ["python", "-u", "handler.py"]
