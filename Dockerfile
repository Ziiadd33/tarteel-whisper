FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    transformers \
    torch \
    librosa \
    soundfile

COPY handler.py .

# Pre-download model during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran')"

CMD ["python", "-u", "handler.py"]
