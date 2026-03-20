FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Upgrade torch + torchvision together (must be compatible versions)
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio

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
