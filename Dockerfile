FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# 1. Upgrade torch stack to >= 2.6 from PyTorch's CUDA 12.4 index
#    (versions are paired correctly in this index)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Install whisperx WITHOUT deps so it can't break our torch versions
RUN pip install --no-cache-dir --no-deps whisperx

# 3. Install whisperx's actual dependencies (minus torch/torchvision/torchaudio)
#    and our other deps
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    librosa \
    soundfile \
    faster-whisper \
    pyannote.audio \
    nltk

COPY handler.py .

# Pre-download Whisper model during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran')"

# Pre-download wav2vec2 Arabic alignment model during build
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ar', device='cpu')"

CMD ["python", "-u", "handler.py"]
