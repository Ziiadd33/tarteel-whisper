FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Install whisperx and other deps first (whisperx pulls its own torch/torchvision)
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    librosa \
    soundfile \
    whisperx

# Force-reinstall compatible torch stack AFTER whisperx
# whisperx installs torchvision 0.25.0 which needs torch==2.10.0,
# but the CUDA base image only supports certain torch+CUDA combos.
# Pin to torch 2.4.x that matches the base image's CUDA 12.4.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

COPY handler.py .

# Pre-download Whisper model during build (faster cold starts)
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    WhisperProcessor.from_pretrained('tarteel-ai/whisper-base-ar-quran'); \
    WhisperForConditionalGeneration.from_pretrained('tarteel-ai/whisper-base-ar-quran')"

# Pre-download wav2vec2 Arabic alignment model during build
RUN python -c "import whisperx; whisperx.load_align_model(language_code='ar', device='cpu')"

CMD ["python", "-u", "handler.py"]
