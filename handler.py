import runpod
import torch
import io
import base64
from transformers import pipeline
import librosa

# Load model once at cold start using pipeline (handles long audio automatically)
MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=device,
    chunk_length_s=30,
    stride_length_s=5,
)


def handler(event):
    """RunPod serverless handler — receives base64 audio, returns segments."""
    audio_b64 = event["input"]["audio"]
    audio_bytes = base64.b64decode(audio_b64)

    # Decode audio to numpy array at 16kHz (Whisper's expected sample rate)
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Transcribe with timestamps — pipeline handles chunking automatically
    result = pipe(
        {"raw": audio_array, "sampling_rate": 16000},
        return_timestamps=True,
        generate_kwargs={"language": "ar", "task": "transcribe"},
    )

    # Extract segments with timestamps
    segments = []
    if result and "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (0, 0))
            segments.append({
                "start": ts[0] if ts[0] is not None else 0,
                "end": ts[1] if ts[1] is not None else 0,
                "text": chunk["text"],
            })

    # Fallback: if no chunks, return full text as single segment
    if not segments and result:
        text = result.get("text", "")
        if text:
            duration = len(audio_array) / 16000
            segments.append({"start": 0.0, "end": duration, "text": text})

    return {"segments": segments}


runpod.serverless.start({"handler": handler})
