import runpod
import torch
import io
import base64
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load model once at cold start (cached across warm invocations)
MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.to("cuda" if torch.cuda.is_available() else "cpu")


def handler(event):
    """RunPod serverless handler — receives base64 audio, returns segments."""
    audio_b64 = event["input"]["audio"]
    audio_bytes = base64.b64decode(audio_b64)

    # Decode audio to numpy array at 16kHz (Whisper's expected sample rate)
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Process with Whisper
    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(model.device)

    # Generate with timestamps
    predicted_ids = model.generate(
        input_features,
        return_timestamps=True,
        language="ar",
        task="transcribe",
    )

    result = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, output_offsets=True
    )

    # Extract segments with timestamps
    segments = []
    if result and len(result) > 0:
        for chunk in result[0].get("offsets", []):
            segments.append({
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1],
                "text": chunk["text"],
            })

    # Fallback: if no offsets, return full text as single segment
    if not segments and result:
        text = result[0] if isinstance(result[0], str) else result[0].get("text", "")
        if text:
            duration = len(audio_array) / 16000
            segments.append({"start": 0.0, "end": duration, "text": text})

    return {"segments": segments}


runpod.serverless.start({"handler": handler})
