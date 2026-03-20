import runpod
import torch
import io
import base64
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import whisperx

# ── Device ────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load Whisper model once at cold start ─────────────────────────────
MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(device)

# ── Load alignment model once at cold start ───────────────────────────
align_model, align_metadata = whisperx.load_align_model(
    language_code="ar", device=device
)


def transcribe_handler(audio_array):
    """Original transcription mode."""
    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    predicted_ids = model.generate(
        input_features,
        return_timestamps=True,
        language="ar",
        task="transcribe",
    )

    result = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, output_offsets=True
    )

    segments = []
    if result and len(result) > 0:
        for chunk in result[0].get("offsets", []):
            segments.append({
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1],
                "text": chunk["text"],
            })

    if not segments and result:
        text = result[0] if isinstance(result[0], str) else result[0].get("text", "")
        if text:
            duration = len(audio_array) / 16000
            segments.append({"start": 0.0, "end": duration, "text": text})

    return {"segments": segments}


def align_handler(audio_array, text, language="ar"):
    """
    Forced alignment mode.
    Given audio + known text, return per-word timestamps.

    Input:
      audio_array: numpy array at 16kHz
      text: the known transcription text (e.g. Quran ayah text)
      language: language code (default "ar")

    Output:
      { "word_segments": [...], "segments": [...] }
    """
    duration = len(audio_array) / 16000

    # WhisperX expects a "transcript" with segments to align
    transcript_segments = [{
        "text": text,
        "start": 0.0,
        "end": duration,
    }]

    aligned = whisperx.align(
        transcript_segments,
        align_model,
        align_metadata,
        audio_array,
        device,
        return_char_alignments=False,
    )

    # Extract word-level timestamps
    word_segments = []
    for ws in aligned.get("word_segments", []):
        word_segments.append({
            "word": ws.get("word", ""),
            "start": ws.get("start", 0.0),
            "end": ws.get("end", 0.0),
            "score": ws.get("score", 0.0),
        })

    # Extract segment-level timestamps (aligned versions of input segments)
    segments = []
    for seg in aligned.get("segments", []):
        segments.append({
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
        })

    return {
        "word_segments": word_segments,
        "segments": segments,
    }


def handler(event):
    """RunPod serverless handler — routes to transcribe or align mode."""
    inp = event["input"]
    mode = inp.get("mode", "transcribe")

    audio_b64 = inp["audio"]
    audio_bytes = base64.b64decode(audio_b64)
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    if mode == "align":
        text = inp["text"]
        language = inp.get("language", "ar")
        return align_handler(audio_array, text, language)
    else:
        return transcribe_handler(audio_array)


runpod.serverless.start({"handler": handler})
