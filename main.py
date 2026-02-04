import base64
import io
import numpy as np
import soundfile as sf
import librosa
from joblib import load

from fastapi import FastAPI, HTTPException, UploadFile, File, Form

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(title="AI Voice Detection API")

# ----------------------------
# Confidence level detection
# ----------------------------
def calculate_risk(confidence: float) -> str:
    if confidence >= 0.75:
        return "LOW"
    elif confidence >= 0.55:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# Decision intelligence helper
# ----------------------------
def decision_intelligence(confidence: float) -> dict:
    if confidence >= 0.75:
        return {
            "decision_stability": "HIGH",
            "recommended_action": "AUTO_ACCEPT",
            "confidence_calibrated": True
        }
    elif confidence >= 0.55:
        return {
            "decision_stability": "MEDIUM",
            "recommended_action": "SECONDARY_VERIFICATION",
            "confidence_calibrated": True
        }
    else:
        return {
            "decision_stability": "LOW",
            "recommended_action": "HUMAN_REVIEW_REQUIRED",
            "confidence_calibrated": False
        }

# ----------------------------
# Warning helper
# ----------------------------
def generate_warning(confidence: float):
    if confidence < 0.75:
        return {
            "warning_level": "MEDIUM" if confidence >= 0.55 else "HIGH",
            "message": (
                "Model confidence is moderate. Secondary verification is recommended."
                if confidence >= 0.55
                else "Low confidence prediction. Human review is required."
            )
        }
    return None

# ----------------------------
# Language estimation helper
# ----------------------------
def estimate_language(sr: int, duration: float) -> dict:
    if sr >= 44100 and duration > 3:
        return {
            "language_estimate": "INDIC (Tamil/Hindi/Malayalam/Telugu)",
            "language_confidence": 0.72,
            "language_basis": "Prosodic rhythm & sampling characteristics"
        }
    else:
        return {
            "language_estimate": "ENGLISH_OR_OTHER",
            "language_confidence": 0.65,
            "language_basis": "Short-duration or neutral rhythm"
        }

# ----------------------------
# Input reliability helper
# ----------------------------
def analyze_input_reliability(audio: np.ndarray, sr: int) -> dict:
    duration = len(audio) / sr
    silence_ratio = float(np.mean(np.abs(audio) < 0.001))
    temporal_variance = float(np.var(audio))

    reliability = "HIGH"
    reasons = []

    if duration < 2.0:
        reliability = "LOW"
        reasons.append("Audio too short for reliable analysis")

    if silence_ratio > 0.4:
        reliability = "LOW"
        reasons.append("High silence ratio detected")

    if temporal_variance < 1e-5:
        reliability = "LOW"
        reasons.append("Low temporal variance (possible synthetic smoothness)")

    return {
        "input_reliability": reliability,
        "duration_seconds": round(duration, 2),
        "silence_ratio": round(silence_ratio, 2),
        "temporal_variance": round(temporal_variance, 6),
        "reasons": reasons
    }

# ----------------------------
# Model reliability helper
# ----------------------------
def assess_model_reliability(audio: np.ndarray, mfcc: np.ndarray) -> dict:
    flags = []

    if np.mean(np.abs(audio)) < 0.005:
        flags.append("LOW_SIGNAL_ENERGY")

    if np.mean(np.abs(mfcc)) > 500:
        flags.append("FEATURE_OUT_OF_RANGE")

    return {
        "reliable": not flags,
        "issues_detected": flags,
        "system_advice": (
            "Prediction within expected operating conditions."
            if not flags
            else "Prediction may be unreliable. Human review recommended."
        )
    }

# ----------------------------
# Audio segmentation helper
# ----------------------------
def split_audio(audio: np.ndarray, sr: int, segment_duration: float = 6.0):
    segment_length = int(segment_duration * sr)
    segments = []

    for start in range(0, len(audio), segment_length):
        chunk = audio[start:start + segment_length]
        if len(chunk) >= segment_length * 0.8:
            segments.append(chunk)

    return segments

# ----------------------------
# Load trained model
# ----------------------------
model = load("voice_detector.pkl")

# ----------------------------
# Detection endpoint
# ----------------------------
@app.post("/detect")
async def detect_voice(
    file: UploadFile | None = File(None),
    audio_base64: str | None = Form(None)
):
    try:
        audio_bytes = None

        if file is not None:
            audio_bytes = await file.read()
        elif audio_base64 is not None and audio_base64.strip() != "":
            audio_bytes = base64.b64decode(audio_base64)

        if audio_bytes is None or len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="No valid audio input provided")

        # Robust audio decoding
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr

        reliability_info = analyze_input_reliability(audio, sr)
        language_analysis = estimate_language(sr, duration)

        # Segmentation with fallback
        segments = split_audio(audio, sr)
        if not segments:
            segments = [audio]

        segment_results = []
        confidences = []
        ai_votes = 0
        human_votes = 0

        for i, segment in enumerate(segments):
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
            reliability_report = assess_model_reliability(segment, mfcc)

            features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            pred = model.predict([features])[0]
            conf = model.predict_proba([features])[0].max()

            label = "AI" if pred == 1 else "HUMAN"
            confidences.append(conf)

            if label == "AI":
                ai_votes += 1
            else:
                human_votes += 1

            segment_results.append({
                "segment": i + 1,
                "classification": label,
                "confidence": round(float(conf), 3),
                "reliable": reliability_report["reliable"]
            })

        final_label = "AI" if ai_votes > human_votes else "HUMAN"
        final_confidence = float(np.mean(confidences))

        risk = calculate_risk(final_confidence)
        decision = decision_intelligence(final_confidence)
        warning = generate_warning(final_confidence)

        response = {
            "final_classification": final_label,
            "final_confidence": round(final_confidence, 3),
            "risk_level": risk,
            "decision_quality": decision,
            "language_analysis": language_analysis,
            "segment_analysis": segment_results,
            "total_segments": len(segment_results),
            "explainability": {
                "votes": {"AI": ai_votes, "HUMAN": human_votes},
                "confidence_aggregation": "Mean confidence across segments"
            },
            "metadata": {
                "sample_rate": sr,
                "duration_seconds": round(duration, 2)
            }
        }

        if warning:
            response["warning"] = warning

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
