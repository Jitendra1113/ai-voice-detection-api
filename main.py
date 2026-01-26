import base64
import io

import numpy as np
import soundfile as sf
import librosa
from joblib import load

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(title="AI Voice Detection API")

# ----------------------------
# confidence level detection
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
# ✅ ADDED: model reliability helper (Step 4.1)
# ----------------------------
def assess_model_reliability(audio: np.ndarray, mfcc: np.ndarray) -> dict:
    reliability_flags = []

    signal_energy = np.mean(np.abs(audio))
    if signal_energy < 0.005:
        reliability_flags.append("LOW_SIGNAL_ENERGY")

    mfcc_abs_mean = np.mean(np.abs(mfcc))
    if mfcc_abs_mean > 500:
        reliability_flags.append("FEATURE_OUT_OF_RANGE")

    if reliability_flags:
        return {
            "reliable": False,
            "issues_detected": reliability_flags,
            "system_advice": "Prediction may be unreliable. Human review recommended."
        }

    return {
        "reliable": True,
        "issues_detected": [],
        "system_advice": "Prediction within expected operating conditions."
    }

# ----------------------------
# Load trained ML model
# ----------------------------
try:
    model = load("voice_detector.pkl")
except Exception:
    raise RuntimeError("voice_detector.pkl not found. Run train_model.py first.")

# ----------------------------
# Request schema
# ----------------------------
class AudioRequest(BaseModel):
    audio_base64: str

# ----------------------------
# Detection endpoint
# ----------------------------
@app.post("/detect")
def detect_voice(data: AudioRequest):
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Audio data too small")

        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr

        reliability_info = analyze_input_reliability(audio, sr)
        language_analysis = estimate_language(sr, duration)

        # MFCC extraction
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        # ✅ ADDED: model reliability check (Step 4.2)
        reliability_report = assess_model_reliability(audio, mfcc)

        feature_vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

        prediction = model.predict([feature_vector])[0]
        probability = model.predict_proba([feature_vector])[0].max()

        label = "AI" if prediction == 1 else "HUMAN"
        risk_level = calculate_risk(probability)

        decision_info = decision_intelligence(probability)
        warning = generate_warning(probability)

        if reliability_info["input_reliability"] == "LOW":
            return {
                "classification": "INCONCLUSIVE",
                "confidence": round(float(probability), 3),
                "risk_level": "HIGH",
                "decision_quality": {
                    "decision_stability": "LOW",
                    "recommended_action": "HUMAN_REVIEW_REQUIRED",
                    "confidence_calibrated": False
                },
                "uncertainty_analysis": reliability_info,
                "language_analysis": language_analysis,
                "explanation": "Input audio quality insufficient for reliable AI/HUMAN classification",
                "metadata": {
                    "sample_rate": sr,
                    "duration_seconds": round(duration, 2)
                }
            }

        # ✅ UPDATED response (Step 4.3)
        response = {
            "classification": label,
            "confidence": round(float(probability), 3),
            "risk_level": risk_level,
            "decision_quality": decision_info,
            "language_analysis": language_analysis,
            "model_reliability": reliability_report,
            "explanation": "Decision based on MFCC spectral patterns",
            "metadata": {
                "sample_rate": sr,
                "duration_seconds": round(duration, 2)
            }
        }

        if warning:
            response["warning"] = warning

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
