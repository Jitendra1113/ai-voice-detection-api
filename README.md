# 🎙️ Trust-Aware AI Voice Detection API

## 🚀 What Is This Project?

This project is a **production-grade AI-generated voice detection system** that determines whether a given voice sample is **AI-generated or human-generated** — while being **honest about uncertainty, risk, and reliability**.

Unlike most demo-style classifiers that only return a label, this system behaves like a **real-world decision engine**:

> It not only predicts — it explains, evaluates risk, detects uncertainty, and recommends what to do next.

This makes it suitable for **fraud detection, voice authentication, media forensics, and AI safety applications**.

---

## 🧩 Why This Project Exists

AI-generated voices are becoming:
- Highly realistic
- Multilingual
- Easy to misuse

Most solutions answer only one question:
> “Is this AI or Human?”

But in real deployments, that’s not enough.

This system answers **five critical questions** instead:

1. What does the model predict?
2. How confident is that prediction?
3. How risky is this decision?
4. Can this result be trusted?
5. What action should a system or human take next?

That difference is what makes this project stand out.

---

## 🌍 Multi-Language by Design

The system supports voice samples in:
- **English**
- **Hindi**
- **Tamil**
- **Telugu**
- **Malayalam**

Instead of using transcription or language-specific models, the system relies on **language-agnostic acoustic features**, making it robust across languages while remaining honest about its confidence.

---

## 🧠 How the System Works (End-to-End)


---

## 🎯 Core Design Principles

### ✅ Not Just Accurate — Trustworthy
The system avoids overconfident predictions and clearly communicates uncertainty.

### ✅ Language-Aware Without Overclaiming
Language characteristics are estimated acoustically, not assumed or hard-classified.

### ✅ Safety-First Decision Making
When input quality is poor or confidence is low, the system **refuses to overclaim** and recommends human review.

### ✅ Production-Ready API
Clean JSON responses, deterministic logic, and explainable outputs.

---

## 🧪 Why MFCCs?

Mel-Frequency Cepstral Coefficients (MFCCs) are widely used in:
- Speech forensics
- Spoof detection
- Speaker recognition

They were chosen because they:
- Capture spectral artifacts common in AI-generated speech
- Are language-independent
- Perform well in low-data and multilingual settings
- Are explainable and auditable

This makes them ideal for a **responsible, hackathon-safe solution**.

---

## 📡 API Overview

### Endpoint

### Input
```json
{
  "audio_base64": "<Base64-encoded audio>"
}

```
### Output 
```
{
  "classification": "AI",
  "confidence": 0.57,
  "risk_level": "MEDIUM",

  "decision_quality": {
    "decision_stability": "MEDIUM",
    "recommended_action": "SECONDARY_VERIFICATION",
    "confidence_calibrated": true
  },

  "language_analysis": {
    "language_estimate": "INDIC (Tamil/Hindi/Malayalam/Telugu)",
    "language_confidence": 0.72,
    "language_basis": "Prosodic rhythm & sampling characteristics"
  },

  "model_reliability": {
    "reliable": true,
    "issues_detected": [],
    "system_advice": "Prediction within expected operating conditions."
  },

  "warning": {
    "warning_level": "MEDIUM",
    "message": "Model confidence is moderate. Secondary verification is recommended."
  },

  "explanation": "Decision based on MFCC spectral patterns",

  "metadata": {
    "sample_rate": 44100,
    "duration_seconds": 5.94
  }
}
```

## 🛡️ Responsible AI & Safety Features

This system is explicitly designed to prioritize safety, transparency, and responsible decision-making:

- Avoids blind automation by incorporating validation checks  
- Surfaces uncertainty clearly instead of forcing confident outputs  
- Detects unreliable, low-quality, or ambiguous inputs  
- Recommends human review when confidence is insufficient  
- Abstains from making decisions when reliability cannot be ensured  

These safeguards make the system safer and more trustworthy than models that always return a confident answer, even when uncertainty is high.

---

## ⚠️ Limitations (Honest Disclosure)

To maintain transparency, the following limitations are clearly acknowledged:

- Performance depends on the diversity and quality of training data  
- Language analysis is heuristic-based, not transcription-driven  
- Very short, noisy, or corrupted audio inputs may result in **INCONCLUSIVE** outputs  
- Results should not be treated as absolute ground truth  

All limitations and confidence indicators are explicitly communicated in the API response to support informed human judgment.


## 🛠️ Installation & Setup Guide

This guide explains how to set up and run the project **from scratch**.  
It is written for team members or reviewers with **no prior ML experience**.

---

### ✅ Prerequisites

Make sure the following are installed:

- **Python 3.9 or later**
  ```bash
  python --version
pip (Python package manager)

pip --version
Git (optional but recommended)

git --version
⚠️ If Python is not installed, download it from:
https://www.python.org/downloads/
(Ensure “Add Python to PATH” is checked during installation.)

📥 Step 1: Clone the Repository
git clone https://github.com/<your-username>/ai-voice-detection-api.git
cd ai-voice-detection-api
Or download the ZIP from GitHub and extract it manually.

📦 Step 2: Create a Virtual Environment (Recommended)
python -m venv venv
Activate it:

Windows

venv\Scripts\activate
macOS / Linux

source venv/bin/activate
You should see (venv) in your terminal.

📚 Step 3: Install Dependencies
pip install -r requirements.txt
This installs:

FastAPI

Uvicorn

librosa

soundfile

numpy

scikit-learn

joblib

🧠 Step 4: Model File Check
Ensure the trained model file exists:

voice_detector.pkl
If missing, generate it:

python train_model.py
⚠️ The API will not start without this file.

🚀 Step 5: Run the API Server
uvicorn main:app --reload
Expected output:

Uvicorn running on http://127.0.0.1:8000
🔍 Step 6: Test the API (Swagger UI)
Open your browser:

http://127.0.0.1:8000/docs
You can:

Upload Base64 audio

Call the /detect endpoint

View structured JSON responses

No external tools required.

🎧 Step 7: Preparing Audio Input (Base64)
Audio must be Base64-encoded.

Example (Python)
import base64

with open("sample.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()
Request body:

{
  "audio_base64": "<PASTE_BASE64_STRING_HERE>"
}
🧪 Step 8: API Response Structure
The API returns:

AI vs HUMAN classification

Confidence score

Risk level

Decision recommendation

Language analysis

Reliability assessment

Warnings (if applicable)

All uncertainty and limitations are clearly communicated.

🧯 Common Issues & Fixes
Issue	Solution
ModuleNotFoundError	Run pip install -r requirements.txt
voice_detector.pkl missing	Run python train_model.py
Audio too small	Use audio longer than 2 seconds
No response	Ensure Uvicorn is running