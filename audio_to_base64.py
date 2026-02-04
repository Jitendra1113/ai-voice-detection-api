import base64

audio_path = "narke.wav"   # MUST match file name exactly

with open(audio_path, "rb") as audio_file:
    encoded = base64.b64encode(audio_file.read()).decode("utf-8")

print(encoded)
