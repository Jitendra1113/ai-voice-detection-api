import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# ⚠️ TEMPORARY DEMO DATA
# Replace later with real AI + human samples
X = np.random.randn(200, 40)  # 200 samples, 40 features
y = np.array([0]*100 + [1]*100)  # 0 = HUMAN, 1 = AI

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_test, y_test))

dump(model, "voice_detector.pkl")
print("Model saved as voice_detector.pkl")
