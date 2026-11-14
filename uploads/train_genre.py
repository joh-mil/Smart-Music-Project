import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Define the Feature Extraction Pipeline
def extract_audio_features(file_path):
    try:
        # Load 30 seconds of audio
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract core acoustic features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Compile Mean and Variance configurations for each feature
        feature_row = [
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rmse), np.var(rmse),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr)
        ]
        
        # Append mean and variance for all 20 MFCC bands
        for m in mfcc:
            feature_row.append(np.mean(m))
            feature_row.append(np.var(m))
            
        return feature_row
    except Exception as e:
        print(f"Skipping corrupted file {file_path}: {e}")
        return None

# 2. Process Dataset Directory
DATASET_PATH = "dataset/genres_original" # Change to your GTZAN folder path
X = []
y_labels = []

print("⏳ Extracting audio features from GTZAN dataset... This takes a few minutes.")
for genre in os.listdir(DATASET_PATH):
    genre_folder = os.path.join(DATASET_PATH, genre)
    if os.path.isdir(genre_folder):
        print(f"Processing genre: {genre}")
        for filename in os.listdir(genre_folder):
            file_path = os.path.join(genre_folder, filename)
            if file_path.endswith(".wav"):
                features = extract_audio_features(file_path)
                if features is not None:
                    X.append(features)
                    y_labels.append(genre)

# 3. Preprocess and Encode Data
X = np.array(X)
y_labels = np.array(y_labels)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Normalize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training and Testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("🚀 Training Random Forest Classifier model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate results
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎉 Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

# 5. Export Model Files for Flask Integration
joblib.dump(model, 'genre_model.pkl')
joblib.dump(scaler, 'genre_scaler.pkl')
joblib.dump(label_encoder, 'genre_encoder.pkl')
print("💾 Model files saved successfully: genre_model.pkl, genre_scaler.pkl, genre_encoder.pkl")