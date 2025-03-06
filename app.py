import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import uvicorn
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import torch
import torchaudio

app = FastAPI()

# Load model and scaler at startup
model = load_model("training_weights/audio_classifier_model.h5")
scaler = joblib.load("training_weights/feature_scaler.joblib")
class_names = ['Stove ON', 'Front Door', 'Stove OFF', 'Running Water']

def extract_audio_features(audio_path):
    """Extract relevant audio features from the audio file."""
    try:
        # Load the audio file
        y,sr=torchaudio.load(audio_path)
        print("loading done")
        y=torchaudio.transforms.Resample(orig_freq=sr,new_freq=41000)(y)
        print(y)
        # y, sr = librosa.load(audio_path, duration=5)

        # Extract features
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)

        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)

        # 6. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Combine all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [spectral_centroid_mean, spectral_centroid_std],
            [zcr_mean, zcr_std],
            [rms_mean, rms_std],
            [rolloff_mean, rolloff_std],
            chroma_mean, chroma_std
        ])

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        print("File name is: ",file.file)
        features = extract_audio_features(file.file)
        if features is None:
            return {"error": "Failed to extract features from audio"}
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return {"predicted_class": predicted_class, "confidence": float(confidence)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1067)