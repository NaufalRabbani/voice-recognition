import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import tempfile

MODEL_PATH = "voice_cnn_model.h5"
model = load_model(MODEL_PATH)

SAMPLE_RATE = 22050
DURATION = 1.5
N_MFCC = 40
SAMPLES = int(SAMPLE_RATE * DURATION)

st.title("🎤 Voice Recognition: Buka / Tutup")

uploaded_file = st.file_uploader("Upload file suara", type=["wav", "mp3", "m4a"])


def preprocess_audio(uploaded):
    # Simpan ke file sementara (fix untuk mp3/m4a)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        file_path = tmp.name

    # Convert ke wav jika format bukan wav
    if uploaded.type != "audio/wav":
        audio = AudioSegment.from_file(file_path)
        wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(wav_temp.name, format="wav")
        file_path = wav_temp.name

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


if uploaded_file:
    st.audio(uploaded_file)

    features = preprocess_audio(uploaded_file)
    pred = model.predict(features)
    label = "BUKA" if pred[0][0] > 0.5 else "TUTUP"

    st.subheader("🔊 Prediksi:")
    st.success(label)
