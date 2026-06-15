import streamlit as st
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile, os
from predict import predict_file

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)
st.title("Deepfake Audio Detector")
st.caption("Upload a .wav file to check if it is Genuine (Human) or Deepfake (AI-Generated)")

audio_file = st.file_uploader("Choose a .wav audio file", type=["wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(audio_file, format="audio/wav")

    with st.spinner("Analyzing audio..."):
        label, confidence = predict_file(tmp_path)

    col1, col2 = st.columns(2)
    if "Genuine" in label:
        col1.success(f"Result: {label}")
    else:
        col1.error(f"Result: {label}")
    col2.metric("Confidence", f"{confidence*100:.1f}%")

    st.subheader("Mel spectrogram")
    y_audio, sr = librosa.load(tmp_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time',
                                    y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title("Mel spectrogram of uploaded audio")
    st.pyplot(fig)
    plt.close(fig)

    os.unlink(tmp_path)

    with st.expander("How does this work?"):
        st.write("""
        1. Your audio is resampled to 16 kHz and trimmed/padded to 3 seconds.
        2. A mel spectrogram is extracted — converting the audio into a 2D frequency-time image.
        3. A trained CNN classifies the spectrogram image as Genuine or Deepfake.
        4. The confidence score is the softmax probability of the predicted class.
        """)