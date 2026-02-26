import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('voice_model.h5')

model = load_my_model()

st.set_page_config(page_title="Affan Arts | AI Voice Shield", page_icon="🛡️")
st.title("🛡️ AI Voice Authenticator")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner('Scanning for digital artifacts...'):
        # MUST MATCH TRAINING: MFCC + Spectral Contrast
        y, sr = librosa.load(uploaded_file, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Combine features into the 47-dimension vector
        feat = np.hstack((np.mean(mfcc, axis=1), np.mean(contrast, axis=1))).reshape(1, 47)
        
        prediction = model.predict(feat)
        score = prediction[0][1] # Real Probability
        
        if score > 0.5:
            st.success(f"✅ VERDICT: REAL HUMAN VOICE ({score*100:.2f}%)")
        else:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({(1-score)*100:.2f}%)")
