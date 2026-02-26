import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Affan Arts | AI Voice Shield", page_icon="🛡️")
st.title("🛡️ AI Voice Authenticator")
st.write("Upload a .wav file to verify its authenticity.")

# THE UPLOADER WIDGET
uploaded_file = st.file_uploader("Choose a voice recording...", type=["wav"])

if uploaded_file is not None:
    # Play the uploaded file
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Neural Network is scanning vocal textures...'):
        # Process the uploaded audio
        audio, sr = librosa.load(uploaded_file, duration=3.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        feat = np.mean(mfccs.T, axis=0).reshape(1, 40)
        
        # Verdict logic (connecting to your trained model)
        prediction = st.session_state.my_model.predict(feat)
        score = prediction[0][1] # Probability of being 'Real'
        
        if score > 0.5:
            st.success(f"✅ VERDICT: REAL HUMAN VOICE ({score*100:.2f}% Confidence)")
        else:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({(1-score)*100:.2f}% Probability)")
