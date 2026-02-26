import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

# 1. THE BRAIN LOADER: This wakes up your 'voice_model.h5' file
@st.cache_resource
def load_my_model():
    # This must match your filename on GitHub exactly
    return tf.keras.models.load_model('voice_model.h5')

# Initialize the model
model = load_my_model()

# UI Setup for 'Affan Arts'
st.set_page_config(page_title="Affan Arts | AI Voice Shield", page_icon="🛡️")
st.title("🛡️ AI Voice Authenticator")
st.write("Upload a .wav file to verify its authenticity.")

# 2. THE UPLOADER
uploaded_file = st.file_uploader("Choose a voice recording...", type=["wav"])

if uploaded_file is not None:
    # Play the audio back to the user
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Neural Network is scanning vocal textures...'):
        # 3. FEATURE EXTRACTION: Must match your training logic
        audio, sr = librosa.load(uploaded_file, duration=2.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        feat = np.mean(mfccs.T, axis=0).reshape(1, 40)
        
        # 4. THE VERDICT: Using the model to predict
        prediction = model.predict(feat)
        score = prediction[0][1] # Probability of being 'Real'
        
        if score > 0.5:
            st.success(f"✅ VERDICT: REAL HUMAN VOICE ({score*100:.2f}% Confidence)")
        else:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({(1-score)*100:.2f}% Probability)")
