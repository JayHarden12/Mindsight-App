
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing.features import extract_audio_features, MAX_LEN
from src.model.trainer import MODEL_PATH
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_risk(audio_file):
    if not os.path.exists(MODEL_PATH):
        st.error("Model not trained yet. Please run the training script.")
        return

    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
        
    st.audio(temp_path)
    
    with st.spinner("Extracting features..."):
        # New extraction returns (Time, Features)
        features, feature_names = extract_audio_features(temp_path, max_len=MAX_LEN)
    
    # Cleanup
    try:
        os.remove(temp_path)
    except:
        pass
        
    if features is None:
        st.error("Could not extract features from audio.")
        return

    # Expand dims for batch: (1, Time, Features)
    features = np.expand_dims(features, axis=0)
    
    # Load Model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Predict
    risk_prob = model.predict(features)[0] # [prob_0, prob_1]
    prediction = np.argmax(risk_prob)
    
    st.divider()
    
    # Display Result
    col1, col2 = st.columns([1, 2])
    
    confidence = risk_prob[prediction]
    
    with col1:
        st.metric("Depression Risk", 
                  "HIGH" if prediction == 1 else "LOW",
                  delta=f"{confidence*100:.1f}% confidence",
                  delta_color="inverse")
    
    with col2:
        st.write("### Interpretation")
        if prediction == 1:
            st.warning("The model detects vocal patterns associated with depression risk (CNN-BiLSTM Analysis). Professional consultation is recommended.")
        else:
            st.success("The model detects vocal patterns consistent with low depression risk.")
            
    # Explainability (Saliency Map for DL)
    # Simple heatmap of Input MFCCs as a proxy for "what the model saw"
    st.subheader("Acoustic Feature Heatmap (MFCC)")
    st.write("Visualizing the Mel-frequency cepstral coefficients used by the neural network.")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    # features shape is (1, time, freq) -> plot (freq, time)
    cax = ax.imshow(features[0].T, aspect='auto', origin='lower', cmap='viridis')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_xlabel('Time Frame')
    fig.colorbar(cax)
    st.pyplot(fig)
    
def show_prediction():
    st.header("Risk Prediction (Deep Learning)")
    st.write("Upload an audio recording (WAV format) to analyze depression markers using the CNN-BiLSTM model.")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    
    if uploaded_file is not None:
        if st.button("Analyze Risk"):
            predict_risk(uploaded_file)
