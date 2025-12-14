
import streamlit as st
import threading
from src.model.trainer import train_model

def show_training():
    st.header("Model Training")
    st.write("Train the depression risk prediction model using the available dataset.")
    
    st.info("This process extracts features from audio files and trains a Random Forest Classifier. It may take a few minutes depending on the dataset size.")
    
    if st.button("Start Training"):
        with st.status("Training in progress...", expanded=True) as status:
            st.write("Initializing...")
            
            # wrapper to capture output or just run it
            # Since train_model prints to stdout, we won't see it in the UI easily without more complex logging redirection.
            # For now, we trust the function returns a status string.
            
            try:
                # We assume the data is in current directory or 'h:/new dataset'
                # Let's try to detect or pass the root
                result = train_model(".") 
                
                if "Complete" in result or "saved" in result:
                    status.update(label="Training Complete!", state="complete", expanded=False)
                    st.success(result)
                else:
                    status.update(label="Training Finished", state="complete", expanded=False)
                    st.info(result)
                    
            except Exception as e:
                status.update(label="Training Failed", state="error", expanded=False)
                st.error(f"An error occurred: {e}")
