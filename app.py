
import streamlit as st
import sys
import os

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

from src.ui.analysis import show_analysis
from src.ui.prediction import show_prediction
from src.ui.training import show_training

st.set_page_config(
    page_title="MindSight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    h1 {
        color: #2c3e50;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.title("🧠 MindSight")
        st.info("Personalized Multimodal Depression Risk Prediction System")
        
        menu = ["Home", "Data Analysis", "Model Training", "Prediction"]
        choice = st.radio("Navigation", menu)
        
        st.markdown("---")
        st.text("Based on AVEC 2017 Dataset")
        
    if choice == "Home":
        st.title("Welcome to MindSight")
        st.image("https://images.unsplash.com/photo-1493836512294-502baa1986e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_column_width=True)
        st.markdown("""
        ### About the System
        MindSight utilizes advanced audio processing and machine learning to estimate depression risk based on vocal biomarkers.
        
        **Key Features:**
        - **Data Analysis**: Explore the distribution of the AVEC 2017 dataset.
        - **Model Training**: Train/Retrain the underlying machine learning model.
        - **Risk Prediction**: Upload audio samples to get real-time risk assessments.
        - **Explainability**: Understand which acoustic features contribute to the risk score.
        
        > **Disclaimer**: This tool is for research purposes only and is not a substitute for professional medical diagnosis.
        """)
        
    elif choice == "Data Analysis":
        show_analysis()
        
    elif choice == "Model Training":
        show_training()
        
    elif choice == "Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
