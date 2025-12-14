
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing.loader import load_split_data

def show_analysis():
    st.header("Dataset Analysis")
    
    # Load training data for analysis
    # Assuming the user has extracted the data to 'h:/new dataset' or current dir
    # We'll try to find where the data is. 
    # For now, hardcode or pass as config.
    base_path = "." 
    
    df = load_split_data(base_path, split='train')
    
    if df is not None:
        st.write(f"**Training Set Size:** {len(df)} samples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender Distribution")
            if 'gender' in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='gender', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Gender column not found.")
                
        with col2:
            st.subheader("Depression Label Distribution")
            # Check for binary label
            if 'PHQ8_Binary' in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='PHQ8_Binary', ax=ax)
                st.pyplot(fig)
            else:
                st.info("PHQ8_Binary column not found.")
        
        st.subheader("PHQ-8 Score Distribution")
        if 'PHQ8_Score' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='PHQ8_Score', bins=10, ax=ax, kde=True)
            st.pyplot(fig)
            
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100))
            
    else:
        st.error("Could not load dataset. Please ensure 'archive.zip' is extracted.")
