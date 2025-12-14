
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from src.model.config import MAX_LEN

# Ensure src modules can be imported
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_processing.features import extract_audio_features, MAX_LEN

MODEL_PATH = 'models/depression_model.keras'

def get_model(input_shape):
    model = Sequential()
    
    # CNN Layer to capture local features
    model.add(Convolution1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Convolution1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    
    # BiLSTM Layer for temporal dependencies
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    
    # Dense Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax')) # Binary classification
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(base_path):
    print("Loading Labels...")
    # Basic loader - reuse simpler logic or import from loader if updated
    csv_path = os.path.join(base_path, 'Labels', 'train_split_Depression_AVEC2017.csv')
    if not os.path.exists(csv_path):
        return "Labels file not found."
        
    train_df = pd.read_csv(csv_path)
    audio_dir = os.path.join(base_path, 'audio')
    
    X = []
    y = []
    
    # Find available files
    print("Scanning audio files...")
    # Only process what we have
    available_files = [f for f in os.listdir(audio_dir) if f.endswith('_AUDIO.wav')]
    
    processed_count = 0
    
    for filename in available_files:
        try:
            p_id = int(filename.split('_')[0])
            row = train_df[train_df['Participant_ID'] == p_id]
            if not row.empty:
                # Get Label
                val = row.iloc[0]['PHQ8_Binary']
                
                audio_path = os.path.join(audio_dir, filename)
                feats, _ = extract_audio_features(audio_path, max_len=MAX_LEN)
                
                if feats is not None:
                    X.append(feats)
                    y.append(val)
                    processed_count += 1
                    if processed_count % 5 == 0:
                        print(f"Processed {processed_count} files...")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            
    if not X:
        return "No training data found."
        
    X = np.array(X)
    y = to_categorical(y, num_classes=2)
    
    print(f"Dataset Shape: {X.shape}")
    
    model = get_model((X.shape[1], X.shape[2]))
    model.summary()
    
    print("Starting Training...")
    history = model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2, verbose=1)
    
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)
    
    print(f"Model saved to {MODEL_PATH}")
    return "Training Complete (CNN-BiLSTM)"

if __name__ == "__main__":
    train_model(r"h:\new dataset")
