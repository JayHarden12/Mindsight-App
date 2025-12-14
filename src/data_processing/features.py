
import librosa
import numpy as np

from src.model.config import MAX_LEN

# MAX_LEN imported from config


def extract_audio_features(audio_path, sr=16000, max_len=MAX_LEN):
    """
    Extracts sequential MFCC features for CNN-BiLSTM.
    Returns shape: (max_len, n_mfcc)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # MFCC
        # n_mfcc=40 is common for DL speech tasks
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # mfccs shape is (n_mfcc, time) -> transpose to (time, n_mfcc)
        mfccs = mfccs.T 
        
        # Pad or truncate to max_len
        if mfccs.shape[0] < max_len:
            # Pad with zeros
            pad_width = max_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate
            mfccs = mfccs[:max_len, :]
            
        feature_names = [f"MFCC_{i}" for i in range(40)] 
                         
        return mfccs, feature_names
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None
