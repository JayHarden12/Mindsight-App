
import zipfile
import os
import pandas as pd
import glob

def extract_data(zip_path, extract_to):
    """
    Extracts the zip file if not already extracted.
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
    else:
        print(f"Data already extracted at {extract_to}")

def load_split_data(base_path, split='train'):
    """
    Loads text labels/metadata for a given split (train, dev, or test).
    Assumes AVEC 2017 structure roughly: Labels/{split}_split_Depression_AVEC2017.csv
    """
    # AVEC 2017 often has nomenclature like 'train_split_Depression_AVEC2017.csv'
    # Check possible paths
    csv_pattern = os.path.join(base_path, 'Labels', f'{split}_split_Depression_AVEC2017.csv')
    
    if os.path.exists(csv_pattern):
        df = pd.read_csv(csv_pattern)
        return df
    else:
        # Fallback or error
        print(f"Warning: Could not find label file at {csv_pattern}")
        return None

def get_audio_path(base_path, participant_id):
    """
    Finds the audio file for a corresponding participant_id.
    """
    # Audio files seem to be in 'audio/' folder based on previous exploration
    # Naming convention might be '{id}_AUDIO.wav'
    audio_path = os.path.join(base_path, 'audio', f'{participant_id}_AUDIO.wav')
    if os.path.exists(audio_path):
        return audio_path
    return None
