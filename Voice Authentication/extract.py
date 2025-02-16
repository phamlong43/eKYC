import librosa
import numpy as np

def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Trích xuất MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Lấy giá trị trung bình của mỗi đặc trưng MFCC
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean
