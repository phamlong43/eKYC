import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Hàm trích xuất MFCC từ âm thanh
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Đọc tệp âm thanh
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Trích xuất MFCC
    return mfcc.T  # Trả về MFCC theo dạng ma trận với mỗi dòng là một vector MFCC

# Hàm tính độ tương đồng giữa 2 mẫu MFCC
def compare_mfcc(mfcc1, mfcc2):
    # Sử dụng Dynamic Time Warping (DTW) để so sánh 2 chuỗi MFCC
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

# Đường dẫn đến tệp âm thanh
sample_voice_file = 'sample_voice.wav'  # Giọng nói mẫu đã lưu
recorded_voice_file = 'recorded_voice.wav'  # Giọng nói đã thu từ người dùng

# Trích xuất MFCC từ giọng nói mẫu và giọng nói thu được
mfcc_sample = extract_mfcc(sample_voice_file)
mfcc_recorded = extract_mfcc(recorded_voice_file)

# So sánh độ tương đồng giữa 2 đặc trưng MFCC
distance = compare_mfcc(mfcc_sample, mfcc_recorded)

# Nếu độ tương đồng thấp, xác thực thành công
if distance < 500:  # Ngưỡng có thể điều chỉnh tùy theo dữ liệu
    print("Xác thực thành công!")
else:
    print("Xác thực thất bại.")