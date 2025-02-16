import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.mixture import GaussianMixture
import pickle
import time
import random

# Danh sách các câu gợi ý
prompts = [
    "Xin chào, đây là hệ thống xác thực của tôi",
    "Bảo mật là ưu tiên hàng đầu của chúng tôi",
    "Giọng nói của tôi là chìa khóa bảo mật",
    "Công nghệ nhận dạng giọng nói rất thú vị"
]

def record_audio(duration=3, fs=16000):
    """Ghi âm từ microphone"""
    print("Bắt đầu ghi âm...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Ghi âm hoàn tất!")
    return recording.flatten()

def save_audio(audio, filename, fs=16000):
    """Lưu âm thanh vào file"""
    sf.write(filename, audio, fs)
    print(f"Đã lưu âm thanh vào {filename}")

def extract_features(audio, sr=16000):
    """Trích xuất đặc trưng MFCC từ tín hiệu âm thanh"""
    # Trích xuất MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    # Chuẩn hóa
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    # Chuyển thành ma trận 2D
    mfccs = mfccs.T
    return mfccs

def enroll_user(username, num_samples=5):
    """Đăng ký người dùng mới"""
    if not os.path.exists("users"):
        os.makedirs("users")
    
    features_list = []
    
    for i in range(num_samples):
        prompt = random.choice(prompts)
        print(f"Mẫu {i+1}/{num_samples}. Vui lòng đọc: '{prompt}'")
        time.sleep(1)
        audio = record_audio(duration=4)
        features = extract_features(audio)
        features_list.append(features)
        
        # Lưu mẫu âm thanh
        if not os.path.exists(f"users/{username}"):
            os.makedirs(f"users/{username}")
        save_audio(audio, f"users/{username}/sample_{i}.wav")
    
    # Gộp tất cả các đặc trưng
    all_features = np.vstack(features_list)
    
    # Huấn luyện mô hình GMM
    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200)
    gmm.fit(all_features)
    
    # Lưu mô hình
    with open(f"users/{username}/model.pkl", "wb") as f:
        pickle.dump(gmm, f)
    
    print(f"Đã đăng ký thành công người dùng {username}!")

def authenticate_user(username):
    """Xác thực người dùng"""
    if not os.path.exists(f"users/{username}"):
        print(f"Không tìm thấy người dùng {username}!")
        return False
    
    # Tải mô hình
    with open(f"users/{username}/model.pkl", "rb") as f:
        gmm = pickle.load(f)
    
    # Chọn ngẫu nhiên một câu gợi ý
    prompt = random.choice(prompts)
    print(f"Vui lòng đọc: '{prompt}'")
    time.sleep(1)
    
    # Ghi âm
    audio = record_audio(duration=4)
    
    # Trích xuất đặc trưng
    features = extract_features(audio)
    
    # Tính điểm log-likelihood
    score = gmm.score(features)
    
    # Ngưỡng xác thực (cần điều chỉnh dựa trên thử nghiệm)
    threshold = -35
    
    if score >= threshold:
        print(f"Xác thực thành công! Điểm số: {score:.2f}")
        return True
    else:
        print(f"Xác thực thất bại! Điểm số: {score:.2f}")
        return False

def main():
    while True:
        print("\n===== HỆ THỐNG XÁC THỰC GIỌNG NÓI =====")
        print("1. Đăng ký người dùng mới")
        print("2. Xác thực người dùng")
        print("3. Thoát")
        choice = input("Nhập lựa chọn của bạn: ")
        
        if choice == "1":
            username = input("Nhập tên người dùng: ")
            enroll_user(username)
        elif choice == "2":
            username = input("Nhập tên người dùng: ")
            authenticate_user(username)
        elif choice == "3":
            break
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()