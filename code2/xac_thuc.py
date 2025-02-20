import numpy as np
import pickle
from speechbrain.pretrained import SpeakerRecognition

# Load pre-trained speaker recognition model
speaker = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir")

def verify_user(audio_file, user_id):
    # Trích xuất embedding từ giọng nói của người dùng
    embedding = speaker.encode_file(audio_file)
    
    # Đọc embedding đã lưu của người dùng từ file
    try:
        with open(f"user_embeddings/{user_id}.pkl", "rb") as f:
            stored_embedding = pickle.load(f)
        print(f"Stored embedding for {user_id} loaded!")
    except FileNotFoundError:
        print(f"User {user_id} not found!")
        return False
    
    # Tính cosine similarity giữa embeddings để xác thực
    similarity = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
    
    print(f"Cosine similarity: {similarity}")
    
    # Nếu cosine similarity lớn hơn ngưỡng, xác nhận người dùng
    threshold = 0.8  # Ngưỡng cho xác thực
    if similarity > threshold:
        print(f"User {user_id} authenticated!")
        return True
    else:
        print(f"User {user_id} authentication failed!")
        return False

# Ví dụ về việc xác thực người dùng
verify_user("user_audio_to_verify.wav", "user123")