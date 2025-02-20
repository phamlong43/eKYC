import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import pickle

# Load pre-trained speaker recognition model
speaker = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir")

def save_user_embedding(audio_file, user_id):
    # Trích xuất embedding cho giọng nói của người dùng từ file audio
    embedding = speaker.encode_file(audio_file)
    
    # Lưu embedding vào file với tên là user_id
    with open(f"user_embeddings/{user_id}.pkl", "wb") as f:
        pickle.dump(embedding, f)
    print(f"User embedding for {user_id} saved!")

# Ví dụ về việc lưu embedding của người dùng
save_user_embedding("user_audio.wav", "user123")