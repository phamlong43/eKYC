import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition
import cv2
import mediapipe as mp
import pickle
import pytesseract

# Load models
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
    return processor, model, verifier

# Speech to text
def speech_to_text(audio_path, processor, model):
    waveform, rate = librosa.load(audio_path, sr=16000)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Verify speaker
def verify_speaker(audio1, audio2, verifier):
    score, _ = verifier.verify_files(audio1, audio2)
    return score.item()

# Extract palm features
def extract_palm_features(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return features
    return None

# Verify palm
def verify_palm(image_path, reference_features):
    features = extract_palm_features(image_path)
    if features is None:
        return False, 0.0
    similarity = np.dot(features, reference_features) / (np.linalg.norm(features) * np.linalg.norm(reference_features))
    return similarity > 0.8, similarity

# Save palm reference
def save_palm_reference(image_path, save_path):
    features = extract_palm_features(image_path)
    if features is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

# Load palm reference
def load_palm_reference(save_path):
    with open(save_path, 'rb') as f:
        return pickle.load(f)

# Extract text from CCCD
def extract_cccd_info(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng')
    return text

if __name__ == "__main__":
    processor, model, verifier = load_model()
    
    # Speech Recognition
    audio_file = "sample.wav"
    text = speech_to_text(audio_file, processor, model)
    print("Transcription:", text)
    
    # Speaker Verification
    audio_file1 = "speaker1.wav"
    audio_file2 = "speaker2.wav"
    score = verify_speaker(audio_file1, audio_file2, verifier)
    print("Speaker similarity score:", score)
    if score > 0.5:
        print("Same speaker")
    else:
        print("Different speakers")
    
    # Palm Verification
    reference_palm = "palm_ref.pkl"
    palm_image = "palm.jpg"
    save_palm_reference(palm_image, reference_palm)
    reference_features = load_palm_reference(reference_palm)
    is_verified, similarity = verify_palm(palm_image, reference_features)
    if is_verified:
        print("Palm matches the reference. Similarity:", similarity)
    else:
        print("Palm does not match the reference. Similarity:", similarity)
    
    # CCCD Extraction
    cccd_image = "cccd.jpg"
    cccd_text = extract_cccd_info(cccd_image)
    print("Extracted CCCD Information:")
    print(cccd_text)
