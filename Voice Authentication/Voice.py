import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition

def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
    return processor, model, verifier

def speech_to_text(audio_path, processor, model):
    # Load audio file
    waveform, rate = librosa.load(audio_path, sr=16000)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    
    # Run through model
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def verify_speaker(audio1, audio2, verifier):
    score, _ = verifier.verify_files(audio1, audio2)
    return score.item()

if __name__ == "__main__":
    processor, model, verifier = load_model()
    
    # Speech-to-text demo
    audio_file = "sample.wav"  # Đổi thành đường dẫn file của bạn
    text = speech_to_text(audio_file, processor, model)
    print("Transcription:", text)
    
    # Speaker verification demo
    audio_file1 = "speaker1.wav"  # Đổi thành file của người 1
    audio_file2 = "speaker2.wav"  # Đổi thành file của người 2
    score = verify_speaker(audio_file1, audio_file2, verifier)
    print("Speaker similarity score:", score)
    
    if score > 0.5:
        print("Same speaker")
    else:
        print("Different speakers")
