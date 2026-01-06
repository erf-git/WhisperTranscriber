"""
whisper_utils.py

~
"""

import whisper
import librosa
from pathlib import Path

# Processes an audio file and returns a text file with the transcription
def recorded_transcribe(file_path: Path) -> str:
    model = whisper.load_model("base")

    audio, sr = librosa.load(str(file_path), sr=16000) # Load audio file with 16000 Hz sampling rate with Librosa

    audio = whisper.pad_or_trim(audio) # Process audio through whisper
    mel = whisper.log_mel_spectrogram(audio).to(model.device) # Convert audio to mel spectrogram

    options = whisper.DecodingOptions(fp16=False) # Set decoding options
    result = whisper.decode(model, mel, options) # Decode the audio
    
    return result.text # Return the transcription