"""
whisper_utils.py

~
"""

import whisper
import librosa
import re
from pathlib import Path

def break_up_lines(result: str) -> str:
    # Split text into sentences and join with newlines
    text = result.text.strip() 

    # Split on sentence-ending punctuation followed by space or end of string
    # Using positive lookahead to keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Join sentences with newlines
    formatted_text = "\n".join(sentence.strip() for sentence in sentences if sentence.strip())

    return formatted_text # Return the transcription with each sentence on a new line


# Processes an audio file and returns a text file with the transcription
def recorded_transcribe(file_path: Path) -> str:
    model = whisper.load_model("base")

    # Load audio file with 16000 Hz sampling rate with Librosa 
    audio, sr = librosa.load(str(file_path), sr=16000) 

    audio = whisper.pad_or_trim(audio) # Process audio through whisper
    mel = whisper.log_mel_spectrogram(audio).to(model.device) # Convert audio to mel spectrogram

    options = whisper.DecodingOptions(fp16=False) # Set decoding options
    result = whisper.decode(model, mel, options) # Decode the audio
    
    return result 