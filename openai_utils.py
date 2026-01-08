"""
openai_utils.py

~
"""

import whisper
import librosa
import re
import os
from pathlib import Path


def break_up_lines(result: str) -> str:
    """
    """

    # Split text into sentences and join with newlines
    text = result.text.strip() 

    # Split on sentence-ending punctuation followed by space or end of string
    # Using positive lookahead to keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Join sentences with newlines
    formatted_text = "\n".join(sentence.strip() for sentence in sentences if sentence.strip())

    return formatted_text # Return the transcription with each sentence on a new line

def transcribe(file_path: Path) -> str:
    """
    Processes an audio file with librosa and transcribes the audio with whisper
    Returns a str with the transcription.
    """

    model = whisper.load_model("base")

    # Load audio file with 16000 Hz sampling rate with Librosa 
    audio, sr = librosa.load(str(file_path), sr=16000) 

    audio = whisper.pad_or_trim(audio) # Process audio through whisper
    mel = whisper.log_mel_spectrogram(audio).to(model.device) # Convert audio to mel spectrogram

    options = whisper.DecodingOptions(fp16=False) # Set decoding options
    result = whisper.decode(model, mel, options) # Decode the audio
    
    return result 


def summarize(text: str, api_key: str) -> str:
    """
    Summarize the transcribed text using an OpenAI.
    Uses OpenAI API by default, but falls back to a simple transcription if API key is not set.
    """
    
    # Try to use OpenAI API if available
    try:
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Using simple transcription instead.")
            return text
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcribed audio content concisely and accurately."},
                {"role": "user", "content": f"Please provide a concise summary of the following transcription:\n\n{text}"}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Warning: Error using OpenAI API ({e}). Using simple extractive summary instead.")
        return text