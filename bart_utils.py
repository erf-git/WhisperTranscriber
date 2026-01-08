"""
bart_utils.py

Utilities for text summarization using BART model from Hugging Face.
"""

import re
from transformers import pipeline


def summarize(text: str) -> str:
    """
    Summarize the transcribed text using a free local model (BART from Hugging Face).
    This runs entirely on your machine - no API calls or costs.
    """
    
    try:
        # Load the summarization model (first time will download ~1.6GB, then cached)
        # Using facebook/bart-large-cnn which is specifically designed for summarization
        print("Loading summarization model... (this may take a moment on first run)")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # BART has a max input length of 1024 tokens, so we may need to chunk long texts
        max_length = 1024
        min_length = 50
        
        # If text is too long, split into chunks and summarize each
        if len(text) > max_length:
            print("Text is long, summarizing in chunks...")
            # Split by sentences to avoid cutting mid-sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Summarize each chunk
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            # Combine summaries
            combined_text = " ".join(summaries)
            
            # If combined summary is still long, summarize it one more time
            if len(combined_text) > max_length:
                print("Combining summaries...")
                final_summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
                return final_summary[0]['summary_text']
            else:
                return combined_text
        else:
            # Text is short enough, summarize directly
            print("Generating summary...")
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        print("Returning original text as fallback.")
        return text
