#!/usr/bin/env python3
"""
Basic Whisper transcription example.
This script demonstrates how to use OpenAI's Whisper for audio transcription.
"""

import whisper
import sys
import os
from transcript_utils import save_simple_transcript

def transcribe_audio(audio_file_path, model_name="base"):
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_name (str): Whisper model to use (tiny, base, small, medium, large)
    
    Returns:
        dict: Transcription result containing text and other metadata
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing: {audio_file_path}")
    result = model.transcribe(audio_file_path, fp16=False)
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python whisper_transcribe.py <audio_file> [model_name]")
        print("\nAvailable models: tiny, base, small, medium, large")
        print("Example: python whisper_transcribe.py audio_samples/sample.wav base")
        return
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    try:
        result = transcribe_audio(audio_file, model_name)
        
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT")
        print("="*50)
        print(f"Text: {result['text']}")
        print(f"\nLanguage: {result['language']}")
        
        # Print segments with timestamps
        print("\nSegments with timestamps:")
        print("-" * 30)
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            print(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
            
        # Save transcript to file
        transcript_text = result['text'].strip()
        if transcript_text:
            saved_file = save_simple_transcript(
                text=transcript_text,
                tool_name="command-line-whisper", 
                model_name=model_name
            )
            if saved_file:
                print(f"\n📝 Transcript saved to: {saved_file}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()