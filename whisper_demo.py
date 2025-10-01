#!/usr/bin/env python3
"""
Interactive Whisper demo script.
This script provides a simple interactive way to test Whisper transcription.
"""

import whisper
import os

def list_audio_files(directory="audio_samples"):
    """List available audio files in the specified directory."""
    if not os.path.exists(directory):
        return []
    
    audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.aac'}
    audio_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory, file))
    
    return audio_files

def demo():
    """Interactive demo of Whisper transcription."""
    print("🎵 Whisper Transcription Demo")
    print("=" * 40)
    
    # List available models
    available_models = ["tiny", "base", "small", "medium", "large"]
    print(f"\nAvailable models: {', '.join(available_models)}")
    
    # Check for audio files
    audio_files = list_audio_files()
    if audio_files:
        print(f"\nFound audio files:")
        for i, file in enumerate(audio_files, 1):
            print(f"  {i}. {file}")
    else:
        print(f"\nNo audio files found in 'audio_samples/' directory.")
        print("Add some audio files to test transcription!")
        return
    
    # User input
    try:
        model_choice = input(f"\nChoose model (default: base): ").strip() or "base"
        if model_choice not in available_models:
            print(f"Invalid model. Using 'base'")
            model_choice = "base"
        
        print(f"\nSelect an audio file (1-{len(audio_files)}): ", end="")
        file_index = int(input()) - 1
        
        if 0 <= file_index < len(audio_files):
            selected_file = audio_files[file_index]
            
            print(f"\nLoading model '{model_choice}'...")
            model = whisper.load_model(model_choice)
            
            print(f"Transcribing '{selected_file}'...")
            result = model.transcribe(selected_file, fp16=False)
            
            print(f"\n{'='*50}")
            print("TRANSCRIPTION RESULT")
            print(f"{'='*50}")
            print(f"File: {selected_file}")
            print(f"Language: {result['language']}")
            print(f"Text: {result['text']}")
            
        else:
            print("Invalid file selection.")
            
    except (ValueError, IndexError):
        print("Invalid input.")
    except KeyboardInterrupt:
        print("\n\nDemo cancelled.")
    except Exception as e:
        print(f"\nError during transcription: {e}")

if __name__ == "__main__":
    demo()