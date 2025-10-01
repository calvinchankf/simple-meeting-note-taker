#!/usr/bin/env python3
"""
Live audio transcription using Whisper with chunked audio capture.
This script captures audio from the microphone in real-time and transcribes it.
"""

import pyaudio
import wave
import whisper
import threading
import time
import tempfile
import os
import queue
from datetime import datetime
from transcript_utils import TranscriptLogger

class LiveTranscriber:
    def __init__(self, model_name="base", chunk_duration=3):
        """
        Initialize the live transcriber.
        
        Args:
            model_name (str): Whisper model to use
            chunk_duration (int): Duration of each audio chunk in seconds
        """
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.model = None
        self.is_recording = False
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        # Threading
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.recording_thread = None
        
        # PyAudio instance
        self.p = pyaudio.PyAudio()
        
        # Transcript logging
        self.transcript_logger = TranscriptLogger(tool_name="basic-whisper", model_name=model_name)
        
    def load_model(self):
        """Load the Whisper model."""
        print(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        print("Model loaded successfully!")
        
    def list_audio_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        print("-" * 40)
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Input device
                print(f"Device {i}: {info['name']} (Input channels: {info['maxInputChannels']})")
    
    def record_audio_chunk(self):
        """Record a chunk of audio and put it in the queue."""
        try:
            stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"🎤 Recording started! Speak into your microphone...")
            print(f"📊 Chunk duration: {self.chunk_duration}s | Model: {self.model_name}")
            print("Press Ctrl+C to stop\n")
            
            while self.is_recording:
                frames = []
                
                # Record for chunk_duration seconds
                for _ in range(int(self.sample_rate / self.chunk_size * self.chunk_duration)):
                    if not self.is_recording:
                        break
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                
                if frames and self.is_recording:
                    # Save chunk to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        
                        wf = wave.open(temp_filename, 'wb')
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        # Add to transcription queue
                        self.audio_queue.put(temp_filename)
                        
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
    
    def transcribe_worker(self):
        """Worker thread that processes audio chunks for transcription."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio file from queue (with timeout to allow checking is_recording)
                audio_file = self.audio_queue.get(timeout=1)
                
                # Transcribe the audio
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 🔄 Transcribing...")
                
                result = self.model.transcribe(audio_file, fp16=False)
                text = result['text'].strip()
                
                if text:
                    print(f"[{timestamp}] 📝 {text}")
                    
                    # Add to transcript log
                    self.transcript_logger.add_transcript(
                        text=text,
                        timestamp=timestamp,
                        language=result['language']
                    )
                else:
                    print(f"[{timestamp}] 🔇 (No speech detected)")
                
                # Clean up temporary file
                try:
                    os.unlink(audio_file)
                except OSError:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error during transcription: {e}")
    
    def start(self):
        """Start live transcription."""
        if not self.model:
            self.load_model()
        
        # Start transcript session
        self.transcript_logger.start_session(chunk_duration=self.chunk_duration)
        
        self.is_recording = True
        
        # Start threads
        self.recording_thread = threading.Thread(target=self.record_audio_chunk)
        self.transcription_thread = threading.Thread(target=self.transcribe_worker)
        
        self.recording_thread.start()
        self.transcription_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop live transcription."""
        print(f"\n🛑 Stopping transcription...")
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()
        
        # Save transcript
        self.transcript_logger.save_transcript()
        
        print("✅ Transcription stopped.")
    
    def cleanup(self):
        """Clean up resources."""
        self.p.terminate()

def main():
    print("🎵 Whisper Live Transcription")
    print("=" * 40)
    
    # Configuration options
    available_models = ["tiny", "base", "small", "medium", "large"]
    
    print(f"Available models: {', '.join(available_models)}")
    model_choice = input("Choose model (default: base): ").strip() or "base"
    
    if model_choice not in available_models:
        print(f"Invalid model. Using 'base'")
        model_choice = "base"
    
    chunk_duration = input("Chunk duration in seconds (default: 3): ").strip()
    try:
        chunk_duration = int(chunk_duration) if chunk_duration else 3
    except ValueError:
        chunk_duration = 3
    
    # Initialize transcriber
    transcriber = LiveTranscriber(model_name=model_choice, chunk_duration=chunk_duration)
    
    # Show available devices
    transcriber.list_audio_devices()
    
    print(f"\nStarting live transcription with:")
    print(f"  Model: {model_choice}")
    print(f"  Chunk duration: {chunk_duration}s")
    
    try:
        transcriber.start()
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.cleanup()

if __name__ == "__main__":
    main()