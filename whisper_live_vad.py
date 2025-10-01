#!/usr/bin/env python3
"""
Live audio transcription using Whisper with Voice Activity Detection (VAD).
This script uses WebRTC VAD to detect speech and only transcribe when voice is detected.
"""

import pyaudio
import wave
import whisper
import webrtcvad
import threading
import time
import tempfile
import os
import queue
import numpy as np
from datetime import datetime
import collections

class VADLiveTranscriber:
    def __init__(self, model_name="base", vad_mode=3):
        """
        Initialize the VAD-based live transcriber.
        
        Args:
            model_name (str): Whisper model to use
            vad_mode (int): VAD aggressiveness (0-3, 3 is most aggressive)
        """
        self.model_name = model_name
        self.vad_mode = vad_mode
        self.model = None
        self.is_recording = False
        
        # Audio settings optimized for WebRTC VAD
        self.sample_rate = 16000  # WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz
        self.channels = 1
        self.chunk_size = 480  # 30ms at 16kHz (WebRTC VAD frame size)
        self.audio_format = pyaudio.paInt16
        
        # VAD settings
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = 30  # WebRTC VAD frame duration
        self.padding_duration_ms = 300  # Amount of audio to include before/after speech
        self.ring_buffer_size = self.padding_duration_ms // self.frame_duration_ms
        
        # Speech detection
        self.num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size)
        self.triggered = False
        self.voiced_frames = []
        
        # Threading
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.recording_thread = None
        
        # PyAudio instance
        self.p = pyaudio.PyAudio()
        
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
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']} (Input channels: {info['maxInputChannels']})")
    
    def is_speech(self, frame):
        """Check if frame contains speech using WebRTC VAD."""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False
    
    def save_audio_segment(self, frames):
        """Save audio frames to a temporary WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return temp_filename
    
    def record_audio_vad(self):
        """Record audio using VAD to detect speech segments."""
        try:
            stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"🎤 VAD Recording started!")
            print(f"📊 VAD Mode: {self.vad_mode} (0=least aggressive, 3=most aggressive)")
            print(f"🎯 Waiting for speech... (speak naturally)")
            print("Press Ctrl+C to stop\n")
            
            while self.is_recording:
                # Read frame
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Check if frame contains speech
                is_speech = self.is_speech(frame)
                
                if not self.triggered:
                    # Not in speech segment, add to ring buffer
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    
                    # If enough voiced frames, start collecting speech
                    if num_voiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = True
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🗣️  Speech detected, recording...")
                        # Add frames from ring buffer
                        self.voiced_frames.extend([f for f, s in self.ring_buffer])
                        self.ring_buffer.clear()
                        
                else:
                    # In speech segment, collect frames
                    self.voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                    
                    # If enough silence, end speech segment
                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = False
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔇 Speech ended, processing...")
                        
                        # Save and queue the speech segment
                        if len(self.voiced_frames) > 10:  # Minimum frames threshold
                            temp_filename = self.save_audio_segment(self.voiced_frames)
                            self.audio_queue.put(temp_filename)
                        
                        # Reset for next speech segment
                        self.voiced_frames = []
                        self.ring_buffer.clear()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Waiting for speech...")
                        
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            
            # Process any remaining voiced frames
            if self.voiced_frames:
                temp_filename = self.save_audio_segment(self.voiced_frames)
                self.audio_queue.put(temp_filename)
    
    def transcribe_worker(self):
        """Worker thread that processes speech segments for transcription."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio file from queue
                audio_file = self.audio_queue.get(timeout=1)
                
                # Transcribe the speech segment
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 🔄 Transcribing speech segment...")
                
                result = self.model.transcribe(audio_file, fp16=False)
                text = result['text'].strip()
                
                if text:
                    print(f"[{timestamp}] 📝 \"{text}\"")
                    print(f"[{timestamp}] 🌍 Language: {result['language']}")
                else:
                    print(f"[{timestamp}] ❓ No clear speech detected in segment")
                
                print("-" * 60)
                
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
        """Start VAD-based live transcription."""
        if not self.model:
            self.load_model()
        
        self.is_recording = True
        
        # Start threads
        self.recording_thread = threading.Thread(target=self.record_audio_vad)
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
        print(f"\n🛑 Stopping VAD transcription...")
        self.is_recording = False
        
        # Process any remaining speech
        if self.voiced_frames:
            print("🔄 Processing final speech segment...")
            temp_filename = self.save_audio_segment(self.voiced_frames)
            self.audio_queue.put(temp_filename)
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()
        
        print("✅ VAD transcription stopped.")
    
    def cleanup(self):
        """Clean up resources."""
        self.p.terminate()

def main():
    print("🎵 Whisper Live Transcription with VAD")
    print("=" * 50)
    
    # Configuration options
    available_models = ["tiny", "base", "small", "medium", "large"]
    vad_modes = {
        "0": "Quality (least aggressive)",
        "1": "Low bitrate",
        "2": "Aggressive", 
        "3": "Very aggressive (best for noisy environments)"
    }
    
    print(f"Available models: {', '.join(available_models)}")
    model_choice = input("Choose model (default: base): ").strip() or "base"
    
    if model_choice not in available_models:
        print(f"Invalid model. Using 'base'")
        model_choice = "base"
    
    print(f"\nVAD modes:")
    for mode, desc in vad_modes.items():
        print(f"  {mode}: {desc}")
    
    vad_mode = input("Choose VAD mode (default: 2): ").strip()
    try:
        vad_mode = int(vad_mode) if vad_mode else 2
        if vad_mode not in range(4):
            vad_mode = 2
    except ValueError:
        vad_mode = 2
    
    # Initialize transcriber
    transcriber = VADLiveTranscriber(model_name=model_choice, vad_mode=vad_mode)
    
    # Show available devices
    transcriber.list_audio_devices()
    
    print(f"\nStarting VAD live transcription with:")
    print(f"  Model: {model_choice}")
    print(f"  VAD Mode: {vad_mode} ({vad_modes[str(vad_mode)]})")
    print(f"  Sample Rate: {transcriber.sample_rate}Hz")
    print(f"\n💡 Tip: Speak naturally with pauses. VAD will detect speech segments automatically.")
    
    try:
        transcriber.start()
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.cleanup()

if __name__ == "__main__":
    main()