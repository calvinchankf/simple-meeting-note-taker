#!/usr/bin/env python3
"""
Live audio transcription using faster-whisper for optimized real-time performance.
This script uses faster-whisper (CTranslate2) for up to 4x faster transcription speed.
"""

import pyaudio
import wave
from faster_whisper import WhisperModel
import webrtcvad
import threading
import time
import tempfile
import os
import queue
import numpy as np
from datetime import datetime
import collections
from transcript_utils import TranscriptLogger

class FasterWhisperLiveTranscriber:
    def __init__(self, model_name="base", device="cpu", compute_type="int8", vad_mode=2):
        """
        Initialize the faster-whisper live transcriber.
        
        Args:
            model_name (str): Whisper model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device (str): Device to use ("cpu", "cuda")
            compute_type (str): Computation type ("int8", "float16", "float32")
            vad_mode (int): VAD aggressiveness (0-3, 3 is most aggressive)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_mode = vad_mode
        self.model = None
        self.is_recording = False
        
        # Audio settings optimized for WebRTC VAD
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 480  # 30ms at 16kHz
        self.audio_format = pyaudio.paInt16
        
        # VAD settings
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = 30
        self.padding_duration_ms = 300
        self.ring_buffer_size = self.padding_duration_ms // self.frame_duration_ms
        
        # Speech detection
        self.num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size)
        self.triggered = False
        self.voiced_frames = []
        
        # Performance tracking
        self.transcription_times = []
        self.total_segments = 0
        
        # Transcript logging using utility
        self.transcript_logger = TranscriptLogger(tool_name="faster-whisper", model_name=model_name)
        
        # Threading
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.recording_thread = None
        
        # PyAudio instance
        self.p = pyaudio.PyAudio()
        
    def load_model(self):
        """Load the faster-whisper model."""
        print(f"Loading faster-whisper model: {self.model_name}")
        print(f"Device: {self.device} | Compute type: {self.compute_type}")
        
        try:
            self.model = WhisperModel(
                self.model_name, 
                device=self.device, 
                compute_type=self.compute_type,
                download_root=None,
                local_files_only=False
            )
            print("✅ Faster-Whisper model loaded successfully!")
            
            # Print model info
            print(f"📊 Model info: {self.model_name} on {self.device} with {self.compute_type}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to CPU with int8...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            
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
            
            print(f"🎤 Faster-Whisper VAD Recording started!")
            print(f"⚡ Expected: ~4x faster transcription speed")
            print(f"📊 VAD Mode: {self.vad_mode} | Device: {self.device}")
            print(f"🎯 Waiting for speech... (speak naturally)")
            print("Press Ctrl+C to stop\n")
            
            while self.is_recording:
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                is_speech = self.is_speech(frame)
                
                if not self.triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    
                    if num_voiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = True
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🗣️  Speech detected, recording...")
                        self.voiced_frames.extend([f for f, s in self.ring_buffer])
                        self.ring_buffer.clear()
                        
                else:
                    self.voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                    
                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = False
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔇 Speech ended, processing with faster-whisper...")
                        
                        if len(self.voiced_frames) > 10:
                            temp_filename = self.save_audio_segment(self.voiced_frames)
                            self.audio_queue.put(temp_filename)
                        
                        self.voiced_frames = []
                        self.ring_buffer.clear()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Waiting for speech...")
                        
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            
            if self.voiced_frames:
                temp_filename = self.save_audio_segment(self.voiced_frames)
                self.audio_queue.put(temp_filename)
    
    def transcribe_worker(self):
        """Worker thread that processes speech segments using faster-whisper."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_file = self.audio_queue.get(timeout=1)
                
                # Measure transcription performance
                start_time = time.time()
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ⚡ Transcribing with faster-whisper...")
                
                # Use faster-whisper transcribe method
                segments, info = self.model.transcribe(
                    audio_file, 
                    beam_size=1,  # Faster beam search
                    vad_filter=True,  # Built-in VAD filtering
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=False  # Disable for speed
                )
                
                # Process segments
                transcription_text = ""
                for segment in segments:
                    transcription_text += segment.text
                
                end_time = time.time()
                transcription_duration = end_time - start_time
                self.transcription_times.append(transcription_duration)
                self.total_segments += 1
                
                text = transcription_text.strip()
                if text:
                    avg_time = sum(self.transcription_times) / len(self.transcription_times)
                    print(f"[{timestamp}] 📝 \"{text}\"")
                    print(f"[{timestamp}] 🌍 Language: {info.language} (confidence: {info.language_probability:.2f})")
                    print(f"[{timestamp}] ⚡ Speed: {transcription_duration:.2f}s | Avg: {avg_time:.2f}s | Total segments: {self.total_segments}")
                    
                    # Add to transcript log using utility
                    self.transcript_logger.add_transcript(
                        text=text,
                        timestamp=timestamp,
                        language=info.language,
                        confidence=info.language_probability,
                        duration=transcription_duration
                    )
                else:
                    print(f"[{timestamp}] ❓ No clear speech detected")
                
                print("-" * 70)
                
                # Clean up
                try:
                    os.unlink(audio_file)
                except OSError:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error during transcription: {e}")
    
    def start(self):
        """Start faster-whisper live transcription."""
        if not self.model:
            self.load_model()
        
        # Start transcript session
        self.transcript_logger.start_session(
            device=self.device,
            compute_type=self.compute_type,
            vad_mode=self.vad_mode
        )
        
        self.is_recording = True
        
        # Start threads
        self.recording_thread = threading.Thread(target=self.record_audio_vad)
        self.transcription_thread = threading.Thread(target=self.transcribe_worker)
        
        self.recording_thread.start()
        self.transcription_thread.start()
        
        try:
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop live transcription and show performance stats."""
        print(f"\n🛑 Stopping faster-whisper transcription...")
        self.is_recording = False
        
        # Process final speech
        if self.voiced_frames:
            print("🔄 Processing final speech segment...")
            temp_filename = self.save_audio_segment(self.voiced_frames)
            self.audio_queue.put(temp_filename)
        
        # Wait for threads
        if self.recording_thread:
            self.recording_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()
        
        # Show performance statistics
        if self.transcription_times:
            avg_time = sum(self.transcription_times) / len(self.transcription_times)
            min_time = min(self.transcription_times)
            max_time = max(self.transcription_times)
            print(f"\n📊 Performance Statistics:")
            print(f"   Total segments: {self.total_segments}")
            print(f"   Average transcription time: {avg_time:.2f}s")
            print(f"   Fastest transcription: {min_time:.2f}s")
            print(f"   Slowest transcription: {max_time:.2f}s")
        
        # Save transcript using utility
        self.transcript_logger.session_metadata['transcription_times'] = self.transcription_times
        self.transcript_logger.save_transcript()
        
        print("✅ Faster-Whisper transcription stopped.")
    
    def cleanup(self):
        """Clean up resources."""
        self.p.terminate()

def main():
    print("⚡ Faster-Whisper Live Transcription")
    print("=" * 50)
    
    # Configuration options
    available_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    devices = ["cpu", "cuda"]
    compute_types = {
        "cpu": ["int8", "float32"],
        "cuda": ["int8", "float16", "float32"]
    }
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
    
    print(f"\nAvailable devices: {', '.join(devices)}")
    device_choice = input("Choose device (default: cpu): ").strip() or "cpu"
    
    if device_choice not in devices:
        device_choice = "cpu"
    
    print(f"\nCompute types for {device_choice}: {', '.join(compute_types[device_choice])}")
    compute_choice = input(f"Choose compute type (default: int8): ").strip() or "int8"
    
    if compute_choice not in compute_types[device_choice]:
        compute_choice = "int8"
    
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
    transcriber = FasterWhisperLiveTranscriber(
        model_name=model_choice, 
        device=device_choice,
        compute_type=compute_choice,
        vad_mode=vad_mode
    )
    
    # Show devices
    transcriber.list_audio_devices()
    
    print(f"\n🚀 Starting Faster-Whisper live transcription:")
    print(f"   Model: {model_choice}")
    print(f"   Device: {device_choice}")
    print(f"   Compute: {compute_choice}")
    print(f"   VAD Mode: {vad_mode} ({vad_modes[str(vad_mode)]})")
    print(f"   Expected: ~4x faster than OpenAI Whisper")
    
    try:
        transcriber.start()
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.cleanup()

if __name__ == "__main__":
    main()