# Simple Meeting Note Taker

A simple meeting note-taking command line tool powered by OpenAI's Whisper speech-to-text models

## Features

🎤 **Multiple Transcription Modes:**
- **File-based transcription** for pre-recorded audio
- **Live transcription** with chunked processing
- **Smart VAD transcription** using voice activity detection
- **High-performance transcription** with faster-whisper (up to 4x faster)

📝 **Automatic Note Generation:**
- Auto-saves all transcriptions to organized files
- Timestamped filenames for easy organization
- Rich metadata including language detection and confidence scores
- Both detailed and continuous transcript formats

⚡ **Optimized Performance:**
- Multiple AI backends (OpenAI Whisper, Faster-Whisper)
- GPU and CPU support with automatic fallback
- Configurable precision (int8, float16, float32)
- Real-time performance monitoring

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify installation:**
   ```bash
   python -c "import whisper; print('Meeting Note Taker ready!')"
   ```

## Usage

### Method 1: Transcribe Audio Files

```bash
python whisper_transcribe.py <audio_file> [model_name]
```

Perfect for transcribing recorded meetings, interviews, or lectures:
```bash
# Transcribe a meeting recording
python whisper_transcribe.py recordings/team_meeting.wav base

# High-quality transcription for important content
python whisper_transcribe.py recordings/interview.mp3 large
```

### Method 2: Live Meeting Transcription

#### Basic Live Transcription (Chunked)
```bash
python whisper_live.py
```

Real-time transcription using fixed-time chunks:
- Captures audio in chunks (default 3 seconds)
- Transcribes speech as you speak
- Shows timestamped results
- Press Ctrl+C to stop

#### Advanced Live Transcription (VAD-based)
```bash
python whisper_live_vad.py
```

Smart transcription using Voice Activity Detection:
- **Only transcribes when speech is detected** (more efficient)
- **Natural speech segmentation** (no arbitrary time cuts)
- **Better accuracy** by processing complete speech segments
- **Noise robust** with configurable VAD aggressiveness
- Shows real-time speech detection status

**VAD Features:**
- Automatic speech/silence detection using WebRTC VAD
- 4 aggressiveness levels (0=quality, 3=very aggressive)
- Optimized for natural conversation patterns
- Reduces unnecessary processing during silence

#### High-Performance Live Meeting Transcription - **Recommended**
```bash
python whisper_live_faster.py
```

**Perfect for real-time meeting note-taking:**
- **Up to 4x faster** than standard Whisper
- **Smart voice detection** - only transcribes when people speak
- **Automatic meeting notes** saved with timestamps
- **Multi-language support** with confidence scoring
- **Performance monitoring** for optimal settings

**Meeting-Focused Features:**
- Real-time transcription as participants speak
- Automatic pause detection for natural conversation flow
- Rich meeting metadata (duration, participants, language)
- Continuous and segmented transcript formats
- **Auto-saves meeting notes** to `transcripts/transcript_YYYYMMDD_HHMMSS.txt`

## Supported Audio Formats

Whisper supports many audio formats including:
- WAV
- MP3
- MP4
- M4A
- FLAC
- AAC

## Model Information

| Model  | Parameters | VRAM | Relative Speed | Accuracy |
|--------|------------|------|----------------|----------|
| tiny   | 39 M       | ~1 GB| Fastest        | Lowest   |
| base   | 74 M       | ~1 GB| Fast           | Good     |
| small  | 244 M      | ~2 GB| Medium         | Better   |
| medium | 769 M      | ~5 GB| Slow           | High     |
| large  | 1550 M     | ~10 GB| Slowest       | Highest  |

## Adding Audio Files

Place your audio files in the `audio_samples/` directory to test transcription.

## Meeting Notes & Transcripts

All transcription tools automatically generate organized meeting notes and store in `transcripts/transcript_YYYYMMDD_HHMMSS.txt`

## Deactivating the Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

- **FFmpeg error**: Install FFmpeg system-wide if you encounter audio format issues
- **Model download**: The first time you use a model, it will be downloaded automatically
- **Memory issues**: Use smaller models (tiny/base) if you run into memory problems
- **PyAudio installation**: On macOS, install PortAudio first: `brew install portaudio`
- **Microphone permissions**: Grant microphone access when prompted by your system
- **No audio detected**: Check microphone levels and speak closer to the device
- **VAD not detecting speech**: Try different VAD modes (0-3) or check audio input levels
- **Choppy transcription**: Use VAD-based script for better speech segmentation