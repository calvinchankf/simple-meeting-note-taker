# Whisper Starter Project

A simple starter project to experiment with OpenAI's Whisper speech-to-text model.

## Setup

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify installation:**
   ```bash
   python -c "import whisper; print('Whisper installed successfully!')"
   ```

## Usage

### Method 1: Command Line Transcription

```bash
python whisper_transcribe.py <audio_file> [model_name]
```

**Examples:**
```bash
# Using default 'base' model
python whisper_transcribe.py audio_samples/sample.wav

# Using different models
python whisper_transcribe.py audio_samples/sample.wav tiny    # Fastest, least accurate
python whisper_transcribe.py audio_samples/sample.wav base    # Good balance
python whisper_transcribe.py audio_samples/sample.wav small   # Better accuracy
python whisper_transcribe.py audio_samples/sample.wav medium  # Even better accuracy
python whisper_transcribe.py audio_samples/sample.wav large   # Best accuracy, slowest
```

### Method 2: Interactive Demo

```bash
python whisper_demo.py
```

This will start an interactive session where you can:
- Choose which model to use
- Select from available audio files
- See transcription results with timestamps

### Method 3: Live Audio Transcription

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

#### High-Performance Live Transcription (Faster-Whisper) - Recommended
```bash
python whisper_live_faster.py
```

Ultra-fast transcription using faster-whisper (CTranslate2):
- **Up to 4x faster** than OpenAI Whisper
- **Lower memory usage** with optimized models
- **Configurable precision** (int8, float16, float32)
- **GPU and CPU support** with automatic fallback
- **Built-in VAD filtering** for optimal performance
- **Real-time performance metrics** tracking
- **Automatic transcript saving** to organized files

**Faster-Whisper Features:**
- CTranslate2 backend for maximum speed
- Multiple compute types for speed/quality balance
- Advanced model optimization (quantization)
- Performance statistics and monitoring
- Compatible with all Whisper model sizes
- **Auto-saves transcripts** to `transcripts/transcript_YYYYMMDD_HHMMSS.txt`

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

## Generated Transcripts

When using the faster-whisper live transcription (`whisper_live_faster.py`):
- Transcripts are **automatically saved** when you stop recording (Ctrl+C)
- Files are saved to `transcripts/transcript_YYYYMMDD_HHMMSS.txt`
- Each file contains:
  - Session metadata (start/end time, model settings, performance stats)
  - Detailed transcript with timestamps and confidence scores
  - Clean continuous transcript for easy reading
- The `transcripts/` folder is created automatically if it doesn't exist

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