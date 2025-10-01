#!/usr/bin/env python3
"""
Transcript utility functions for Whisper applications.
Provides shared functionality for saving and managing transcripts across all Whisper tools.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class TranscriptLogger:
    """
    A utility class for logging and saving transcripts across all Whisper tools.
    """
    
    def __init__(self, tool_name: str = "whisper", model_name: str = "unknown"):
        """
        Initialize the transcript logger.
        
        Args:
            tool_name (str): Name of the tool (e.g., "faster-whisper", "vad", "basic")
            model_name (str): Name of the Whisper model being used
        """
        self.tool_name = tool_name
        self.model_name = model_name
        self.session_start_time = None
        self.transcript_entries = []
        self.session_metadata = {}
        
    def start_session(self, **metadata):
        """
        Start a new transcript session.
        
        Args:
            **metadata: Additional metadata to store (device, compute_type, etc.)
        """
        self.session_start_time = datetime.now()
        self.transcript_entries = []
        self.session_metadata = metadata
        
    def add_transcript(self, text: str, timestamp: Optional[str] = None, **extra_data):
        """
        Add a transcript entry.
        
        Args:
            text (str): The transcribed text
            timestamp (str, optional): Timestamp string. If None, current time is used.
            **extra_data: Additional data (language, confidence, duration, etc.)
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
        entry = {
            'timestamp': timestamp,
            'text': text,
            **extra_data
        }
        
        self.transcript_entries.append(entry)
        
    def save_transcript(self, output_dir: str = "transcripts") -> Optional[str]:
        """
        Save the complete transcript to a timestamped text file.
        
        Args:
            output_dir (str): Directory to save transcripts (default: "transcripts")
            
        Returns:
            Optional[str]: Path to saved file, or None if no transcript to save
        """
        if not self.transcript_entries or not self.session_start_time:
            print("📝 No transcript to save.")
            return None
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created directory: {output_dir}/")
            
        # Create filename with session start timestamp
        filename = os.path.join(
            output_dir, 
            f"transcript_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                self._write_transcript_content(f)
                
            print(f"📝 Transcript saved to: {filename}")
            print(f"   Total segments: {len(self.transcript_entries)}")
            print(f"   File size: {os.path.getsize(filename)} bytes")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            return None
            
    def _write_transcript_content(self, f):
        """Write the transcript content to the file object."""
        # Write header
        f.write("=" * 70 + "\n")
        f.write(f"{self.tool_name.upper()} TRANSCRIPTION SESSION\n")
        f.write("=" * 70 + "\n")
        f.write(f"Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tool: {self.tool_name}\n")
        f.write(f"Model: {self.model_name}\n")
        
        # Write metadata
        for key, value in self.session_metadata.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
        f.write(f"Total segments: {len(self.transcript_entries)}\n")
        
        # Write performance stats if available
        if 'transcription_times' in self.session_metadata:
            times = self.session_metadata['transcription_times']
            if times:
                avg_time = sum(times) / len(times)
                f.write(f"Average transcription time: {avg_time:.2f}s\n")
                
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED TRANSCRIPT\n")
        f.write("=" * 70 + "\n\n")
        
        # Write detailed transcript entries
        for i, entry in enumerate(self.transcript_entries, 1):
            f.write(f"[{entry['timestamp']}] Segment {i}\n")
            f.write(f"Text: {entry['text']}\n")
            
            # Write additional entry data
            for key, value in entry.items():
                if key not in ['timestamp', 'text']:
                    if key == 'confidence':
                        f.write(f"{key.title()}: {value:.2f}\n")
                    elif key == 'duration':
                        f.write(f"Processing time: {value:.2f}s\n")
                    else:
                        f.write(f"{key.title()}: {value}\n")
                        
            f.write("-" * 50 + "\n\n")
            
        # Write continuous transcript
        f.write("=" * 70 + "\n")
        f.write("CONTINUOUS TRANSCRIPT\n")
        f.write("=" * 70 + "\n\n")
        
        continuous_text = " ".join([entry['text'] for entry in self.transcript_entries])
        f.write(continuous_text)
        f.write("\n")
        
    def get_continuous_transcript(self) -> str:
        """
        Get the continuous transcript as a single string.
        
        Returns:
            str: All transcript entries joined together
        """
        return " ".join([entry['text'] for entry in self.transcript_entries])
        
    def get_transcript_count(self) -> int:
        """Get the number of transcript entries."""
        return len(self.transcript_entries)
        
    def clear_transcript(self):
        """Clear all transcript entries."""
        self.transcript_entries = []


# Convenience functions for quick usage

def create_transcript_logger(tool_name: str, model_name: str = "unknown") -> TranscriptLogger:
    """
    Create a new transcript logger instance.
    
    Args:
        tool_name (str): Name of the tool using the logger
        model_name (str): Whisper model name
        
    Returns:
        TranscriptLogger: New logger instance
    """
    return TranscriptLogger(tool_name=tool_name, model_name=model_name)


def save_simple_transcript(text: str, tool_name: str = "whisper", model_name: str = "unknown") -> Optional[str]:
    """
    Quick function to save a simple transcript from a single text string.
    
    Args:
        text (str): The transcript text to save
        tool_name (str): Name of the tool
        model_name (str): Model name
        
    Returns:
        Optional[str]: Path to saved file, or None if failed
    """
    logger = TranscriptLogger(tool_name=tool_name, model_name=model_name)
    logger.start_session()
    logger.add_transcript(text)
    return logger.save_transcript()


def save_transcript_from_segments(segments: List[Dict[str, Any]], tool_name: str = "whisper", 
                                model_name: str = "unknown", **metadata) -> Optional[str]:
    """
    Save transcript from a list of segment dictionaries.
    
    Args:
        segments (List[Dict[str, Any]]): List of segment dictionaries with 'text' key
        tool_name (str): Name of the tool
        model_name (str): Model name
        **metadata: Additional session metadata
        
    Returns:
        Optional[str]: Path to saved file, or None if failed
    """
    logger = TranscriptLogger(tool_name=tool_name, model_name=model_name)
    logger.start_session(**metadata)
    
    for segment in segments:
        logger.add_transcript(**segment)
        
    return logger.save_transcript()