"""Google Gemini API Client Module
Handles audio timing analysis and script parsing using Gemini 1.5 Flash.
Includes validation and improved error handling.
"""

import google.generativeai as genai
import json
import re
import os
import tempfile
from typing import Optional, List, Dict, Tuple

def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Gemini API key format.
    
    Args:
        api_key: API key string to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    if len(api_key) < 20:
        return False, "API key is too short (minimum 20 characters)"
    
    if len(api_key) > 200:
        return False, "API key is too long"
    
    # Basic format check (alphanumeric and some special chars)
    if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
        return False, "API key contains invalid characters"
    
    return True, None

def analyze_audio_timing(
    api_key: str,
    audio_data: bytes,
    audio_filename: str,
    script: str,
    panel_count: int,
    timeout: int = 60
) -> Optional[List[Dict]]:
    """
    Analyze audio file with script to extract timing for each panel.
    
    Uses Google Gemini 1.5 Flash multimodal capabilities to analyze
    the audio and match it with the provided script.
    
    Args:
        api_key: Google Gemini API key
        audio_data: Raw audio file bytes
        audio_filename: Original filename for mime type detection
        script: The dialogue script text
        panel_count: Number of panels to sync
        timeout: Timeout in seconds (default: 60)
    
    Returns:
        List of dicts with panel timings:
        [{"panel": 1, "start_time": 0.0, "duration": 2.5},
         {"panel": 2, "start_time": 2.5, "duration": 3.0}]
        Returns None if analysis fails
    """
    # Validate inputs
    if not api_key:
        raise ValueError("API key is required")
    
    if not audio_data:
        raise ValueError("Audio data is required")
    
    if panel_count < 1:
        raise ValueError("Panel count must be at least 1")
    
    if not script or not script.strip():
        raise ValueError("Script is required")
    
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to configure Gemini API: {str(e)}")
    
    # Determine mime type
    mime_type = "audio/mpeg"  # default
    ext = audio_filename.lower().split('.')[-1]
    mime_types = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'm4a': 'audio/mp4',
        'ogg': 'audio/ogg'
    }
    mime_type = mime_types.get(ext, mime_type)
    
    # Create temp file
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Upload audio file with error handling
        try:
            audio_file = genai.upload_file(temp_audio_path)
        except Exception as e:
            raise ValueError(f"Failed to upload audio file: {str(e)}")
        
        # Create detailed prompt
        prompt = f"""You are an expert audio timing analyst for comic-to-video conversion.

I have {panel_count} comic panels and this audio narration. The script for the narration is:

{script}

Please analyze the audio and determine:
1. When each panel's dialogue/narration starts (start_time in seconds)
2. How long each panel should be displayed (duration in seconds)

Return ONLY a JSON array with this exact format:
[
  {{"panel": 1, "start_time": 0.0, "duration": 2.5}},
  {{"panel": 2, "start_time": 2.5, "duration": 3.0}}
]

Rules:
- Must have exactly {panel_count} entries
- Times must be accurate to the audio
- Each panel's start_time should equal previous panel's (start_time + duration)
- Durations should be realistic (typically 2-10 seconds per panel)
- Return ONLY the JSON array, no other text
- Do not add markdown formatting or backticks
"""
        
        # Use Gemini 1.5 Flash with timeout handling
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        
        try:
            response = model.generate_content(
                [prompt, audio_file],
                request_options={'timeout': timeout}
            )
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                raise TimeoutError(f"Audio analysis timed out after {timeout} seconds")
            elif 'quota' in error_msg.lower():
                raise ValueError("API quota exceeded. Please check your Gemini API quota.")
            else:
                raise ValueError(f"Audio analysis failed: {error_msg}")
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to find JSON array in response
        # Method 1: Look for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Method 2: Look for raw JSON array
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"Could not find JSON in response: {response_text[:200]}")
                return None
        
        try:
            timing_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"JSON string: {json_str[:200]}")
            return None
        
        # Validate timing data
        if not isinstance(timing_data, list):
            print(f"Timing data is not a list: {type(timing_data)}")
            return None
        
        if len(timing_data) != panel_count:
            print(f"Expected {panel_count} panels, got {len(timing_data)}")
            # Try to adjust if close
            if len(timing_data) > panel_count:
                timing_data = timing_data[:panel_count]
            else:
                return None
        
        # Validate each timing entry
        for i, timing in enumerate(timing_data):
            if not isinstance(timing, dict):
                print(f"Timing entry {i} is not a dict")
                return None
            
            if 'panel' not in timing or 'start_time' not in timing or 'duration' not in timing:
                print(f"Timing entry {i} missing required fields")
                return None
            
            # Ensure numeric values
            try:
                timing['panel'] = int(timing['panel'])
                timing['start_time'] = float(timing['start_time'])
                timing['duration'] = float(timing['duration'])
            except (ValueError, TypeError):
                print(f"Timing entry {i} has invalid numeric values")
                return None
            
            # Validate ranges
            if timing['start_time'] < 0 or timing['duration'] <= 0:
                print(f"Timing entry {i} has invalid ranges")
                return None
        
        return timing_data
        
    except Exception as e:
        print(f"Error in analyze_audio_timing: {e}")
        raise
        
    finally:
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

def analyze_audio_with_gemini(
    api_key: str,
    audio_data: bytes,
    audio_filename: str,
    script: str,
    panel_count: int
) -> Optional[List[Dict]]:
    """Wrapper function for backwards compatibility."""
    return analyze_audio_timing(api_key, audio_data, audio_filename, script, panel_count)

def generate_fallback_timings(panel_count: int, total_duration: float = 10.0) -> List[Dict]:
    """
    Generate evenly-spaced fallback timings when audio analysis fails.
    
    Args:
        panel_count: Number of panels
        total_duration: Total duration in seconds (default: 10.0)
    
    Returns:
        List of timing dicts with even distribution
    """
    if panel_count < 1:
        raise ValueError("Panel count must be at least 1")
    
    if total_duration <= 0:
        raise ValueError("Total duration must be positive")
    
    duration_per_panel = total_duration / panel_count
    timings = []
    
    for i in range(panel_count):
        timings.append({
            "panel": i + 1,
            "start_time": i * duration_per_panel,
            "duration": duration_per_panel
        })
    
    return timings

def estimate_audio_duration(audio_data: bytes, audio_filename: str) -> Optional[float]:
    """
    Estimate audio duration in seconds.
    
    Args:
        audio_data: Raw audio bytes
        audio_filename: Filename for extension detection
    
    Returns:
        Duration in seconds, or None if estimation fails
    """
    try:
        from mutagen import File
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_filename.split(".")[-1]}') as temp:
            temp.write(audio_data)
            temp_path = temp.name
        
        try:
            audio = File(temp_path)
            if audio and hasattr(audio.info, 'length'):
                return float(audio.info.length)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"Could not estimate audio duration: {e}")
    
    return None
