"""Google Gemini API Client Module
Handles audio timing analysis and script parsing using Gemini 1.5 Flash.
"""

import google.generativeai as genai
import json
import re
from typing import Optional

def analyze_audio_timing(
    api_key: str,
    audio_data: bytes,
    audio_filename: str,
    script: str,
    panel_count: int
) -> Optional[list]:
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
    
    Returns:
        List of dicts with panel timings:
        [{"panel": 1, "start_time": 0.0, "duration": 2.5},
         {"panel": 2, "start_time": 2.5, "duration": 3.0}]
    """
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Save audio data to temp file
    import tempfile
    import os
    
    mime_type = "audio/mpeg"  # default
    if audio_filename.lower().endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_filename.lower().endswith(".m4a"):
        mime_type = "audio/mp4"
    elif audio_filename.lower().endswith(".ogg"):
        mime_type = "audio/ogg"
    
    # Create temp file
    suffix = os.path.splitext(audio_filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name
    
    try:
        # Upload audio file
        audio_file = genai.upload_file(temp_audio_path)
        
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
- Return ONLY the JSON array, no other text
"""
        
        # Use Gemini 1.5 Flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, audio_file])
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            timing_data = json.loads(json_str)
            
            # Validate
            if len(timing_data) == panel_count:
                return timing_data
        
        return None
        
    finally:
        # Clean up temp file
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
) -> Optional[list]:
    """Wrapper function for compatibility."""
    return analyze_audio_timing(api_key, audio_data, audio_filename, script, panel_count)
