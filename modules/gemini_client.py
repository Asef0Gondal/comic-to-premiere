"""Google Gemini API Client Module
Handles audio timing analysis using Gemini 1.5 Flash.
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
        List of dicts with panel timing:
        [{"panel": 1, "start": 0.0, "end": 3.5}, ...]
        Returns None if analysis fails
    """
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Determine mime type based on file extension
    ext = audio_filename.lower().split('.')[-1]
    mime_types = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'm4a': 'audio/mp4',
    }
    mime_type = mime_types.get(ext, 'audio/mpeg')
    
    # Create the prompt
    prompt = f"""I am providing an audio file and a script. The script has {panel_count} panels.

Please analyze the audio and provide the exact Start Time and End Time in seconds for each panel's dialogue.

SCRIPT:
{script}

IMPORTANT INSTRUCTIONS:
1. Listen to the audio carefully and identify when each panel's dialogue starts and ends.
2. If the script doesn't clearly mark panel boundaries, divide the dialogue evenly based on the content.
3. Ensure there are exactly {panel_count} panel entries in your response.
4. Times should be in seconds with up to 2 decimal places.
5. Panels should be sequential - each panel starts where the previous one ends.

Return the data as a pure JSON list in EXACTLY this format (no other text, just the JSON):
[
  {{"panel": 1, "start": 0.0, "end": 3.5}},
  {{"panel": 2, "start": 3.5, "end": 7.2}},
  ...
]"""

    # Upload the audio file
    audio_file = genai.upload_file(
        data=audio_data,
        display_name=audio_filename,
        mime_type=mime_type
    )
    
    # Use Gemini 1.5 Flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate response
    response = model.generate_content([prompt, audio_file])
    
    # Clean up the uploaded file
    try:
        audio_file.delete()
    except:
        pass
    
    # Parse the response
    response_text = response.text.strip()
    
    # Handle cases where the model wraps JSON in markdown code blocks
    json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            timings = json.loads(json_str)
            return timings
        except json.JSONDecodeError:
            pass
    
    # If extraction failed, try parsing the whole response
    try:
        timings = json.loads(response_text)
        return timings
    except json.JSONDecodeError:
        return None

def generate_fallback_timings(panel_count: int, total_duration: float) -> list:
    """
    Generate evenly distributed fallback timings when AI analysis fails.
    
    Args:
        panel_count: Number of panels
        total_duration: Total audio duration in seconds
    
    Returns:
        List of timing dicts
    """
    duration_per_panel = total_duration / panel_count
    timings = []
    for i in range(panel_count):
        timings.append({
            "panel": i + 1,
            "start": round(i * duration_per_panel, 2),
            "end": round((i + 1) * duration_per_panel, 2)
        })
    return timings

def analyze_audio_with_gemini(audio_bytes, audio_filename, panel_count, api_key):
    """
    Wrapper function for compatibility with app.py.
    Analyzes audio and returns timing data for comic panels.
    
    Args:
        audio_bytes: Raw audio file bytes
        audio_filename: Original filename
        panel_count: Number of panels
        api_key: Google Gemini API key
    
    Returns:
        List of timing dictionaries or None if analysis fails
    """
    # Create a simple generic script for panel analysis
    script = f"Comic panels 1-{panel_count} with dialogue"
    
    return analyze_audio_timing(
        api_key=api_key,
        audio_data=audio_bytes,
        audio_filename=audio_filename,
        script=script,
        panel_count=panel_count
    )
