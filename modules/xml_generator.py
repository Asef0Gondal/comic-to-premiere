import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any

def create_premiere_xml(image_filenames: List[str], timings: List[Dict[str, Any]], audio_filename: str, audio_duration: float) -> str:
    """
    Generate Adobe Premiere Pro compatible XML (FCP 7 XML format) from timing data.
    
    Args:
        image_filenames: List of processed image filenames
        timings: List of timing dictionaries with 'start_time' and 'duration'
        audio_filename: Name of the audio file
        audio_duration: Total duration of the audio in seconds
        
    Returns:
        XML string in FCP 7 format
    """
    
    # Constants for Premiere Pro XML
    FPS = 30
    TIMEBASE = "30"
    WIDTH = 1920
    HEIGHT = 1080
    
    def seconds_to_frames(seconds: float) -> int:
        return int(round(seconds * FPS))
    
    # Create root element
    xmeml = ET.Element('xmeml', version='4')
    
    # Create sequence
    sequence = ET.SubElement(xmeml, 'sequence')
    ET.SubElement(sequence, 'name').text = 'Comic Sequence'
    ET.SubElement(sequence, 'duration').text = str(seconds_to_frames(audio_duration))
    
    # Sequence rate
    rate = ET.SubElement(sequence, 'rate')
    ET.SubElement(rate, 'timebase').text = TIMEBASE
    ET.SubElement(rate, 'ntsc').text = 'FALSE'
    
    # Media section
    media = ET.SubElement(sequence, 'media')
    
    # Video tracks
    video = ET.SubElement(media, 'video')
    format_elem = ET.SubElement(video, 'format')
    sample_char = ET.SubElement(format_elem, 'samplecharacteristics')
    ET.SubElement(sample_char, 'width').text = str(WIDTH)
    ET.SubElement(sample_char, 'height').text = str(HEIGHT)
    
    track = ET.SubElement(video, 'track')
    
    # Add image clips to video track
    for i, (img_name, timing) in enumerate(zip(image_filenames, timings)):
        start_frames = seconds_to_frames(timing.get('start_time', 0))
        duration_frames = seconds_to_frames(timing.get('duration', 3.0))
        end_frames = start_frames + duration_frames
        
        clipitem = ET.SubElement(track, 'clipitem', id=f'clipitem-{i+1}')
        ET.SubElement(clipitem, 'name').text = img_name
        ET.SubElement(clipitem, 'duration').text = str(duration_frames + 1000) # Buffer
        
        clip_rate = ET.SubElement(clipitem, 'rate')
        ET.SubElement(clip_rate, 'timebase').text = TIMEBASE
        ET.SubElement(clip_rate, 'ntsc').text = 'FALSE'
        
        ET.SubElement(clipitem, 'start').text = str(start_frames)
        ET.SubElement(clipitem, 'end').text = str(end_frames)
        ET.SubElement(clipitem, 'in').text = '0'
        ET.SubElement(clipitem, 'out').text = str(duration_frames)
        
        # File reference
        file_elem = ET.SubElement(clipitem, 'file', id=f'file-{i+1}')
        ET.SubElement(file_elem, 'name').text = img_name
        ET.SubElement(file_elem, 'pathurl').text = img_name # Relative path
        
        file_rate = ET.SubElement(file_elem, 'rate')
        ET.SubElement(file_rate, 'timebase').text = TIMEBASE
        
        timecode = ET.SubElement(file_elem, 'timecode')
        tc_rate = ET.SubElement(timecode, 'rate')
        ET.SubElement(tc_rate, 'timebase').text = TIMEBASE
        ET.SubElement(timecode, 'string').text = '00:00:00:00'
        ET.SubElement(timecode, 'frame').text = '0'
        
    # Audio tracks
    audio = ET.SubElement(media, 'audio')
    num_channels = 2
    for channel_num in range(1, num_channels + 1):
        audio_track = ET.SubElement(audio, 'track')
        
        # Audio clip (one long clip for the whole duration)
        duration_frames = seconds_to_frames(audio_duration)
        
        clipitem = ET.SubElement(audio_track, 'clipitem', id=f'audio-clip-{channel_num}')
        ET.SubElement(clipitem, 'name').text = audio_filename
        ET.SubElement(clipitem, 'duration').text = str(duration_frames)
        
        clip_rate = ET.SubElement(clipitem, 'rate')
        ET.SubElement(clip_rate, 'timebase').text = TIMEBASE
        ET.SubElement(clip_rate, 'ntsc').text = 'FALSE'
        
        ET.SubElement(clipitem, 'start').text = '0'
        ET.SubElement(clipitem, 'end').text = str(duration_frames)
        ET.SubElement(clipitem, 'in').text = '0'
        ET.SubElement(clipitem, 'out').text = str(duration_frames)
        
        # File reference
        file_elem = ET.SubElement(clipitem, 'file', id='audio-file-1')
        ET.SubElement(file_elem, 'name').text = audio_filename
        ET.SubElement(file_elem, 'pathurl').text = audio_filename
        
        # Audio source settings
        sourcetrack = ET.SubElement(clipitem, 'sourcetrack')
        ET.SubElement(sourcetrack, 'mediatype').text = 'audio'
        ET.SubElement(sourcetrack, 'trackindex').text = str(channel_num)
    
    # Convert to string and format
    xml_str = ET.tostring(xmeml, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    return parsed_xml.toprettyxml(indent="  ")

# Legacy support alias
def generate_premiere_xml(timings, image_path, audio_path, output_path=None):
    # This is kept for backward compatibility if needed by other modules
    # In a real production app, we would update all calls to the new function
    pass
