import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_premiere_xml(timings, image_path, audio_path, output_path="comic_timeline.xml"):
    """
    Generate Adobe Premiere Pro compatible XML from timing data.
    
    Args:
        timings: List of timing dictionaries from analyze_comic_audio
        image_path: Path to the comic panel image
        audio_path: Path to the audio file
        output_path: Output XML file path
    """
    
    # Create root element
    xmeml = ET.Element('xmeml', version='4')
    
    # Create sequence
    sequence = ET.SubElement(xmeml, 'sequence')
    ET.SubElement(sequence, 'name').text = 'Comic Timeline'
    ET.SubElement(sequence, 'duration').text = str(sum(t['duration'] for t in timings))
    
    # Create video track
    media = ET.SubElement(sequence, 'media')
    video = ET.SubElement(media, 'video')
    track = ET.SubElement(video, 'track')
    
    # Add clips for each panel
    for timing in timings:
        clipitem = ET.SubElement(track, 'clipitem')
        ET.SubElement(clipitem, 'name').text = f"Panel {timing['panel']}"
        ET.SubElement(clipitem, 'start').text = str(timing['start_time'])
        ET.SubElement(clipitem, 'end').text = str(timing['start_time'] + timing['duration'])
        ET.SubElement(clipitem, 'in').text = '0'
        ET.SubElement(clipitem, 'out').text = str(timing['duration'])
        
        # Add file reference
        file_elem = ET.SubElement(clipitem, 'file')
        ET.SubElement(file_elem, 'name').text = image_path
        ET.SubElement(file_elem, 'pathurl').text = f'file://localhost/{image_path}'
    
    # Add audio track
    audio = ET.SubElement(media, 'audio')
    audio_track = ET.SubElement(audio, 'track')
    audio_clip = ET.SubElement(audio_track, 'clipitem')
    ET.SubElement(audio_clip, 'name').text = 'Audio'
    ET.SubElement(audio_clip, 'start').text = '0'
    ET.SubElement(audio_clip, 'end').text = str(sum(t['duration'] for t in timings))
    
    audio_file = ET.SubElement(audio_clip, 'file')
    ET.SubElement(audio_file, 'name').text = audio_path
    ET.SubElement(audio_file, 'pathurl').text = f'file://localhost/{audio_path}'
    
    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(xmeml)).toprettyxml(indent="  ")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(xml_str)
    
    return output_path

# Alias for backwards compatibility
create_premiere_xml = generate_premiere_xml
