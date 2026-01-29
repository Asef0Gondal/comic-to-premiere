"""Image Processing Module
Handles resizing images to 1920x1080 with blurred background fill.
Includes AI-powered text/speech bubble removal.
"""
from PIL import Image, ImageFilter, ImageEnhance
import io
import re
import json

def detect_text_free_region(image_data: bytes, api_key: str) -> tuple:
    """
    Use Gemini Vision to detect the main artwork area without text/speech bubbles.
    
    Args:
        image_data: Raw image bytes
        api_key: Google Gemini API key
    
    Returns:
        Tuple of (left, top, right, bottom) as percentages (0-100), or None if detection fails
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    # Upload image for analysis
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save to temp buffer for upload
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=85)
    img_buffer.seek(0)
    
    uploaded_file = genai.upload_file(
        data=img_buffer.getvalue(),
        display_name="comic_panel.jpg",
        mime_type="image/jpeg"
    )
    
    prompt = """Analyze this comic panel image. I need to crop out ALL text, speech bubbles, dialogue boxes, captions, and any text overlays to keep ONLY the artwork/illustration.

Look at the image and find the LARGEST rectangular area that contains the main artwork WITHOUT any text, speech bubbles, or dialogue.

Return ONLY a JSON object with the crop coordinates as percentages (0-100) of the image dimensions:
{"left": X, "top": Y, "right": X2, "bottom": Y2}

Where:
- left: percentage from left edge where the crop starts
- top: percentage from top edge where the crop starts  
- right: percentage from left edge where the crop ends
- bottom: percentage from top edge where the crop ends

If there's no text in the image, return {"left": 0, "top": 0, "right": 100, "bottom": 100}

IMPORTANT: Return ONLY the JSON, no other text."""
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, uploaded_file])
    
    # Clean up
    try:
        uploaded_file.delete()
    except:
        pass
    
    # Parse response
    response_text = response.text.strip()
    
    # Extract JSON
    json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            coords = json.loads(json_match.group(1))
            return (coords.get('left', 0), coords.get('top', 0), coords.get('right', 100), coords.get('bottom', 100))
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None

def crop_text_from_image(image_data: bytes, api_key: str) -> bytes:
    """
    Crop text/speech bubbles from a comic panel using AI detection.
    
    Args:
        image_data: Raw image bytes
        api_key: Gemini API key for vision analysis
    
    Returns:
        Cropped image as bytes (JPEG)
    """
    # Detect text-free region
    region = detect_text_free_region(image_data, api_key)
    
    # Open image
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    if region:
        left_pct, top_pct, right_pct, bottom_pct = region
        
        # Convert percentages to pixels
        left = int(width * left_pct / 100)
        top = int(height * top_pct / 100)
        right = int(width * right_pct / 100)
        bottom = int(height * bottom_pct / 100)
        
        # Ensure valid crop area (at least 10% of original)
        min_width = width * 0.1
        min_height = height * 0.1
        if (right - left) >= min_width and (bottom - top) >= min_height:
            img = img.crop((left, top, right, bottom))
    
    # Return as bytes
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=95)
    output.seek(0)
    return output.getvalue()

def resize_with_blur_background(image_data: bytes, target_size: tuple = (1920, 1080)) -> Image.Image:
    """
    Resize an image to target size with a blurred background fill.
    
    The original image is centered on a blurred, scaled version of itself
    that fills the entire target canvas.
    
    Args:
        image_data: Raw image bytes
        target_size: Target dimensions (width, height), default 1920x1080
    
    Returns:
        PIL Image object with the composited result
    """
    # Open the image
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary (handles PNG transparency, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    target_width, target_height = target_size
    orig_width, orig_height = img.size
    
    # Scale image to fill the entire target area (may crop)
    bg_scale = max(target_width / orig_width, target_height / orig_height)
    bg_size = (int(orig_width * bg_scale), int(orig_height * bg_scale))
    background = img.resize(bg_size, Image.Resampling.LANCZOS)
    
    # Crop to target size (center crop)
    left = (bg_size[0] - target_width) // 2
    top = (bg_size[1] - target_height) // 2
    background = background.crop((left, top, left + target_width, top + target_height))
    
    # Apply strong blur
    background = background.filter(ImageFilter.GaussianBlur(radius=30))
    
    # Darken the background slightly for better contrast
    enhancer = ImageEnhance.Brightness(background)
    background = enhancer.enhance(0.5)
    
    # Scale foreground image to fit within target (preserve aspect ratio)
    fg_scale = min(target_width / orig_width, target_height / orig_height)
    # Leave some padding (90% of available space)
    fg_scale *= 0.9
    fg_size = (int(orig_width * fg_scale), int(orig_height * fg_scale))
    foreground = img.resize(fg_size, Image.Resampling.LANCZOS)
    
    # Calculate position to center the foreground
    x = (target_width - fg_size[0]) // 2
    y = (target_height - fg_size[1]) // 2
    
    # Composite
    background.paste(foreground, (x, y))
    
    return background

def process_image_to_bytes(image_data: bytes, format: str = "JPEG", remove_text: bool = False, api_key: str = None) -> bytes:
    """
    Process an image and return as bytes.
    
    Args:
        image_data: Raw image bytes
        format: Output format (JPEG, PNG)
        remove_text: If True, use AI to crop out text/speech bubbles first
        api_key: Gemini API key (required if remove_text is True)
    
    Returns:
        Processed image as bytes
    """
    # First, crop out text if requested
    if remove_text and api_key:
        image_data = crop_text_from_image(image_data, api_key)
    
    # Then apply the blur background resize
    processed = resize_with_blur_background(image_data)
    output = io.BytesIO()
    processed.save(output, format=format, quality=95)
    output.seek(0)
    return output.getvalue()
