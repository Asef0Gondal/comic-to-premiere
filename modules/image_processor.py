"""Image Processing Module
Handles resizing images to 1920x1080 with blurred background fill.
Includes AI-powered text/speech bubble removal and validation.
"""
from PIL import Image, ImageFilter, ImageEnhance
import io
import re
import json
from typing import Tuple, Optional

def validate_image(image_data) -> Tuple[bool, Optional[str]]:
    """
    Validate that uploaded file is a valid image.
    
    Args:
        image_data: File-like object or bytes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Read data if it's a file-like object
        if hasattr(image_data, 'read'):
            data = image_data.read()
            image_data.seek(0)  # Reset
        else:
            data = image_data
        
        # Try to open as image
        img = Image.open(io.BytesIO(data))
        img.verify()  # Verify it's a valid image
        
        # Re-open after verify (verify closes the file)
        img = Image.open(io.BytesIO(data))
        
        # Check dimensions
        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image is too small (minimum 100x100 pixels)"
        
        if width > 10000 or height > 10000:
            return False, "Image is too large (maximum 10000x10000 pixels)"
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def detect_text_free_region(image_data: bytes, api_key: str, timeout: int = 30) -> Optional[Tuple[float, float, float, float]]:
    """
    Use Gemini Vision to detect the main artwork area without text/speech bubbles.
    
    Args:
        image_data: Raw image bytes
        api_key: Google Gemini API key
        timeout: API timeout in seconds
    
    Returns:
        Tuple of (left, top, right, bottom) as percentages (0-100), or None if detection fails
    """
    try:
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
        
        # Upload with error handling
        try:
            uploaded_file = genai.upload_file(
                data=img_buffer.getvalue(),
                display_name="comic_panel.jpg",
                mime_type="image/jpeg"
            )
        except Exception as e:
            print(f"Failed to upload image: {e}")
            return None
        
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
        
        # Set timeout if available
        generation_config = {"temperature": 0.1}
        
        try:
            response = model.generate_content(
                [prompt, uploaded_file],
                generation_config=generation_config
            )
        except Exception as e:
            print(f"Generation failed: {e}")
            return None
        
        # Clean up uploaded file
        try:
            uploaded_file.delete()
        except:
            pass
        
        # Parse response
        response_text = response.text.strip()
        
        # Extract JSON - try multiple patterns
        json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try finding raw JSON
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        try:
            coords = json.loads(json_str)
            left = float(coords.get('left', 0))
            top = float(coords.get('top', 0))
            right = float(coords.get('right', 100))
            bottom = float(coords.get('bottom', 100))
            
            # Validate coordinates
            if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
                return None
            
            return (left, top, right, bottom)
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None
    
    except Exception as e:
        print(f"Error in detect_text_free_region: {e}")
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
    # Open image
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Detect text-free region with timeout
    try:
        region = detect_text_free_region(image_data, api_key, timeout=30)
    except Exception as e:
        print(f"Text detection failed: {e}")
        region = None
    
    if region:
        left_pct, top_pct, right_pct, bottom_pct = region
        
        # Convert percentages to pixels
        left = int(width * left_pct / 100)
        top = int(height * top_pct / 100)
        right = int(width * right_pct / 100)
        bottom = int(height * bottom_pct / 100)
        
        # Ensure valid crop area (at least 20% of original)
        min_width = width * 0.2
        min_height = height * 0.2
        
        if (right - left) >= min_width and (bottom - top) >= min_height:
            # Crop is valid
            img = img.crop((left, top, right, bottom))
        else:
            print(f"Crop area too small, using original image")
    else:
        print("No text region detected, using original image")
    
    # Return as bytes
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=95)
    output.seek(0)
    return output.getvalue()

def resize_with_blur_background(image_data: bytes, target_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
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

def process_image_to_bytes(
    image_data: bytes, 
    format: str = "JPEG", 
    remove_text: bool = False, 
    api_key: Optional[str] = None
) -> bytes:
    """
    Process an image and return as bytes.
    
    Args:
        image_data: Raw image bytes
        format: Output format (JPEG, PNG)
        remove_text: If True, use AI to crop out text/speech bubbles first
        api_key: Gemini API key (required if remove_text is True)
    
    Returns:
        Processed image as bytes
    
    Raises:
        ValueError: If remove_text is True but api_key is not provided
    """
    # Validate inputs
    if remove_text and not api_key:
        raise ValueError("API key is required for text removal")
    
    # First, crop out text if requested
    if remove_text and api_key:
        try:
            image_data = crop_text_from_image(image_data, api_key)
        except Exception as e:
            print(f"Text removal failed, using original image: {e}")
            # Continue with original image
    
    # Then apply the blur background resize
    try:
        processed = resize_with_blur_background(image_data)
        output = io.BytesIO()
        processed.save(output, format=format, quality=95)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")
