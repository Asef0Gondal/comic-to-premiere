"""Image Processing Module
Handles resizing images to 1920x1080 with blurred background fill.
"""
from PIL import Image, ImageFilter
import io

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
    from PIL import ImageEnhance
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

def process_image_to_bytes(image_data: bytes, format: str = "JPEG") -> bytes:
    """
    Process an image and return as bytes.
    
    Args:
        image_data: Raw image bytes
        format: Output format (JPEG, PNG)
    
    Returns:
        Processed image as bytes
    """
    processed = resize_with_blur_background(image_data)
    output = io.BytesIO()
    processed.save(output, format=format, quality=95)
    output.seek(0)
    return output.getvalue()
