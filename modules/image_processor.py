"""Image Processing Module with OpenCV Panel Detection
Automatically detects and crops individual comic scenes from vertical panels.
Uses OpenCV contour detection for fast, reliable panel splitting.
"""
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2
import numpy as np
from typing import List, Tuple, Optional


def validate_image(image_data) -> Tuple[bool, Optional[str]]:
    """
    Validate that uploaded file is a valid image.
    """
    try:
        if hasattr(image_data, 'read'):
            data = image_data.read()
            image_data.seek(0)
        else:
            data = image_data
        
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        
        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image is too small (minimum 100x100 pixels)"
        
        if width > 10000 or height > 10000:
            return False, "Image is too large (maximum 10000x10000 pixels)"
        
        return True, None
    
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def detect_panels(img_array: np.ndarray, min_area: int = 3000) -> List[Tuple[int, int, int, int]]:
    """
    Detect individual panel boundaries using OpenCV contour detection.
    
    Args:
        img_array: OpenCV image array (BGR)
        min_area: Minimum panel area in pixels
    
    Returns:
        List of (x, y, w, h) tuples for each detected panel, sorted top-to-bottom
    """
    orig_h, orig_w = img_array.shape[:2]
    proc = img_array.copy()
    
    # Downscale large images for faster processing
    max_dim = 1200
    scale = 1.0
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        proc = cv2.resize(proc, (int(orig_w*scale), int(orig_h*scale)), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Adaptive threshold to detect panel borders
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 15, 9)
    
    # Morphological closing to connect panel borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # Scale area back to original image size
        area_orig = area / (scale * scale)
        
        # Filter by area and size
        if area_orig >= min_area and w > 30 and h > 30:
            # Convert back to original coordinates
            x_o = int(x/scale)
            y_o = int(y/scale) 
            w_o = int(w/scale)
            h_o = int(h/scale)
            panels.append((x_o, y_o, w_o, h_o))
    
    # Sort top-to-bottom, left-to-right
    panels_sorted = sorted(panels, key=lambda r: (r[1], r[0]))
    
    return panels_sorted


def split_panels_from_image(image_data: bytes, min_area: int = 3000) -> List[Tuple[bytes, int]]:
    """
    Automatically split a comic panel into individual scenes using OpenCV.
    
    Args:
        image_data: Raw image bytes
        min_area: Minimum panel area in pixels (default 3000, lower for small panels)
    
    Returns:
        List of (cropped_image_bytes, scene_number) tuples
        If detection fails, returns original image as single scene
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")
            return [(image_data, 1)]
        
        # Detect panels
        panels = detect_panels(img, min_area=min_area)
        
        if not panels or len(panels) <= 1:
            print(f"Panel detection found {len(panels) if panels else 0} panels, using full image")
            return [(image_data, 1)]
        
        print(f"Detected {len(panels)} panels")
        
        result = []
        for i, (x, y, w, h) in enumerate(panels, start=1):
            # Add small padding
            pad = 8
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            
            # Crop panel
            crop = img[y0:y1, x0:x1].copy()
            
            # Encode back to bytes
            success, encoded = cv2.imencode('.png', crop)
            if success:
                result.append((encoded.tobytes(), i))
                print(f"Cropped scene {i}: position ({y0}, {y1}), size {w}x{h}")
        
        return result if result else [(image_data, 1)]
    
    except Exception as e:
        print(f"Error splitting panels: {e}")
        import traceback
        traceback.print_exc()
        return [(image_data, 1)]


def process_image_to_bytes(image_data: bytes, remove_text: bool = False, api_key: str = None) -> bytes:
    """
    Process image: resize to 1920x1080 with blurred background.
    
    Args:
        image_data: Raw image bytes
        remove_text: (unused, kept for compatibility)
        api_key: (unused, kept for compatibility)
    
    Returns:
        Processed image as bytes
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        target_width, target_height = 1920, 1080
        img_width, img_height = img.size
        
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        img_ratio = img_width / img_height
        
        # Create blurred background
        background = img.copy()
        background = background.resize((target_width, target_height), Image.LANCZOS)
        background = background.filter(ImageFilter.GaussianBlur(radius=30))
        
        # Darken the background
        enhancer = ImageEnhance.Brightness(background)
        background = enhancer.enhance(0.6)
        
        # Resize main image to fit
        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center the image
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        
        background.paste(img, (x, y))
        
        # Convert to bytes
        output = io.BytesIO()
        background.save(output, format='JPEG', quality=95)
        
        return output.getvalue()
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return image_data
