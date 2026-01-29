237"""Comic-to-Premiere Web Application
A Streamlit app that syncs comic panels with voice-over using Google Gemini AI.

Production-ready version with error handling, validation, and enhanced features.
"""
import streamlit as st
import zipfile
import io
import tempfile
import os
from pathlib import Path
from PIL import Image
import traceback
from typing import List, Optional, Tuple

# Import our modules
from modules.image_processor import process_image_to_bytes, validate_image, split_panels_from_image
from modules.gemini_client import analyze_audio_timing, generate_fallback_timings, validate_api_key
from modules.xml_generator import create_premiere_xml

# Constants
MAX_FILE_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 20
MAX_PANELS = 50
MIN_PANELS = 1
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'webp']
SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'ogg', 'm4a']

# Page configuration
st.set_page_config(
    page_title="Comic to Premiere", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stApp { max-width: 1400px; margin: 0 auto; }
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: bold;
}
.step-header {
    color: #667eea;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.success-box {
    padding: 1rem;
    background-color: #d4edda;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
}
.info-box {
    padding: 1rem;
    background-color: #d1ecf1;
    border-radius: 0.5rem;
    border: 1px solid #bee5eb;
    margin: 1rem 0;
}
.warning-box {
    padding: 1rem;
    background-color: #fff3cd;
    border-radius: 0.5rem;
    border: 1px solid #ffc107;
}
.error-box {
    padding: 1rem;
    background-color: #f8d7da;
    border-radius: 0.5rem;
    border: 1px solid #f5c6cb;
}
.preview-image {
    max-width: 100%;
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def validate_file_size(file, max_size_mb: int, file_type: str = "File") -> Tuple[bool, Optional[str]]:
    """Validate file size."""
    if file is None:
        return False, f"{file_type} is required"
    
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size_bytes = max_size_mb * 1024 * 1024
    if size > max_size_bytes:
        return False, f"{file_type} is too large ({format_file_size(size)}). Maximum: {max_size_mb}MB"
    
    if size == 0:
        return False, f"{file_type} is empty"
    
    return True, None

def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return None, "⚠️ API quota exceeded. Please check your Gemini API quota or try again later."
        elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
            return None, "⚠️ Invalid API key. Please check your Gemini API key."
        elif "timeout" in error_msg.lower():
            return None, "⚠️ Request timed out. Please try again."
        else:
            return None, f"⚠️ API Error: {error_msg}"

# Initialize session state
if 'panel_count' not in st.session_state:
    st.session_state.panel_count = 3
if 'audio_count' not in st.session_state:
    st.session_state.audio_count = 1
if 'processed_images_cache' not in st.session_state:
    st.session_state.processed_images_cache = {}
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Header
st.markdown('<h1 class="main-header">🎬 Comic to Premiere</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Upload comic panels and voice-over → AI syncs the timing → Export Premiere Pro XML</p>', unsafe_allow_html=True)

# Sidebar for API Key and Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key_input = st.text_input(
        "Google Gemini API Key", 
        type="password", 
        help="Get your free API key at https://aistudio.google.com",
        key="api_key_input"
    )
    
    # Try to get from secrets first, then from input
    api_key = st.secrets.get("GEMINI_API_KEY", api_key_input) if hasattr(st, 'secrets') else api_key_input
    
    # Validate API key format
    if api_key:
        is_valid, key_msg = validate_api_key(api_key)
        if is_valid:
            st.success("✅ API key format valid")
        else:
            st.error(key_msg)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Processing Settings")
    
    remove_text = st.checkbox(
        "Remove text/speech bubbles", 
        value=True, 
        help="Use AI to detect and crop out dialogue text from panels"
    )
    
    show_previews = st.checkbox(
        "Show image previews", 
        value=False, 
        help="Display processed images before export (slower)"
    )
    
    fallback_panel_duration = st.slider(
        "Fallback duration per panel (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Duration used if AI analysis fails"
    )
    
    st.markdown("---")
    
    # How to use
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        1. Enter your Gemini API key
        2. Upload your comic panels
        3. Upload your voice-over audio
        4. Enter your script with panel markers
        5. Click Process & Generate
        6. Download the ZIP file
        7. Import XML into Premiere Pro
        """)
    
    # About
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        This app uses Google Gemini 1.5 Flash to analyze your audio and automatically sync comic panels with dialogue timing.
        
        The generated XML uses FCP 7 format compatible with Adobe Premiere Pro.
        
        **Version:** 2.0 (Production Ready)
        """)
    
    # Stats
    with st.expander("📊 System Info", expanded=False):
        st.markdown(f"""
        **Limits:**
        - Max panels: {MAX_PANELS}
        - Max image size: {MAX_IMAGE_SIZE_MB}MB
        - Max audio size: {MAX_FILE_SIZE_MB}MB
        - Supported images: {', '.join(SUPPORTED_IMAGE_FORMATS)}
        - Supported audio: {', '.join(SUPPORTED_AUDIO_FORMATS)}
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 class="step-header">📸 Panel Setup</h3>', unsafe_allow_html=True)
    
    # Multi-file panel uploader
    st.markdown("##### Upload Comic Panels")
    
    # Initialize session state for uploaded panels
    if 'uploaded_panels_list' not in st.session_state:
        st.session_state.uploaded_panels_list = []
    if 'panels_order_reversed' not in st.session_state:
        st.session_state.panels_order_reversed = False
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Select all panel images (they will be ordered as you select them)",
        type=SUPPORTED_IMAGE_FORMATS,
        accept_multiple_files=True,
        help=f"Upload all panels at once (max {MAX_IMAGE_SIZE_MB}MB per image, up to {MAX_PANELS} panels total)",
        key="panels_uploader"
    )
    
    # Update session state when files are uploaded
    if uploaded_files:
        st.session_state.uploaded_panels_list = uploaded_files
        st.session_state.panel_count = len(uploaded_files)
    
    # Show panel count and order controls
    if st.session_state.uploaded_panels_list:
        col_info, col_reverse = st.columns([3, 1])
        
        with col_info:
            st.info(f"📊 {len(st.session_state.uploaded_panels_list)} panel(s) uploaded")
        
        with col_reverse:
            if st.button("🔄 Reverse Order"):
                st.session_state.uploaded_panels_list = list(reversed(st.session_state.uploaded_panels_list))
                st.session_state.panels_order_reversed = not st.session_state.panels_order_reversed
                st.rerun()
        
        # Display panel order preview
        with st.expander("📋 View Panel Order", expanded=False):
            for i, panel in enumerate(st.session_state.uploaded_panels_list):
                st.text(f"{i+1}. {panel.name}")
    
    st.markdown("---")
    
    # Validate uploaded panels
    uploaded_panels = []
    panel_errors = []
    
    if st.session_state.uploaded_panels_list:
        for i, uploaded_file in enumerate(st.session_state.uploaded_panels_list):
            # Validate file size
            is_valid_size, size_error = validate_file_size(uploaded_file, MAX_IMAGE_SIZE_MB, f"Panel {i+1}")
            if not is_valid_size:
                st.error(f"Panel {i+1} ({uploaded_file.name}): {size_error}")
                panel_errors.append(size_error)
                uploaded_file = None
            else:
                # Validate image integrity
                is_valid_img, img_error = validate_image(uploaded_file)
                if not is_valid_img:
                    st.error(f"Panel {i+1} ({uploaded_file.name}): {img_error}")
                    panel_errors.append(img_error)
                    uploaded_file = None
            
            uploaded_panels.append(uploaded_file)
        
        # Check if we exceeded max panels
        if len(st.session_state.uploaded_panels_list) > MAX_PANELS:
            st.error(f"⚠️ Maximum {MAX_PANELS} panels allowed. You uploaded {len(st.session_state.uploaded_panels_list)}.")
            panel_errors.append(f"Too many panels: {len(st.session_state.uploaded_panels_list)}")

with col2:
    st.markdown('<h3 class="step-header">🎤 Audio & Script</h3>', unsafe_allow_html=True)
    
    # Audio section (simplified to single audio for now)
    st.markdown("##### Voice-Over Audio")
    
    audio_file = st.file_uploader(
        "Upload audio file", 
        type=SUPPORTED_AUDIO_FORMATS,
        help=f"Upload the full voice-over audio file (max {MAX_FILE_SIZE_MB}MB)",
        key="audio_main"
    )
    
    audio_error = None
   if audio_file:
        is_valid_size, size_error = validate_file_size(audio_file, MAX_FILE_SIZE_MB, "Audio file")
        if not is_valid_size:
            st.error(size_error)
            audio_error = size_error
            audio_file = None
        else:
            st.success(f"✅ {format_file_size(audio_file.size)}")
            try:
                st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
            except Exception as e:
                st.warning("Could not preview audio")
    
    st.markdown("---")    
    
    # Script input with character counter
    script = st.text_area(
        "Script / Dialogue", 
        height=200, 
        placeholder="Enter your script here. You can mark panel boundaries like:\n\nPanel 1: Hello, this is the first panel's dialogue...\n\nPanel 2: And this continues in the second panel...\n\nOr just paste the full dialogue and the AI will divide it evenly.",
        help="The AI will use this to determine timing for each panel",
        max_chars=10000
    )
    
    if script:
        char_count = len(script)
        word_count = len(script.split())
        st.caption(f"📝 {char_count} characters, {word_count} words")

# Info boxes
st.markdown("---")

# Show status info
panels_uploaded = sum(1 for p in uploaded_panels if p is not None)
ready_to_process = api_key and panels_uploaded > 0 and audio_file and script.strip() and not panel_errors and not audio_error

if not ready_to_process:
    st.markdown('<h3 class="step-header">⚡ Ready to Generate?</h3>', unsafe_allow_html=True)
    
    missing = []
    if not api_key:
        missing.append("✗ API Key in sidebar")
    else:
        missing.append("✓ API Key configured")
    
    if panels_uploaded == 0:
        missing.append("✗ At least one panel image")
    else:
        missing.append(f"✓ {panels_uploaded} panel(s) uploaded")
    
    if not audio_file:
        missing.append("✗ Audio file")
    else:
        missing.append("✓ Audio file uploaded")
    
    if not script.strip():
        missing.append("✗ Script text")
    else:
        missing.append("✓ Script provided")
    
    if panel_errors:
        missing.append(f"✗ {len(panel_errors)} file error(s)")
    
    if audio_error:
        missing.append(f"✗ Audio error")
    
    # Display checklist
    st.markdown("**Checklist:**")
    for item in missing:
        if item.startswith("✓"):
            st.markdown(f"- {item}")
        else:
            st.markdown(f"- {item}")
    
    if not ready_to_process:
        st.warning("⚠️ Please complete all requirements above before processing.")

# Process button
st.markdown('<h3 class="step-header">⚡ Generate</h3>', unsafe_allow_html=True)

if st.button("🚀 Process & Generate XML", type="primary", disabled=not ready_to_process, use_container_width=True):
    st.session_state.processing_complete = False
    
    with st.spinner("Processing..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process images
            status_text.text("Step 1/4: Processing images...")
            progress_bar.progress(5)
            
            processed_images = []
            image_filenames = []
            
            for i, panel in enumerate(uploaded_panels):
                if panel is not None:
                    try:
                        status_text.text(f"Step 1/4: Processing panel {i+1}/{panels_uploaded}...")
                        
                        # Check cache first
                        cache_key = f"{panel.name}_{remove_text}"
                        if cache_key in st.session_state.processed_images_cache:
                            processed = st.session_state.processed_images_cache[cache_key]
                        else:
                            img_data = panel.read()
                            
                            # Process with error handling
                            processed, error = safe_api_call(
                                process_image_to_bytes, 
                                img_data, 
                                remove_text=remove_text, 
                                api_key=api_key
                            )
                            
                            if error:
                                st.error(f"Panel {i+1}: {error}")
                                # Use original image as fallback
                                processed = img_data
                            else:
                                # Cache the result
                                st.session_state.processed_images_cache[cache_key] = processed
                            
                            panel.seek(0)  # Reset for potential reuse
                        
                        filename = f"panel_{i+1:03d}.jpg"
                        processed_images.append((filename, processed))
                        image_filenames.append(filename)
                        
                        # Update progress
                        progress = 5 + int(25 * (i + 1) / panels_uploaded)
                        progress_bar.progress(progress)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing panel {i+1}: {str(e)}")
                        continue
            
            if not processed_images:
                raise ValueError("No panels were successfully processed")
            
            progress_bar.progress(30)
            
            # Step 2: Analyze audio with Gemini
            status_text.text("Step 2/4: Analyzing audio with AI...")
            
            try:
                audio_data = audio_file.read()
                audio_filename = audio_file.name
                audio_file.seek(0)
                
                timings, error = safe_api_call(
                    analyze_audio_timing,
                    api_key=api_key,
                    audio_data=audio_data,
                    audio_filename=audio_filename,
                    script=script,
                    panel_count=len(image_filenames)
                )
                
                if error:
                    st.warning(error)
                    timings = None
                
                progress_bar.progress(60)
                
                # Fallback if AI analysis failed
                if timings is None:
                    status_text.text("AI analysis unavailable, using fallback timing...")
                    total_duration = len(image_filenames) * fallback_panel_duration
                    timings = generate_fallback_timings(len(image_filenames), total_duration)
                    st.info(f"ℹ️ Using fallback timing: {fallback_panel_duration}s per panel")
                else:
                    st.success("✅ AI timing analysis complete")
                
            except Exception as e:
                st.error(f"❌ Error analyzing audio: {str(e)}")
                # Use fallback
                total_duration = len(image_filenames) * fallback_panel_duration
                timings = generate_fallback_timings(len(image_filenames), total_duration)
                st.info(f"ℹ️ Using fallback timing: {fallback_panel_duration}s per panel")
            
            # Step 3: Generate XML
            status_text.text("Step 3/4: Generating Premiere Pro XML...")
            
            try:
                # Calculate total audio duration from timings
                audio_duration = max(
                    [t.get('start_time', 0) + t.get('duration', 0) for t in timings], 
                    default=30.0
                ) if timings else 30.0
                
                xml_content = create_premiere_xml(
                    image_filenames=image_filenames,
                    timings=timings,
                    audio_filename=audio_filename,
                    audio_duration=audio_duration
                )
                
                progress_bar.progress(80)
                
            except Exception as e:
                st.error(f"❌ Error generating XML: {str(e)}")
                st.error(traceback.format_exc())
                raise
            
            # Step 4: Create ZIP file
            status_text.text("Step 4/4: Creating ZIP package...")
            
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Add XML
                    zf.writestr("comic_sequence.xml", xml_content)
                    
                    # Add processed images
                    for filename, img_data in processed_images:
                        zf.writestr(filename, img_data)
                    
                    # Add original audio
                    audio_file.seek(0)
                    zf.writestr(audio_filename, audio_file.read())
                    
                    # Add README
                    readme_content = f"""Comic-to-Premiere Export
=========================

Files included:
- comic_sequence.xml: Premiere Pro project file
- {len(processed_images)} panel image(s): panel_001.jpg, panel_002.jpg, etc.
- {audio_filename}: Original audio file

Import Instructions:
1. Extract all files to the same directory
2. Open Adobe Premiere Pro
3. Go to File → Import
4. Select comic_sequence.xml
5. Your sequence will appear with all panels and audio synced!

Generated by Comic-to-Premiere
{panels_uploaded} panels | {audio_duration:.1f}s duration
"""
                    zf.writestr("README.txt", readme_content)
                
                zip_buffer.seek(0)
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                st.session_state.processing_complete = True
                st.session_state.zip_data = zip_buffer.getvalue()
                st.session_state.timings = timings
                st.session_state.processed_images_preview = processed_images if show_previews else []
                
            except Exception as e:
                st.error(f"❌ Error creating ZIP: {str(e)}")
                raise
            
            # Success!
            st.balloons()
            st.success("✅ Processing complete! Your ZIP file is ready.")
            
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")
            st.error("Please check your inputs and try again.")
            if st.checkbox("Show detailed error"):
                st.code(traceback.format_exc())

# Display results if processing complete
if st.session_state.processing_complete:
    st.markdown("---")
    st.markdown("### 📦 Download Your Files")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Download button
        st.download_button(
            label="📥 Download ZIP Package",
            data=st.session_state.zip_data,
            file_name="comic_to_premiere.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        # Show file size
        zip_size = len(st.session_state.zip_data)
        st.metric("Package Size", format_file_size(zip_size))
    
    # Show timing results
    with st.expander("📊 View Generated Timings"):
        st.json(st.session_state.timings)
        
        # Calculate stats
        total_duration = sum(t.get('duration', 0) for t in st.session_state.timings)
        avg_duration = total_duration / len(st.session_state.timings) if st.session_state.timings else 0
        
        cols = st.columns(3)
        cols[0].metric("Total Duration", f"{total_duration:.1f}s")
        cols[1].metric("Panels", len(st.session_state.timings))
        cols[2].metric("Avg per Panel", f"{avg_duration:.1f}s")
    
    # Show preview if enabled
    if st.session_state.processed_images_preview:
        with st.expander("🖼️ Preview Processed Images"):
            preview_cols = st.columns(min(3, len(st.session_state.processed_images_preview)))
            for i, (filename, img_data) in enumerate(st.session_state.processed_images_preview):
                col_idx = i % len(preview_cols)
                with preview_cols[col_idx]:
                    st.image(img_data, caption=filename, use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📖 Next Steps")
    st.info("""
    **How to import into Adobe Premiere Pro:**
    
    1. **Unzip** the downloaded file to a folder on your computer
    2. **Open** Adobe Premiere Pro
    3. Go to **File → Import**
    4. Navigate to the extracted folder and select **comic_sequence.xml**
    5. Your sequence will appear in the Project panel
    6. **Drag** the sequence to the timeline to start editing!
    
    **Tips:**
    - Keep all files in the same directory for proper linking
    - The XML uses FCP 7 format compatible with Premiere Pro
    - You can adjust timing manually in Premiere if needed
    - Audio and video are on separate tracks for easy editing
    """)
    
    # Reset button
    if st.button("🔄 Process Another Comic", use_container_width=True):
        st.session_state.processing_complete = False
        st.session_state.processed_images_cache = {}
        st.rerun()

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #999; font-size: 0.8rem;">Made with ❤️ using Streamlit & Google Gemini AI | Version 2.0</p>', unsafe_allow_html=True)

