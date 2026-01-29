"""Comic-to-Premiere Web Application
A Streamlit app that syncs comic panels with voice-over using Google Gemini AI.
"""
import streamlit as st
import zipfile
import io
import tempfile
import os
from pathlib import Path

# Import our modules
from modules.image_processor import process_image_to_bytes
from modules.gemini_client import analyze_audio_timing, generate_fallback_timings
from modules.xml_generator import create_premiere_xml

# Page configuration
st.set_page_config(page_title="Comic to Premiere", page_icon="🎬", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.stApp { max-width: 1200px; margin: 0 auto; }
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
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 Comic to Premiere</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Upload comic panels and voice-over → AI syncs the timing → Export Premiere Pro XML</p>', unsafe_allow_html=True)

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ API Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get your free API key at https://aistudio.google.com")
    
    st.markdown("---")
    
    # How to use
    st.markdown("""
    **📖 How to use:**
    1. Enter your Gemini API key
    2. Upload your comic panels
    3. Upload your voice-over audio
    4. Enter your script with panel markers
    5. Click Process & Generate
    6. Download the ZIP file
    """)
    
    st.markdown("---")
    
    # About
    st.markdown("""
    **ℹ️ About:**
    This app uses Google Gemini 1.5 Flash to analyze your audio and automatically sync comic panels with dialogue timing.
    
    The generated XML can be imported directly into Adobe Premiere Pro.
    """)

# Main content area
col1, col2 = st.columns([1, 1])

# Initialize session state for panels
if 'panel_count' not in st.session_state:
    st.session_state.panel_count = 3
    if 'audio_count' not in st.session_state:
        st.session_state.audio_count = 1

with col1:
    st.markdown('<h3 class="step-header">📸 Panel Setup</h3>', unsafe_allow_html=True)
    
    # Panel count controls
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 2])
    with sub_col1:
        if st.button("➕ Add Panel"):
            st.session_state.panel_count += 1
            st.rerun()
    with sub_col2:
        if st.button("➖ Remove") and st.session_state.panel_count > 1:
            st.session_state.panel_count -= 1
                st.rerun()
    remove_text = st.checkbox("Remove text/speech bubbles", value=True, help="Use AI to detect and crop out dialogue text from panels")
    st.markdown("---")
    
    # Panel uploaders
    uploaded_panels = []
    for i in range(st.session_state.panel_count):
        uploaded_file = st.file_uploader(f"Panel {i+1}", type=['png', 'jpg', 'jpeg', 'webp'], key=f"panel{i}")
        uploaded_panels.append(uploaded_file)

with col2:
    st.markdown('<h3 class="step-header">🎤 Audio & Script</h3>', unsafe_allow_html=True)
    
    # Audio count controls
    audio_sub_col1, audio_sub_col2, audio_sub_col3 = st.columns([1, 1, 2])
    
    with audio_sub_col1:
        if st.button("➕ Add Audio"):
            st.session_state.audio_count += 1
            st.rerun()
    
    with audio_sub_col2:
        if st.button("➖ Remove Audio") and st.session_state.audio_count > 1:
            st.session_state.audio_count -= 1
            st.rerun()
    
    with audio_sub_col3:
        st.write(f"Total audio files: {st.session_state.audio_count}")
    
    st.markdown("---")
    
    # Audio uploaders
    uploaded_audios = []
    for i in range(st.session_state.audio_count):
        audio_file = st.file_uploader(f"Voice-Over Audio {i+1}", type=['mp3', 'wav', 'ogg', 'm4a'], help="Upload the full voice-over audio file", key=f"audio{i}")
        uploaded_audios.append(audio_file    if uploaded_audios and uploaded_audios[0]:
        st.audio(uploaded_audios[0], format=f"audio/{uploaded_audios[0].name.split('.')[-1]}")
    
    st.markdown("---")    
    
    script = st.text_area("Script / Dialogue", height=200, 
        placeholder="Enter your script here. You can mark panel boundaries like:\n\nPanel 1: Hello, this is the first panel's dialogue...\n\nPanel 2: And this continues in the second panel...\n\nOr just paste the full dialogue and the AI will divide it evenly.",
        help="The AI will use this to determine timing for each panel")

st.markdown("---")
st.markdown('<h3 class="step-header">⚡ Generate</h3>', unsafe_allow_html=True)

# Process button
panels_uploaded = sum(1 for p in uploaded_panels if p is not None)
ready_to_process = api_key and panels_uploaded > 0 and any(uploaded_audios) and script.strip()
if not ready_to_process:
    missing = []
    if not api_key:
        missing.append("API Key in sidebar")
    if panels_uploaded == 0:
        missing.append("At least one panel image")
        if not any(uploaded_audios):
    if not script.strip():
        missing.append("Script text")
    st.warning(f"⚠️ Missing: {', '.join(missing)}")

if st.button("🚀 Process & Generate XML", type="primary", disabled=not ready_to_process):
    with st.spinner("Processing..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process images
            status_text.text("Step 1/4: Processing images...")
            progress_bar.progress(10)
            
            processed_images = []
            image_filenames = []
            for i, panel in enumerate(uploaded_panels):
                if panel is not None:
                    img_data = panel.read()
                    status_text.text(f"Step 1/4: Processing panel {i+1}...")
                    processed = process_image_to_bytes(img_data, remove_text=remove_text, api_key=api_key)
                    filename = f"panel_{i+1:03d}.jpg"
                    processed_images.append((filename, processed))
                    image_filenames.append(filename)
                    panel.seek(0)  # Reset for potential reuse
            progress_bar.progress(30)
            
            # Step 2: Analyze audio with Gemini
            status_text.text("Step 2/4: Analyzing audio with AI...")
                audio_data = uploaded_audios[0].read()                audio_filename = uploaded_audios[0].name            
                audio_filename = uploaded_audios[0].name                audio_data=audio_data,
                
                # Get the first audio file for processing                
                timings = analyze_audio_timing(            progress_bar.progress(60)
            
            # Fallback if AI analysis failed
            if timings is None:
                status_text.text("AI analysis returned no results, using fallback timing...")
                # Estimate 3 seconds per panel as fallback
                total_duration = len(image_filenames) * 3.0
                timings = generate_fallback_timings(len(image_filenames), total_duration)
                st.warning("⚠️ Could not analyze audio timing. Using evenly distributed fallback timings.")
            
            # Step 3: Generate XML
            status_text.text("Step 3/4: Generating Premiere Pro XML...")
            
            # Calculate total audio duration from timings
            audio_duration = max([t.get('end', 0) for t in timings], default=30.0) if timings else 30.0
            
            xml_content = create_premiere_xml(
                image_filenames=image_filenames,
                timings=timings,
                audio_filename=audio_filename,
                audio_duration=audio_duration
            )
            progress_bar.progress(80)
            
            # Step 4: Create ZIP file
            status_text.text("Step 4/4: Creating ZIP package...")
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
            
            zip_buffer.seek(0)
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Success message
            st.success("✅ Processing complete! Your ZIP file is ready.")
            
            # Show timing results
            with st.expander("📊 View Generated Timings"):
                st.json(timings)
            
            # Download button
            st.download_button(
                label="📥 Download ZIP",
                data=zip_buffer.getvalue(),
                file_name="comic_to_premiere.zip",
                mime="application/zip",
                type="primary"
            )
            
            # Instructions
            st.info("""
            **📦 Next steps:**
            1. Unzip the downloaded file
            2. Open Adobe Premiere Pro
            3. Go to File → Import
            4. Select the `comic_sequence.xml` file
            5. Your sequence will appear with all panels and audio synced!
            """)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

st.markdown("---")
st.markdown('<p style="text-align: center; color: #999; font-size: 0.8rem;">Made with ❤️ using Streamlit & Google Gemini AI</p>', unsafe_allow_html=True)
