import streamlit as st
import os
import zipfile
from io import BytesIO
from modules.image_processor import process_images
from modules.gemini_client import analyze_audio_with_gemini
from modules.xml_generator import generate_premiere_xml

st.set_page_config(page_title="Comic to Premiere", page_icon="🎬", layout="wide")

st.title("🎬 Comic Panel to Premiere Pro Timeline")
st.markdown("Upload your comic panels and voiceover audio to generate a Premiere Pro project")

# API Key input
api_key = st.text_input("Google Gemini API Key", type="password", help="Get your free API key from https://aistudio.google.com")

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Comic Panels")
    image_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
with col2:
    st.subheader("🎤 Upload Voiceover")
    audio_file = st.file_uploader("Choose audio file", type=['mp3', 'wav', 'ogg', 'm4a'])

if st.button("Generate Premiere Project", type="primary", disabled=not (image_files and audio_file and api_key)):
    with st.spinner("Processing your content..."):
        try:
            # Process images
            st.info("📸 Processing images...")
            processed_images = process_images(image_files)
            
            # Analyze audio
            st.info("🎵 Analyzing audio timing with AI...")
            audio_bytes = audio_file.read()
            timings = analyze_audio_with_gemini(audio_bytes, audio_file.name, len(image_files), api_key)
            
            # Generate XML
            st.info("📝 Generating Premiere Pro XML...")
            xml_content = generate_premiere_xml(processed_images, audio_file.name, timings)
            
            # Create ZIP file
            st.info("📦 Creating download package...")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add processed images
                for i, img_data in enumerate(processed_images):
                    zip_file.writestr(f"panel_{i+1:03d}.png", img_data)
                
                # Add audio file
                zip_file.writestr(audio_file.name, audio_bytes)
                
                # Add XML file
                zip_file.writestr("premiere_project.xml", xml_content)
            
            zip_buffer.seek(0)
            
            st.success("✅ Project generated successfully!")
            st.download_button(
                label="📥 Download Premiere Project ZIP",
                data=zip_buffer,
                file_name="comic_premiere_project.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("### How to use:")
st.markdown("""
1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com)
2. Upload your comic panel images in order
3. Upload your voiceover audio file
4. Click Generate and download your Premiere Pro project
5. Extract the ZIP and import the XML file in Premiere Pro
""")
