import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
from datetime import datetime
import random

# Updated art style presets with new additions
ART_STYLES = {
    "Anime Styles": [
        "Painted Anime", "Cinematic", "Digital Painting", "Concept Art",
        "Vintage Anime", "Neon Vintage Anime", "3D Disney Character",
        "2D Disney Character", "50s Infomercial Anime", "Studio Ghibli",
        "Drawn Anime", "Cute Anime", "Soft Anime"
    ],
    "Painting & Realism": [
        "Oil Painting - Realism", "Oil Painting - Old", "Fantasy Painting",
        "Fantasy Landscape", "Fantasy Portrait", "Digital Painting",
        "Watercolor", "Painterly", "Concept Sketch", "Disney Sketch"
    ],
    "Comic/Illustration": [
        "Vintage Comic", "Franco-Belgian Comic", "Tintin Comic",
        "Flat Illustration", "Vintage Pulp Art", "Medieval",
        "Traditional Japanese", "YuGiOh Art", "MTG Card"
    ],
    "3D & Digital": [
        "3D Pokemon", "Painted Pokemon", "3D Isometric Icon",
        "Cute 3D Icon", "Claymotion", "3D Emoji", "Cute 3D Icon Set"
    ],
    "Retro/Vintage": [
        "1990s Photo", "1980s Photo", "1970s Photo", "1960s Photo",
        "1950s Photo", "1940s Photo", "1930s Photo", "1920s Photo",
        "50s Enamel Sign", "Vintage Pulp Art"
    ],
    "Specialized Techniques": [
        "Pixel Art", "Oil Painting", "Crayon Drawing", "Pencil Sketch",
        "Tattoo Design", "Professional Photo", "Cortoon Style"
    ],
    "Fantasy & Unique": [
        "Fantasy World Map", "Fantasy City Map", "Mongo Style",
        "Nihongo Pointing", "Waifu Style", "Cursed Photo",
        "Furry - Cinematic", "Furry - Pointed", "Claymotion"
    ]
}

# Set up the app title and icon
st.set_page_config(page_title="AKHAND IMAGE GENERATION V2", page_icon="🎨")

# Initialize Hugging Face Inference client
def get_client():
    api_key = st.secrets.get("HUGGINGFACE_TOKEN")
    if not api_key:
        st.error("API token not found. Check secrets configuration.")
        st.stop()
    return InferenceClient(token=api_key)

client = get_client()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# App UI
st.title("🎨 AKHAND IMAGE GENERATION V2 ")
st.write("Professional Art Generation with 100+ Style Presets")

# Input parameters
with st.expander("🖌️ Art Configuration", expanded=True):
    # Style selection
    col_style1, col_style2 = st.columns(2)
    with col_style1:
        style_category = st.selectbox(
            "Art Style Category",
            list(ART_STYLES.keys()),
            index=0
        )
    with col_style2:
        selected_style = st.selectbox(
            "Specific Art Style",
            ART_STYLES[style_category],
            index=0
        )
    
    # Prompt inputs
    col_prompt1, col_prompt2 = st.columns([3, 2])
    with col_prompt1:
        prompt = st.text_input("Main Art Description", 
                             placeholder="Describe your artwork...")
    with col_prompt2:
        negative_prompt = st.text_input("Exclusion List", 
                                      placeholder="Elements to avoid...")

# Advanced parameters
with st.expander("⚙️ Technical Settings", expanded=False):
    col_params1, col_params2, col_params3, col_params4 = st.columns(4)
    with col_params1:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    with col_params2:
        steps = st.slider("Generation Steps", 10, 150, 50)
    with col_params3:
        num_images = st.slider("Number of Variations", 1, 4, 1)
    with col_params4:
        height = st.selectbox("Canvas Height", [512, 768])
        width = st.selectbox("Canvas Width", [512, 768])

# Generate button
if st.button("Create Artwork", type="primary"):
    if not prompt:
        st.warning("Please enter an art description")
        st.stop()
    
    # Build enhanced prompt
    full_prompt = f"{prompt}, {selected_style} style, masterpiece, ultra-detailed"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    images = []
    
    try:
        for i in range(num_images):
            status_text.text(f"Creating Variation {i+1}/{num_images}...")
            progress_bar.progress((i+1)/num_images)
            
            # Generate unique seed for each variation
            seed = random.randint(0, 2**32 - 1)
            
            # Generate artwork
            result = client.text_to_image(
                prompt=full_prompt,
                model="black-forest-labs/FLUX.1-dev",
                negative_prompt=negative_prompt or None,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=steps,
                seed=seed
            )
            
            # Convert and store image
            if isinstance(result, Image.Image):
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
            else:
                image_bytes = result
            
            images.append(image_bytes)
            
            # Display artwork
            with st.expander(f"Variation #{i+1} - {selected_style}", expanded=True):
                st.image(image_bytes, use_container_width=True)
                st.download_button(
                    label="Download Artwork",
                    data=image_bytes,
                    file_name=f"{selected_style.replace(' ', '_')}_{i+1}.png",
                    mime="image/png",
                    key=f"dl_{i}"
                )
        
        # Update history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_prompt": prompt,
            "style": selected_style,
            "full_prompt": full_prompt,
            "images": images,
            "params": {
                "guidance_scale": guidance_scale,
                "steps": steps,
                "size": f"{width}x{height}",
                "seeds": [seed]
            }
        })
        
        # Maintain history limit
        if len(st.session_state.history) > 5:
            st.session_state.history.pop(0)
        
        progress_bar.empty()
        status_text.success("Art generation complete!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Generation failed: {str(e)}")
        st.stop()

# History section
if st.session_state.history:
    st.markdown("---")
    st.subheader("🎭 Creation History")
    
    for gen in reversed(st.session_state.history):
        with st.expander(f"{gen['timestamp']} - {gen['style']}"):
            st.write(f"**Concept:** {gen['base_prompt']}")
            st.write(f"**Style:** {gen['style']}")
            st.write(f"**Full Prompt:** `{gen['full_prompt']}`")
            if gen['params'].get('negative_prompt'):
                st.write(f"**Exclusions:** {gen['negative_prompt']}")
            
            hist_cols = st.columns(len(gen['images']))
            for idx, (col, img_bytes) in enumerate(zip(hist_cols, gen['images'])):
                with col:
                    st.image(img_bytes, use_container_width=True)
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=f"hist_{gen['timestamp']}_{idx+1}.png",
                        mime="image/png",
                        key=f"hist_{gen['timestamp']}_{idx}"
                    )

# App footer
st.markdown("---")
st.markdown("""
**Art Studio Tools**
- Developed by [SHUVO](https://sites.google.com/view/mr-shuvo/Home)
- Powered by [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- 150+ Artistic Style Presets
- Professional-Grade Generation Parameters
- Temporary Browser-Based Session Storage
""")
