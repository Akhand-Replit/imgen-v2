import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
from datetime import datetime
import random

# Art style presets based on your images
ART_STYLES = {
    "Anime Styles": [
        "Pointed Anime", "Cinemotic", "Digital Painting", "Concept Art",
        "Vintage Anime", "Neon Vintage Anime", "3D Disney Character",
        "2D Disney Character", "50s Infomercial Anime"
    ],
    "Comic/Illustration": [
        "Vintage Comic", "Franco-Belgian Comic", "Tintin Comic",
        "Flat Illustration", "Vintage Pulp Art", "Medieval",
        "Traditional Japanese", "YuGiOh Art", "MTG Card"
    ],
    "3D Styles": [
        "3D Pokemon", "Painted Pokemon", "3D Isometric Icon",
        "Cute 3D Icon", "Claymotion", "3D Emoji"
    ],
    "Retro/Vintage": [
        "1990s Photo", "1980s Photo", "1970s Photo", "1960s Photo",
        "1950s Photo", "1940s Photo", "1930s Photo", "1920s Photo"
    ],
    "Specialized Techniques": [
        "Pixel Art", "Oil Painting", "Watercolor", "Painterly",
        "Concept Sketch", "Disney Sketch", "Crayon Drawing",
        "Pencil Sketch", "Tattoo Design"
    ],
    "Unique Categories": [
        "Furry - Cinematic", "Furry - Pointed", "Cursed Photo",
        "Fantasy World Map", "Fantasy City Map", "Mongo Style",
        "Nihongo Pointing", "Waifu Style", "Cortoon Style"
    ]
}

# Set up the app title and icon
st.set_page_config(page_title="FLUX Art Generator", page_icon="🎨")

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
st.title("🎨 FLUX Art Generator")
st.write("Create unique artwork with style presets and advanced controls")

# Input parameters
with st.expander("⚙️ Generation Settings", expanded=True):
    # Art style selection
    style_category = st.selectbox(
        "Select Art Style Category",
        list(ART_STYLES.keys()),
        index=0
    )
    
    selected_style = st.selectbox(
        "Choose Specific Style",
        ART_STYLES[style_category],
        index=0
    )
    
    # Main prompt inputs
    prompt = st.text_input("Base Prompt", 
                         placeholder="Describe the main subject/scene...")
    
    negative_prompt = st.text_input("Negative prompt (optional)", 
                                  placeholder="Elements to exclude...")
    
    # Advanced parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    with col2:
        steps = st.slider("Number of Steps", 10, 150, 50)
    with col3:
        num_images = st.slider("Number of Images", 1, 4, 1)
    with col4:
        height = st.selectbox("Height", [512, 768])
        width = st.selectbox("Width", [512, 768])

# Generate button
if st.button("Generate Artwork", type="primary"):
    if not prompt:
        st.warning("Please enter a base prompt")
        st.stop()
    
    # Combine base prompt with selected style
    full_prompt = f"{prompt}, {selected_style} style"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    images = []
    
    try:
        for i in range(num_images):
            status_text.text(f"Generating artwork {i+1} of {num_images}...")
            progress_bar.progress((i+1)/num_images)
            
            # Generate unique seed for each image
            seed = random.randint(0, 2**32 - 1)
            
            # Generate single image per API call with unique seed
            result = client.text_to_image(
                prompt=full_prompt,
                model="black-forest-labs/FLUX.1-dev",
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=steps,
                seed=seed
            )
            
            # Convert PIL Image to bytes
            if isinstance(result, Image.Image):
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
            else:
                image_bytes = result
            
            images.append(image_bytes)
            
            # Display each image as it's generated
            with st.expander(f"Artwork {i+1} - {selected_style}", expanded=True):
                st.image(image_bytes, use_container_width=True)
                st.download_button(
                    label="Download",
                    data=image_bytes,
                    file_name=f"{selected_style.replace(' ', '_')}_{i+1}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )
        
        # Add to history (store last 5 generations)
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
        
        # Keep only last 5 generations
        if len(st.session_state.history) > 5:
            st.session_state.history.pop(0)
        
        progress_bar.empty()
        status_text.success("All artworks generated successfully!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Error generating artwork {i+1}: {str(e)}")
        st.stop()

# Display history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Generation History")
    
    for gen in reversed(st.session_state.history):
        with st.expander(f"🕒 {gen['timestamp']} - {gen['style']}"):
            st.write(f"**Base Prompt:** {gen['base_prompt']}")
            st.write(f"**Style:** {gen['style']}")
            st.write(f"**Full Prompt:** {gen['full_prompt']}")
            if gen.get('negative_prompt'):
                st.write(f"**Negative Prompt:** {gen['negative_prompt']}")
            st.write(f"**Parameters:** Guidance {gen['params']['guidance_scale']}, Steps {gen['params']['steps']}, Size {gen['params']['size']}")
            
            hist_cols = st.columns(len(gen['images']))
            for idx, (col, img_bytes) in enumerate(zip(hist_cols, gen['images'])):
                with col:
                    st.image(img_bytes, use_container_width=True)
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=f"hist_{gen['timestamp']}_{idx+1}.png",
                        mime="image/png",
                        key=f"hist_dl_{gen['timestamp']}_{idx}"
                    )

# App info
st.markdown("---")
st.markdown("""
**App Info:**
- Model: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- Art styles curated from your provided references
- Images stored temporarily in browser session
- Built with Streamlit & Hugging Face Inference API
""")
