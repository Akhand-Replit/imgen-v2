import streamlit as st
from huggingface_hub import InferenceClient
from datetime import datetime
import re

# Set up page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Initialize Hugging Face client
if "HF_TOKEN" not in st.secrets:
    st.error("Hugging Face API token not found in Streamlit secrets.")
    st.stop()

client = InferenceClient(token=st.secrets["HF_TOKEN"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define helper functions
def parse_response(response):
    """Parse the model response into structured sections"""
    sections = {
        "Thinking Role": None,
        "Problem Definition": None,
        "Task Execution": None,
        "Final Answer": None
    }
    
    # Use regex to extract each section
    for section in sections:
        match = re.search(fr"{section}:\s*(.*?)(?=\n\w+:|$)", response, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections

def generate_transcript():
    """Generate chat transcript text"""
    transcript = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            transcript.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            transcript.append(f"Assistant:\n{msg['content']}")
    return "\n\n".join(transcript)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            sections = parse_response(message["content"])
            if sections["Thinking Role"]:
                st.markdown(f"🧠 Thinking Role:** {sections['Thinking Role']}")
            if sections["Problem Definition"]:
                st.markdown(f"🔍 Problem Definition:** {sections['Problem Definition']}")
            if sections["Task Execution"]:
                st.markdown(f"⚙ Task Execution:** {sections['Task Execution']}")
            if sections["Final Answer"]:
                st.markdown(f"📝 Final Answer:** {sections['Final Answer']}")

# Chat input form
with st.form("chat_input", clear_on_submit=True):
    model_name = st.text_input("Model Name", value="DeepSeek-R1")
    user_input = st.text_area("Your Message", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate prompt with conversation history
    prompt = f"""You are a helpful assistant. Always format responses with:
- Thinking Role: [Your assumed role]
- Problem Definition: [Clear problem statement]
- Task Execution: [Step-by-step processing]
- Final Answer: [Concise solution]

Conversation History:
"""
    for msg in st.session_state.messages[-4:]:  # Keep recent history
        if msg["role"] == "user":
            prompt += f"\nUser: {msg['content']}"
        elif msg["role"] == "assistant":
            prompt += f"\nAssistant: {msg['content']}"
    
    prompt += f"\n\nUser: {user_input}\nAssistant:"
    
    # Generate response
    try:
        response = client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        st.stop()
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
    st.rerun()

# Download transcript
if st.session_state.messages:
    st.divider()
    with st.expander("📥 Download Chat Transcript"):
        if st.checkbox("Generate transcript (approved)"):
            transcript = generate_transcript()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name=f"chat_transcript_{timestamp}.txt",
                mime="text/plain"
            )
