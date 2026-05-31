import streamlit as st
import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image

# ==========================================
# 1. APPLICATION CONFIGURATION & API KEYS
# ==========================================
# Set up the webpage structure layout
st.set_page_config(
    page_title="Multi-Modal AI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Insert your active API Keys here
GEMINI_API_KEY = "AQ.Ab8RN6KBX_oYLq3IqUWfLHwuxNncnPoTiyaciVpL1tFs2ceu5A"
HUGGINGFACE_API_KEY = "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"

# Initialize the Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# Define the Hugging Face API URL for Stable Diffusion
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# ==========================================
# 2. APPLICATION STATE MANAGEMENT (MEMORY)
# ==========================================
# Streamlit reruns the whole script on user interaction. 
# We use st.session_state to keep the chat history alive.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you or your team today?"}
    ]

# ==========================================
# 3. USER INTERFACE (UI) LAYOUT
# ==========================================
st.title("🤖 Multi-Modal AI Application (Chat + Image Generator)")
st.write("---")

# Split the screen into two equal columns
col1, col2 = st.columns(2)

# ------------------------------------------
# LEFT COLUMN: GEMINI AI CHATBOT
# ------------------------------------------
with col1:
    st.header("💬 AI Text Assistant (Gemini)")
    
    # Display the conversation history from session state
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    # Capture user text input field
    user_prompt = st.chat_input("Ask me anything...")
    
    if user_prompt:
        # Display user message immediately in UI
        with chat_container:
            with st.chat_message("user"):
                st.write(user_prompt)
        
        # Append to session state history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Fetch response from Gemini model
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(user_prompt)
            ai_reply = response.text
            
            # Display AI response in UI
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(ai_reply)
            
            # Save AI response to session state history
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")

# ------------------------------------------
# RIGHT COLUMN: STABLE DIFFUSION IMAGE GENERATOR
# ------------------------------------------
with col2:
    st.header("🎨 AI Image Generator (Stable Diffusion)")
    
    # Capture user image description prompt
    image_prompt = st.text_input(
        label="Describe the image you want to generate:",
        placeholder="e.g., A futuristic library run by helper robots, digital art"
    )
    
    generate_btn = st.button("Generate Image", use_container_width=True)
    
    # Target placeholder area where image will display
    image_placeholder = st.empty()
    
    if generate_btn:
        if not image_prompt:
            st.warning("Please enter an image description prompt first!")
        else:
            with st.spinner("Generating your masterpiece... Please wait roughly 10-20 seconds."):
                try:
                    # Send payload data request to Hugging Face
                    response = requests.post(HF_API_URL, headers=headers, json={"inputs": image_prompt})
                    
                    if response.status_code == 200:
                        # Convert binary bytes data directly into a viewable PIL image format
                        image_bytes = response.content
                        image = Image.open(BytesIO(image_bytes))
                        
                        # Render image inside the dashboard container
                        image_placeholder.image(image, caption=f'Generated: "{image_prompt}"', use_container_width=True)
                    else:
                        st.error(f"Hugging Face error (Status Code {response.status_code}). The model might be starting up; try again shortly.")
                
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")