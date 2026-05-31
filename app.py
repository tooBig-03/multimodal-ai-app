import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
import time

# ==========================================
# 1. APPLICATION CONFIGURATION & SECRETS
# ==========================================
st.set_page_config(
    page_title="Multi-Modal AI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fetch API Keys securely from Streamlit Cloud Secrets dashboard
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError:
    st.error("🔑 API Keys missing! Please configure GEMINI_API_KEY and HUGGINGFACE_API_KEY in your Streamlit Cloud App Secrets settings.")
    st.stop()

# Initialize the Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Hugging Face Inference Client
client = InferenceClient(
    model="stabilityai/stable-diffusion-xl-base-1.0", 
    token=HUGGINGFACE_API_KEY
)

# ==========================================
# 2. APPLICATION STATE MANAGEMENT (MEMORY)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you or your team today?"}
    ]

# ==========================================
# 3. USER INTERFACE (UI) LAYOUT
# ==========================================
st.title("🤖 Multi-Modal AI Application (Chat + Image Generator)")
st.write("---")

col1, col2 = st.columns(2)

# ------------------------------------------
# LEFT COLUMN: GEMINI AI CHATBOT
# ------------------------------------------
with col1:
    st.header("💬 AI Text Assistant (Gemini)")
    
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    user_prompt = st.chat_input("Ask me anything...")
    
    if user_prompt:
        with chat_container:
            with st.chat_message("user"):
                st.write(user_prompt)
        
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(user_prompt)
            ai_reply = response.text
            
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(ai_reply)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")

# ------------------------------------------
# RIGHT COLUMN: STABLE DIFFUSION IMAGE GENERATOR
# ------------------------------------------
with col2:
    st.header("🎨 AI Image Generator (Stable Diffusion)")
    
    image_prompt = st.text_input(
        label="Describe the image you want to generate:",
        placeholder="e.g., A futuristic library run by helper robots, digital art"
    )
    
    generate_btn = st.button("Generate Image", use_container_width=True)
    image_placeholder = st.empty()
    
    if generate_btn:
        if not image_prompt:
            st.warning("Please enter an image description prompt first!")
        else:
            with st.spinner("Connecting to Hugging Face and generating image... (May take 10-20s)"):
                # Retry logic to combat temporary server connection drops
                success = False
                for attempt in range(3):
                    try:
                        image = client.text_to_image(image_prompt)
                        image_placeholder.image(image, caption=f'Generated: "{image_prompt}"', use_container_width=True)
                        success = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(3)  # Wait 3 seconds before retrying
                            continue
                        else:
                            st.error(f"An unexpected error occurred after multiple retries: {e}")
                            st.info("💡 Tip: The Hugging Face server might be heavily loaded right now. Try your request again in a minute.")
