# app.py
import streamlit as st
from retrieval_model import RetrievalModel
from generative import generate_gemini_text
import os
import re

st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("💬 Mental Health Chatbot (Retrieval + Gemini Flash)")

# Sidebar / settings
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Retrieval", "Generative (Gemini)", "Hybrid (Retrieval first)"])
gemini_key = st.sidebar.text_input("GEMINI API KEY (or set env var)", type="password", placeholder="paste your key here (optional)")
model_name = st.sidebar.text_input("Gemini model", value="gemini-1.5-flash")
threshold = st.sidebar.slider("Retrieval confidence threshold", 0.0, 1.0, 0.35)

# Emergency phrases - escalate immediately without sending to model
EMERGENCY_PHRASES = [
    "i want to end", "i want to die", "kill myself", "suicide",
    "i will kill myself", "i will end my life", "ending it all",
    "i want to harm myself", "self harm", "end my life","suicide",
    "die by suicide","suicide"
]

# India-specific emergency helpline message
EMERGENCY_MESSAGE = (
    "I’m really concerned about your safety.\n\n"
    "**If you are in India**, please reach out to one of these helplines right now:\n"
    "- ☎️ AASRA: +91-9820466726 (24x7)\n"
    "- ☎️ Vandrevala Foundation: 1860 2662 345 / +91-9999 666 555\n"
    "- ☎️ iCall (TISS): +91-9152987821 (Mon–Sat, 8 AM–10 PM)\n"
    "- ☎️ Snehi: +91-9582208181 (24x7)\n"
    "- 🚨 National emergency: 112\n\n"
    "You are not alone — help is available, please talk to someone."
)

def contains_emergency(text):
    t = text.lower()
    return any(phrase in t for phrase in EMERGENCY_PHRASES)


def contains_emergency(text):
    t = text.lower()
    return any(phrase in t for phrase in EMERGENCY_PHRASES)

# Load retrieval model lazily
@st.cache_resource
def load_retrieval(threshold):
    rm = RetrievalModel(threshold=threshold)
    return rm

if mode in ("Retrieval", "Hybrid (Retrieval first)"):
    try:
        retrieval = load_retrieval(threshold)
    except Exception as e:
        st.sidebar.error(f"Could not load retrieval model: {e}")
        retrieval = None
else:
    retrieval = None

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.text_input("You:", key="input")
send_btn = st.button("Send")

if send_btn and user_input.strip():
    user_text = user_input.strip()
    # Emergency check
    if contains_emergency(user_text):
        st.session_state.messages.append(("You", user_text))
        st.session_state.messages.append(("Bot", EMERGENCY_MESSAGE))
    else:
        bot_reply = None
        if mode == "Retrieval":
            if retrieval:
                resp, tag, prob = retrieval.get_response(user_text)
                bot_reply = resp or "I'm not sure I understand. Would you like to tell me more?"
            else:
                bot_reply = "Retrieval model not loaded. Run training script."
        elif mode == "Generative (Gemini)":
            key_to_use = gemini_key or os.getenv("GEMINI_API_KEY")
            if not key_to_use:
                bot_reply = "Gemini API key not provided. Set it in sidebar or as environment variable GEMINI_API_KEY."
            else:
                bot_reply = generate_gemini_text(user_text, api_key=key_to_use, model=model_name)
        else:  # Hybrid: try retrieval first, if low confidence fallback to Gemini
            if retrieval:
                resp, tag, prob = retrieval.get_response(user_text)
                if resp and prob >= threshold:
                    bot_reply = resp + f"\n\n_(retrieval tag: {tag}, conf={prob:.2f})_"
                else:
                    key_to_use = gemini_key or os.getenv("GEMINI_API_KEY")
                    if not key_to_use:
                        bot_reply = resp or "No Gemini key provided for fallback."
                    else:
                        gem_reply = generate_gemini_text(user_text, api_key=key_to_use, model=model_name)
                        bot_reply = gem_reply
            else:
                bot_reply = "Retrieval model missing; set mode to Generative or train the retrieval model."

        st.session_state.messages.append(("You", user_text))
        st.session_state.messages.append(("Bot", bot_reply))

# Display chat
for role, msg in st.session_state.messages:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Footer
st.markdown("---")
st.caption("Tip: keep your Gemini API key safe. For production hosting, follow Google AI Studio / Gemini API usage and policies.")
