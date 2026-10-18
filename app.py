import streamlit as st
import time
from groq import Groq
from transformers import pipeline

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


st.set_page_config(
    page_title="Lumière",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .stChatMessage {
        background-color: #f0f7ff;
        border-radius: 12px;
        padding: 10px;
    }
    .stChatInput input {
        border-radius: 20px;
        border: 1.5px solid #90caf9;
    }
    h1 {
        color: #1565c0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    p {
        color: #546e7a;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>✨ Lumière</h1>", unsafe_allow_html=True)
st.markdown("<p>A gentle space to talk about how you're feeling. You're not alone.</p>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_detector = load_emotion_model()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = "neutral"

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "show_breathing" not in st.session_state:
    st.session_state.show_breathing = False

if "show_grounding" not in st.session_state:
    st.session_state.show_grounding = False


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    emotion_result = emotion_detector(user_input)[0]
    detected_emotion = emotion_result["label"] if emotion_result["score"] >= 0.7 else "uncertain"
    st.session_state.detected_emotion = detected_emotion
    st.session_state.emotion_history.append(detected_emotion)

    with st.chat_message("user"):
        st.write(user_input)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": f"""You are Lumière — someone who has personally been through anxiety and hard times, and came out the other side. You are now talking with someone who might be going through something similar.

You talk like a real friend who genuinely gets it. Not a therapist. Not a chatbot. Just someone warm who understands.

You make people feel heard first. Always. Before anything else.

You keep the conversation going naturally — not with forced questions, just like a real conversation flows.

You never give advice unless asked. You never suggest solutions. You just make them feel like whatever they're feeling is completely okay and normal.
Keep every response under 2 sentences. Short is more human than long.
Use simple everyday words. Never use expressions like "big ol'" or informal spellings like "tryin'" — just normal natural English.
The user's detected emotion is: {detected_emotion}.

If someone seems in crisis, gently share: 3114 (France, free, 24/7)."""
            }
        ] + st.session_state.messages
    )

    reply = response.choices[0].message.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    with st.chat_message("assistant"):
        st.write(reply)


negative = ["fear", "sadness", "anger", "disgust"]
history = st.session_state.emotion_history

if len(history) >= 3:
    last_three = history[-3:]
    all_negative = all(e in negative for e in last_three)

    if all_negative and not st.session_state.show_breathing:
        st.markdown("---")
        st.write("💜 You've been carrying something heavy for a while. Would a short breathing exercise help?")
        if st.button("🫁 Try breathing exercise", key="breathing_btn"):
            st.session_state.show_breathing = True



if len(history) >= 5:
    last_five = history[-5:]
    still_negative = all(e in negative for e in last_five)
    
    if still_negative and not st.session_state.show_grounding:
        st.markdown("---")
        st.write("🌱 You've been going through a lot. Want to try a quick grounding exercise to help you feel more present?")
        if st.button("🌱 Try grounding exercise", key="grounding_btn"):
            st.session_state.show_grounding = True



if st.session_state.show_breathing:
    st.markdown("---")
    st.subheader("🫁 4-7-8 Breathing Exercise")
    st.write("This will help calm your nervous system.")
    st.markdown("""
    1. **Inhale** through your nose for **4 seconds**
    2. **Hold** your breath for **7 seconds**
    3. **Exhale** slowly through your mouth for **8 seconds**
    """)

    if st.button("Start", key="start_breathing"):
        with st.empty():
            for phase, duration, color in [
                ("🫁 Inhale...", 4, "#E3F2FD"),
                ("⏸ Hold...", 7, "#FFF3E0"),
                ("💨 Exhale...", 8, "#E8F5E9")
            ]:
                for i in range(duration, 0, -1):
                    st.markdown(f"""
                    <div style='background:{color}; padding:40px; border-radius:12px; text-align:center; font-size:24px;'>
                    {phase}<br><br>
                    <span style='font-size:48px; font-weight:bold;'>{i}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
        st.success("Well done. Take a moment to notice how you feel. 💜")
        st.session_state.show_breathing = False
        st.session_state.emotion_history = []
    

    
if st.session_state.show_grounding:
    st.markdown("---")
    st.subheader("🌱 Grounding Exercise — 5-4-3-2-1")
    st.write("This brings you back to the present moment. Take your time with each one.")
    st.markdown("""
    - **5** things you can **see** right now
    - **4** things you can **touch** right now  
    - **3** things you can **hear** right now
    - **2** things you can **smell** right now
    - **1** thing you can **taste** right now
    """)
    st.write("Go slowly. Really notice each one. 💜")
    if st.button("I'm done", key="grounding_done"):
        st.success("Good. You're here. You're okay. 💜")
        st.session_state.show_grounding = False
        st.session_state.emotion_history = []