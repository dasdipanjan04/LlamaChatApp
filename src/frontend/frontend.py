import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8000/query"

st.title("Chat with Llama")

st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
    }
    .model-bubble {
        background-color: #F1F0F0;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state["history"] = []
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""


def format_text(text):
    return re.sub(r"([.,!?])(?=\S)", r"\1 ", text).strip()


def display_chat_message(speaker, message, is_user=False):
    bubble_class = "user-bubble" if is_user else "model-bubble"
    st.markdown(
        f"""
        <div class="chat-container">
            <div class="{bubble_class}">{message}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def stream_response(user_input):
    response_text = ""
    try:
        response = requests.post(
            API_URL,
            json={"queries": [user_input]},
            stream=True,
        )

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                token = chunk.decode("utf-8").strip()
                response_text += format_text(token) + " "
                yield response_text.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")


user_input = st.text_input(
    "You:",
    value=st.session_state["input_text"],
    key="input",
    placeholder="Type your message here...",
)

if st.button("Send") and user_input.strip():
    st.session_state["history"].append(("You", user_input))
    st.session_state["input_text"] = ""

    response_placeholder = st.empty()
    response_text = ""

    with st.spinner("Generating response..."):
        for response in stream_response(user_input):
            response_placeholder.markdown("")
            response_placeholder.markdown(
                f"""
                <div class="chat-container">
                    <div class="model-bubble">{response}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
            response_text = response

    st.session_state["history"].append(("Model", response_text.strip()))

for speaker, text in st.session_state["history"]:
    display_chat_message(speaker, text, is_user=(speaker == "You"))
