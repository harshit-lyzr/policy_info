import streamlit as st
import os
from lyzr import QABot
import openai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Policy Information Chatbot",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Policy Information Chatbot")
st.markdown("### Welcome to the Lyzr Policy Information Chatbot!")
st.markdown("This app uses Lyzr QABot to assist with your policy and insurance-related queries. We have integrated comprehensive health insurance data to provide accurate and helpful information. Feel free to ask any questions related to health insurance.")
st.markdown("Suggested Question: ")
st.markdown("1) is this insurance coverage cover my disorders of lens?")
st.markdown("2) My 34% coverage period is gone how much refund i get if i cancel my insurance?")

prompt=f"""
You are an Health Insurance Policy Chatbot.
Task: Your Task Is to provide information about available health insurance plans, assist with policy inquiries,
coverage limits,terms and conditions, help with claims processing, or offer general healthcare advice from given Document.
Context : 
1/ Collect comprehensive information about the health insurance policies from the document.Ensure that the given Answer complies with relevant healthcare regulations and data privacy laws, such as HIPAA (Health Insurance Portability and Accountability Act) in the United States. 
2/ Implement measures to protect sensitive user information and provide clear disclosures about data handling practices.
3/ Understand the needs and preferences of User.

Review:
The Answer is retrieved from document only.
"""

@st.cache_resource
def rag_implementation():
    with st.spinner("Generating Embeddings...."):
        qa = QABot.pdf_qa(
            input_files=["insurance.pdf"],
            system_prompt=prompt

        )
    return qa


st.session_state["chatbot"] = rag_implementation()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "chatbot" in st.session_state:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state["chatbot"].query(prompt)
            chat_response = response.response
            response = st.write(chat_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": chat_response}
        )
else:
    with st.sidebar.expander("ℹ️ - About this App"):
        st.markdown(
            """
        This app uses Lyzr Core to generate notes from transcribed audio. The audio transcription is powered by OpenAI's Whisper model. For any inquiries or issues, please contact Lyzr.

        """
        )
        st.link_button("Lyzr", url="https://www.lyzr.ai/", use_container_width=True)
        st.link_button(
            "Book a Demo", url="https://www.lyzr.ai/book-demo/", use_container_width=True
        )
        st.link_button(
            "Discord", url="https://discord.gg/nm7zSyEFA2", use_container_width=True
        )
        st.link_button(
            "Slack",
            url="https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw",
            use_container_width=True,
        )