from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import streamlit as st
from datetime import datetime
import dateparser
import re

# Load environment variables
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize session state
if 'flowmessage' not in st.session_state:
    st.session_state['flowmessage'] = [
        SystemMessage(content="You are an AI assistant. When the user asks to be called, "
                              "ask them for their name, phone number, and email in a friendly manner.")
    ]
if 'waiting_for' not in st.session_state:
    st.session_state['waiting_for'] = None  # Tracks the expected information (name, phone, email)

# Function to truncate messages to fit within token limits
def truncate_messages(messages, token_limit=30000):
    """Truncate messages to ensure total token count is within the limit."""
    total_tokens = 0
    truncated_messages = []
    for msg in reversed(messages):  # Start from the most recent message
        msg_tokens = len(msg.content.split())  # Approximation using word count
        if total_tokens + msg_tokens > token_limit:
            break
        total_tokens += msg_tokens
        truncated_messages.append(msg)
    return list(reversed(truncated_messages))  # Reverse to preserve order

# Format messages for input to the AI
def format_messages(messages):
    formatted = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "Human"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# Function to validate email and phone number
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

def validate_phone(phone):
    pattern = r'^\+?\d{10,15}$'  # Accepts phone numbers with or without "+" and 10-15 digits
    return re.match(pattern, phone)

# Function to extract date
def extract_date(user_input):
    parsed_date = dateparser.parse(user_input)
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d")
    return None

# Function to handle AI response
def get_response(question):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    chat = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, token=huggingface_api_key)

    # Add user question to chat history
    st.session_state['flowmessage'].append(HumanMessage(content=question))

    # Truncate the chat history before sending
    st.session_state['flowmessage'] = truncate_messages(st.session_state['flowmessage'])

    # Format the chat history into a single string
    prompt = format_messages(st.session_state['flowmessage'])

    # Generate AI response
    answer = chat(prompt, max_new_tokens=256)  # Reduced max_new_tokens to 256

    # Add AI's response to chat history
    st.session_state['flowmessage'].append(AIMessage(content=answer))

    return answer

# Streamlit application
st.title("Conversational Form AI")

# Input field for user questions
user_input = st.text_input("Enter your message: ", key="input")

# Submit button
if st.button("Send"):
    if user_input.strip():
        # Check if the user initiated a "call me" request
        if "call me" in user_input.lower() or "contact me" in user_input.lower():
            st.session_state['waiting_for'] = "name"  # Start by asking for the name
            response = "Sure! Can I have your name, please?"
            st.session_state['flowmessage'].append(AIMessage(content=response))
            st.subheader("AI Response:")
            st.write(response)
        elif st.session_state['waiting_for'] == "name":
            # Validate and ask for the phone number
            st.session_state['waiting_for'] = "phone"
            response = f"Thank you, {user_input}! Can I have your phone number?"
            st.session_state['flowmessage'].append(AIMessage(content=response))
            st.subheader("AI Response:")
            st.write(response)
        elif st.session_state['waiting_for'] == "phone":
            if validate_phone(user_input):
                # Ask for the email
                st.session_state['waiting_for'] = "email"
                response = "Thanks! Lastly, could you provide your email address?"
                st.session_state['flowmessage'].append(AIMessage(content=response))
                st.subheader("AI Response:")
                st.write(response)
            else:
                st.warning("Please provide a valid phone number.")
        elif st.session_state['waiting_for'] == "email":
            if validate_email(user_input):
                # End conversation
                st.session_state['waiting_for'] = None
                response = "Great! We'll contact you shortly. Have a nice day!"
                st.session_state['flowmessage'].append(AIMessage(content=response))
                st.subheader("AI Response:")
                st.write(response)
            else:
                st.warning("Please provide a valid email address.")
        else:
            # Continue normal conversation
            response = get_response(user_input)
            st.subheader("AI Response:")
            st.write(response)
    else:
        st.warning("Please enter a message before submitting.")
