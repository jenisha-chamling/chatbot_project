from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize session state
if 'flowmessage' not in st.session_state:
    st.session_state['flowmessage'] = [
        SystemMessage(content='You are an AI assistant.')
    ]

def format_messages(messages):
    """Convert list of messages to a single string for the AI model."""
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



# Usage in the get_response function
def get_response(question):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    chat = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, token=huggingface_api_key)

    # Add the user's question to the chat history
    st.session_state['flowmessage'].append(HumanMessage(content=question))

    # Format the chat history into a single string for the model
    prompt = format_messages(st.session_state['flowmessage'])

    # Generate AI response
    answer = chat(prompt)  # This returns a plain string

    # Append the AI's response to the chat history
    st.session_state['flowmessage'].append(AIMessage(content=answer))

    # Return the plain string response (no .content needed)
    return answer




# Streamlit application
# Input field for user questions
user_input = st.text_input("Ask your question: ", key="input")

# Submit button
if st.button("Ask"):
    if user_input.strip():
        response = get_response(user_input)  # Get the AI's response
        st.subheader("AI Response:")
        st.write(response)  # Display only the AI's response
    else:
        st.warning("Please enter a question before submitting.")

