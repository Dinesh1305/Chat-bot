import streamlit as st
import httpx  # Used for making async HTTP requests
import json
import uuid

# Configuration
FASTAPI_URL = "http://localhost:8000/chat/stream"

st.set_page_config(page_title="LangGraph-FastAPI Chatbot")
st.title("LangGraph Agent with Streamlit & FastAPI")

# Initialize Chat History and Thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.info(f"New conversation started (ID: {st.session_state.thread_id[:8]})")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask the agent anything..."):
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare the Streamlit response container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 3. Prepare data for FastAPI
        data = {"user_query": prompt, "thread_id": st.session_state.thread_id}

        try:
            # 4. Make an asynchronous streaming request using httpx
            with httpx.stream("POST", FASTAPI_URL, json=data, timeout=None) as response:
                if response.status_code != 200:
                    st.error(f"Error from API: Status {response.status_code}. Check backend console.")

                    # 🛑 CORRECTED: Replaced 'return' with 'st.stop()' to halt script execution
                    st.stop()

                for line in response.iter_lines():
                    if line.startswith("data:"):
                        # Extract the JSON data part of the SSE event
                        try:
                            json_data = line[len("data:"):].strip()
                            if json_data == "[DONE]":
                                break

                            event = json.loads(json_data)

                            # Concatenate and update the UI
                            if 'content' in event:
                                full_response += event['content']
                                message_placeholder.markdown(full_response + "▌")  # Add blinking cursor

                        except json.JSONDecodeError:
                            continue

            # Final cleanup
            message_placeholder.markdown(full_response)

            # 5. Add the final full response to the session history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except httpx.ConnectError:
            st.error(
                "Connection Error: Could not connect to the FastAPI backend. Ensure the backend server is running at http://localhost:8000.")
            st.session_state.messages.pop()