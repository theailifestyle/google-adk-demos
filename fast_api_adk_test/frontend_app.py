import streamlit as st
import requests
import json
import uuid
from sseclient import SSEClient # For handling Server-Sent Events

# --- Configuration ---
BACKEND_URL = "http://localhost:8000/chat_sse" # URL of your FastAPI backend

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ADK FastAPI Test Frontend", layout="wide")
st.title("🗣️ ADK Agent via FastAPI - Test Frontend")
st.caption("Interacting with an ADK agent (using in-memory sessions) served by a FastAPI backend.")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history: {"role": "user/assistant", "content": "..."}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Unique session ID for this browser session
    st.sidebar.info(f"New client session started: {st.session_state.session_id}")

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Client Session ID:** `{st.session_state.session_id}`")
if st.sidebar.button("Start New Chat Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.sidebar.info(f"New client session started: {st.session_state.session_id}")
    st.rerun()


# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) or isinstance(message["content"], list):
            st.json(message["content"])
        else:
            st.markdown(message["content"], unsafe_allow_html=True)

# --- Handle User Input ---
if prompt := st.chat_input("Ask the agent..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request to the FastAPI backend
    payload = {
        "query": prompt,
        "user_id": "streamlit_test_user", # Can be dynamic if needed
        "session_id": st.session_state.session_id
    }

    # Display assistant's chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for the streaming response
        full_response_parts = []
        raw_events_displayed = False

        try:
            # Use requests.post with stream=True and SSEClient
            response = requests.post(BACKEND_URL, json=payload, stream=True, timeout=120)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # The SSEClient usage has been problematic, so it's commented out.
            # We will rely on manual SSE parsing.
            # client = SSEClient(response) 
            # st.write("Receiving events from backend...")

            # final_agent_response_text = "" # Initialized before manual processing block

            # try:
            #     for event in client: # This is where the error occurs
            #         if not event.data:
            #             continue
            #     # Add processing logic for event.event, event.id, event.data, event.retry
            #     # For ADK, event.data will be a JSON string.
            #     # Example:
            #     # if event.data:
            #     #     try:
            #     #         event_payload = json.loads(event.data)
            #     #         # Process event_payload based on ADK structure
            #     #         # Update message_placeholder and final_agent_response_text
            #     #     except json.JSONDecodeError:
            #     #         st.warning(f"Received non-JSON SSE data: {event.data}")
            # except Exception as iter_ex:
            #     st.error(f"Error iterating over SSEClient: {iter_ex}")
            #     if final_agent_response_text == "": 
            #         final_agent_response_text = f"Error during event stream processing: {iter_ex}"
            #     message_placeholder.markdown(final_agent_response_text if final_agent_response_text else "Stream processing error.", unsafe_allow_html=True)

            # Manual SSE processing
            st.write("Receiving events from backend...") # General message
            final_agent_response_text = ""
            event_buffer = []

            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data:'):
                    event_buffer.append(line[len('data:'):].strip())
                elif not line and event_buffer: # Empty line signifies end of an event
                    event_data_str = "".join(event_buffer)
                    event_buffer = [] # Reset buffer for next event
                    
                    if not event_data_str:
                        continue

                    try:
                        event_data = json.loads(event_data_str)
                        
                        if not raw_events_displayed:
                            with st.expander("Raw Backend Events (Live Stream)", expanded=False):
                                st.session_state.raw_events_container = st.container()
                            raw_events_displayed = True
                        
                        if 'raw_events_container' in st.session_state:
                            st.session_state.raw_events_container.json(event_data)

                        # Process ADK event structure
                        content = event_data.get("content")
                        author = event_data.get("author", "agent") # Get author for display
                        error_msg = event_data.get("error_message") # Check for top-level error

                        if error_msg:
                            final_agent_response_text = f"**[Error from {author}]** {error_msg}"
                            message_placeholder.markdown(final_agent_response_text, unsafe_allow_html=True)
                            break # Stop processing further events on error
                        
                        if content and content.get("parts"):
                            part = content["parts"][0]
                            # Check for text in the part, typically from the "model" role
                            if part.get("text") and content.get("role") == "model":
                                current_text = part["text"].strip()
                                if current_text: # Ensure there's actual text
                                    final_agent_response_text = current_text # Overwrite with the latest model text
                                    # Update the placeholder with the latest model text, prepended by author
                                    message_placeholder.markdown(f"**[{author}]** {final_agent_response_text}", unsafe_allow_html=True)
                            
                            # (Optional) Handle other parts like function calls/responses if needed for display,
                            # but typically the final text comes from the model role.
                        
                    except json.JSONDecodeError:
                        st.warning(f"Received non-JSON event data string: {event_data_str}")
                    except Exception as e:
                        st.error(f"Error processing manually parsed event: {e} - Data: {event_data_str}")
                        final_agent_response_text = f"Frontend error processing event: {e}"
                        message_placeholder.markdown(final_agent_response_text, unsafe_allow_html=True)
                        break
                # Other SSE fields like 'event:', 'id:', 'retry:' could be handled here if needed
            
            if not final_agent_response_text and not st.session_state.messages[-1]["role"] == "assistant": # Check if an error already logged
                 # If loop finishes and no final_agent_response_text, means stream ended or was empty.
                if not any(msg["role"] == "assistant" and "Error" in msg["content"] for msg in st.session_state.messages[-2:]): # Avoid double error
                    final_agent_response_text = "Agent interaction complete. (Stream ended or no specific final text captured)"
                    message_placeholder.markdown(final_agent_response_text, unsafe_allow_html=True)


            # Add the final assistant response to chat history if one was determined
            if final_agent_response_text:
                 st.session_state.messages.append({"role": "assistant", "content": final_agent_response_text})
            elif not any(msg["role"] == "assistant" for msg in st.session_state.messages[-1:]): # if no assistant msg yet for this turn
                 st.session_state.messages.append({"role": "assistant", "content": "No specific response captured from agent."})


        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: Could not connect to backend at {BACKEND_URL}. Ensure backend is running. Details: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error connecting to backend: {e}"})
        except Exception as e: # Catch any other unexpected error during the setup or request
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Unexpected error: {e}"})

# Instructions for running
st.sidebar.markdown("---")
st.sidebar.header("How to Run This Test")
st.sidebar.markdown("""
1.  **Set API Keys:** Ensure `GOOGLE_API_KEY` (and optionally `OPENAI_API_KEY`) are set as environment variables before starting the backend.
    ```bash
    export GOOGLE_API_KEY="your_google_api_key"
    # export OPENAI_API_KEY="your_openai_api_key"
    ```
2.  **Start the Backend:**
    Navigate to the `fast_api_adk_test` directory and run:
    ```bash
    python backend_app.py
    ```
    The backend should start on `http://localhost:8000`. (Note: Currently configured for in-memory sessions, so no database file will be created by default).
3.  **Run this Frontend:**
    In a new terminal, navigate to `fast_api_adk_test` and run:
    ```bash
    streamlit run frontend_app.py
    ```
4.  **Test:** Use the chat interface. Observe the backend logs and the "Raw Backend Events" expander here for details. Try queries that involve different agents (math, Spanish, search).
5.  **Check for Errors:** Monitor backend logs for any operational errors. (Note: The `UNIQUE constraint failed` error is specific to SQLite and won't occur with the current in-memory setup).
""")
