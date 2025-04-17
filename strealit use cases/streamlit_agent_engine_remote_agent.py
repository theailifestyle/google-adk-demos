import streamlit as st
import vertexai
from vertexai import agent_engines
from google.api_core import exceptions as api_exceptions # For specific error handling
import traceback
import time

# --- Configuration ---
st.set_page_config(page_title="Vertex AI Translator", layout="wide")
st.title("🌍 Vertex AI Agent Engine Translator")
st.write("Interact with your deployed multi-language translation agent engine.")

# --- Helper Function to Get Client (Cached) ---
# --- Helper Function to Get Client (Cached) ---
# --- Helper Function to Get Client (Cached) ---
@st.cache_resource(show_spinner="Connecting to Agent Engine...")
def get_remote_app_client(project_id, location, agent_engine_id):
    """Initializes Vertex AI and gets the Agent Engine client."""
    # Use the project_id passed from the input for vertexai.init
    st.write(f"Attempting to initialize Vertex AI for project '{project_id}' in '{location}'...")
    try:
        vertexai.init(project=project_id, location=location)
        st.write("Vertex AI Initialized.")

        # --- FIX: Use 'reasoningEngines' and Project Number in resource name ---
        project_number = "463926261379" # From your deployment logs
        # Correct resource type based on deployment log
        resource_name = f"projects/{project_number}/locations/{location}/reasoningEngines/{agent_engine_id}"
        # --- END FIX ---

        st.write(f"Getting Agent Engine client for: {resource_name}")

        # Import and instantiate AgentEngine directly
        # NOTE: Even though the resource is reasoningEngines, the SDK class might still be AgentEngine
        from vertexai.agent_engines._agent_engines import AgentEngine
        remote_app = AgentEngine(resource_name=resource_name)

        st.success("Successfully connected to Agent Engine!")
        return remote_app
    except api_exceptions.NotFound:
        st.error(f"Reasoning Engine not found. Please check your Project ID/Number, Location, and Engine ID.\nResource name tried: {resource_name}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI or get Reasoning Engine client.", icon="🔥")
        st.exception(e) # Show the actual exception
        return None

# --- Helper Function to Query Agent ---
def query_agent_engine(client, session_id, user_id, query_message):
    """Sends a query to the agent engine and parses the response stream."""
    final_text_response = ""
    try:
        st.write(f"Streaming query to session '{session_id}': '{query_message}'")
        response_stream = client.stream_query(
            session_id=session_id,
            message=query_message,
            user_id=user_id
        )

        # Use the dictionary parsing logic confirmed to work
        for event in response_stream:
            current_text = ""
            try:
                if isinstance(event, dict) and 'content' in event and 'parts' in event['content'] and isinstance(event['content']['parts'], list) and len(event['content']['parts']) > 0:
                    first_part = event['content']['parts'][0]
                    if 'text' in first_part:
                        current_text = first_part.get('text', '')
                    elif 'function_response' in first_part:
                        function_response_data = first_part.get('function_response', {})
                        response_data = function_response_data.get('response', {})
                        current_text = response_data.get('result', '')
            except Exception as e:
                 st.warning(f"Warning: Error processing event structure - {e}. Event: {event}", icon="⚠️")

            if current_text:
                cleaned_text = current_text.strip()
                if cleaned_text:
                    final_text_response = cleaned_text

        st.write("Stream finished.")
        if not final_text_response:
             st.warning("Agent did not return a text response.", icon="❓")

        return final_text_response

    except Exception as e:
        st.error("An error occurred during the agent query.", icon="🔥")
        st.exception(e)
        return None


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Agent Engine Configuration")
    # Use details from your successful deployment output as defaults
    project_id = st.text_input("Google Cloud Project ID", value="the-ai-lifestyle")
    location = st.text_input("Agent Engine Location", value="us-central1")
    # Extract the ID from the resource name
    agent_engine_id = st.text_input(
        "Agent Engine ID",
        value="6672795331484188672", # From your log: reasoningEngines/ID
        help="Find this in the Cloud Console (Vertex AI -> Agent Engines) or deployment script output."
        )

    st.markdown("---")
    st.header("Session Info")
    if 'agent_session_id' not in st.session_state:
        st.session_state.agent_session_id = None
    st.write(f"Current Session ID: {st.session_state.agent_session_id}")
    if st.button("Clear Session ID"):
         st.session_state.agent_session_id = None
         st.rerun()

# --- Main App Area ---
remote_app_client = None
if project_id and location and agent_engine_id:
    # Attempt to get the client using the cached function
    remote_app_client = get_remote_app_client(project_id, location, agent_engine_id)
else:
    st.info("Please enter your Project ID, Location, and Agent Engine ID in the sidebar.")

if remote_app_client:
    # Attempt to create a session if we don't have one and the client is valid
    if not st.session_state.agent_session_id:
        try:
            with st.spinner("Creating agent session..."):
                 # Using a generic user ID for this demo app
                session_info = remote_app_client.create_session(user_id="streamlit_test_user")
                st.session_state.agent_session_id = session_info["id"]
            st.success(f"Created Session ID: {st.session_state.agent_session_id}")
            time.sleep(0.1) # Short delay might help state update before rerun
            st.rerun() # Rerun to update UI now that session ID exists
        except Exception as e:
            st.error("Failed to create agent session.", icon="❌")
            st.exception(e)
            # Clear the client cache if session creation fails, forcing re-connection attempt
            get_remote_app_client.clear()

    # Proceed only if we have a client AND a session ID
    if st.session_state.agent_session_id:
        st.header("Translate Text")
        text_to_translate = st.text_area("Enter text to translate:", height=100, key="text_input")
        target_language = st.selectbox(
            "Select target language:",
            ("Spanish", "French", "Old English"),
            key="lang_select"
        )

        if st.button(f"Translate to {target_language}", key="translate_button"):
            if text_to_translate and target_language:
                # Construct the query message
                query = f"Translate '{text_to_translate}' to {target_language}"

                with st.spinner(f"Asking Agent Engine to translate to {target_language}..."):
                    translation_result = query_agent_engine(
                        client=remote_app_client,
                        session_id=st.session_state.agent_session_id,
                        user_id="streamlit_test_user", # Use the same user ID
                        query_message=query
                    )

                if translation_result is not None:
                    st.subheader("Translation Result:")
                    # Use markdown potentially for better formatting if needed
                    st.markdown(f"**{translation_result}**")
                # Error messages are handled within query_agent_engine

            else:
                st.warning("Please enter text to translate.", icon="⚠️")
else:
     st.warning("Cannot proceed without a valid Agent Engine connection. Please check configuration in the sidebar.", icon="⚙️")

st.markdown("---")
st.caption(f"App Time: {time.strftime('%Y-%m-%d %H:%M:%S')}") # Show current time