import streamlit as st
import os
from dotenv import load_dotenv
import traceback # Import traceback for detailed error printing
# --- Attempt to import ADK components ---
try:
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    # Tool import kept here, but tool removed from debug agent below
    from google.adk.tools import built_in_code_execution
    ADK_AVAILABLE = True
except ImportError as e:
    st.error(f"Required Google ADK libraries not found. Error: {e}. Please ensure 'google.adk' is installed or accessible.")
    ADK_AVAILABLE = False
    # Define dummy classes/functions
    class DummyAgent:
        def __init__(self, *args, **kwargs): pass
    class DummySessionService:
        def create_session(self, *args, **kwargs): return type('obj', (object,), {'state': {}})()
        def get_session(self, *args, **kwargs): return type('obj', (object,), {'state': {}})()
    class DummyRunner:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs):
            st.warning("ADK Runner is unavailable.")
            yield type('obj', (object,), {'is_final_response': lambda: True, 'content': type('obj', (object,), {'parts': [type('obj', (object,), {'text': 'ADK Libraries unavailable.'})()]})()})()
    built_in_code_execution = None # Dummy tool placeholder
    SequentialAgent, LlmAgent, InMemorySessionService, Runner = DummyAgent, DummyAgent, DummySessionService, DummyRunner

# --- Attempt to import GenAI types ---
try:
    from google.genai import types as genai_types
    GENAI_TYPES_AVAILABLE = True
except ImportError:
    st.warning("google-generativeai library not found.")
    GENAI_TYPES_AVAILABLE = False
    genai_types = type('obj', (object,), {'Content': lambda **kwargs: type('obj', (object,), kwargs)(), 'Part': lambda **kwargs: type('obj', (object,), kwargs)()})

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
APP_NAME = "streamlit_code_pipeline_generator" # Changed name slightly
USER_ID = "streamlit_user_01"
SESSION_ID_PREFIX = "pipeline_session_"
GEMINI_MODEL = "gemini-2.0-flash"

# --- Agent Definitions and Initialization (DEBUGGING VERSION) ---
# --- Agents are initialized only ONCE per Streamlit session using st.session_state ---
if ADK_AVAILABLE:

    # Code Writer Agent
    if 'code_writer_agent' not in st.session_state:
        st.session_state.code_writer_agent = LlmAgent(
            name="CodeWriterAgent", model=GEMINI_MODEL,
            instruction="Write Python code based on user request. Output only raw code in ```python ... ```.",
            description="Writes initial code.", output_key="generated_code"
        )

    # --- DEBUGGING VERSIONS of subsequent agents ---
    # Code Reviewer Agent (DEBUG)
    if 'code_reviewer_agent' not in st.session_state:
        st.session_state.code_reviewer_agent = LlmAgent(
            name="CodeReviewerAgent",
            model=GEMINI_MODEL,
            # --- Use the original, full instruction ---
            instruction="""You are a Code Reviewer AI.
    Review the Python code provided in the session state under the key 'generated_code'.
    Provide constructive feedback as bullet points (*). Focus on:
    * Potential bugs or errors.
    * Adherence to Python best practices (PEP 8).
    * Possible improvements for clarity, efficiency, or robustness.
    * Missing error handling or edge cases.
    Output only the review comments. Do not include the code itself in your output.
    """,
            description="Reviews code and provides feedback.",
            output_key="review_comments" # Write to the correct key
        )


    # Code Refactorer Agent (DEBUG)
    if 'code_refactorer_agent' not in st.session_state:
        st.session_state.code_refactorer_agent = LlmAgent(
            name="CodeRefactorerAgent",
            model=GEMINI_MODEL,
            # --- Use the original, full instruction ---
            instruction="""You are a Code Refactorer AI.
    Take the original Python code provided in the session state key 'generated_code'
    and the review comments found in the session state key 'review_comments'.
    Refactor the original code *strictly* based on the provided review comments to improve its quality, clarity, and correctness.
    If the review comments are empty or non-actionable, return the original code.
    Output *only* the final, refactored Python code block, enclosed in triple backticks (```python ... ```).
    """,
            description="Refactors code based on review comments.",
            output_key="refactored_code" # Write to the correct key
        )

    # Code Interpreter Agent (DEBUG - Tool Removed)
    if 'code_interpreter_agent' not in st.session_state:
        st.session_state.code_interpreter_agent = LlmAgent(
            name="CodeInterpreterAgent",
            model=GEMINI_MODEL,
            # --- Add the tool back ---
            tools=[built_in_code_execution],
            # --- Keep the original, full instruction ---
            instruction="""You are a Code Execution Assistant.
    1. Examine the session state for Python code, prioritizing the key 'refactored_code'. If it's empty or absent, use the code from 'generated_code'.
    2. Extract *only* the raw Python code from the relevant state key (remove markdown fences like ```python).
    3. If code is found, execute it using the provided code execution tool.
    - For code defining functions/classes without direct execution, add simple example usage if feasible (e.g., call a function with sample inputs) to test its execution. Run scripts directly.
    4. Your final output *must* be only a plain text summary detailing the execution outcome. Format:
    Execution Outcome: [Success/Failure]
    Output:
    [Captured stdout/stderr or 'No output captured.']""",
            description="Executes the generated/refactored code and reports the outcome.",
            output_key="execution_summary" # Write to the correct key
        )

    # --- Initialize Sequential Agent only ONCE ---
    if 'code_pipeline_agent' not in st.session_state:
        st.session_state.code_pipeline_agent = SequentialAgent(
            name="CodePipelineAgent_Debug", # Added Debug to name
            sub_agents=[
                # Reference agents FROM st.session_state
                st.session_state.code_writer_agent,
                st.session_state.code_reviewer_agent, # Uses DEBUG version
                st.session_state.code_refactorer_agent, # Uses DEBUG version
                st.session_state.code_interpreter_agent # Uses DEBUG version
            ]
        )

    # --- Session Service Initialization (Only Once) ---
    if 'session_service' not in st.session_state:
        st.session_state.session_service = InMemorySessionService()

    # --- Runner Initialization (Only Once) ---
    if 'runner' not in st.session_state:
        st.session_state.runner = Runner(
            agent=st.session_state.code_pipeline_agent, # Use pipeline agent FROM state
            app_name=APP_NAME,
            session_service=st.session_state.session_service
        )

# --- Helper Function (Keep as is) ---
def clean_code_output(text):
    if text is None: return ""
    text = text.strip()
    if text.startswith("```python"): text = text[len("```python"):].strip()
    elif text.startswith("```"): text = text[len("```"):].strip()
    if text.endswith("```"): text = text[:-len("```")].strip()
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="AI Code Pipeline (Debug)", layout="wide")
st.title("🚧 AI Code Generation Pipeline 🚧")
st.markdown("Enter a description -> Write Code -> Review Code -> Refactor Code -> Execute Code (Code Interpreter)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Your Request")
    user_query = st.text_area("Describe the Python code you want:", height=100, placeholder="e.g., print hello sam in python")

    session_counter = st.session_state.get('session_counter', 0)
    current_session_id = f"{SESSION_ID_PREFIX}{session_counter}"

    if st.button("⚙️ Run Code Generator Pipeline", disabled=not ADK_AVAILABLE):
        if user_query and ADK_AVAILABLE and GENAI_TYPES_AVAILABLE:
            st.session_state.session_counter = session_counter + 1
            st.session_state.results = {}
            st.session_state.error = None
            st.session_state.ran_query = user_query

            try:
                # Get session_service and runner from Streamlit's session state
                session_service = st.session_state.session_service
                runner = st.session_state.runner

                # Create a new ADK session for this specific run
                session = session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id
                )

                # Prepare the initial message
                initial_content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_query)])

                # Run the pipeline and capture events for detailed logging
                with st.spinner("🤖 Running simplified debug sequence..."):
                    events = runner.run(
                        user_id=USER_ID, session_id=current_session_id, new_message=initial_content
                    )

                    # --- DETAILED EVENT LOGGING ---
                    st.write("--- Processing Events ---")
                    event_list_for_debug = []
                    final_response_text = "Pipeline completed." # Default message
                    for i, event in enumerate(events):
                        # Safely get attributes, provide default if missing
                        event_details = {
                            "index": i,
                            "id": getattr(event, 'id', 'N/A'),
                            "author": getattr(event, 'author', 'N/A'),
                            "is_final": event.is_final_response(),
                            "interrupted": getattr(event, 'interrupted', None),
                            "error_code": getattr(event, 'error_code', None),
                            "error_message": getattr(event, 'error_message', None),
                            "content_parts": []
                        }
                        # Check content and parts carefully
                        if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                             for part in event.content.parts:
                                part_info = {}
                                # Extract text content if present
                                if hasattr(part, 'text') and part.text: part_info['text'] = part.text
                                # Safely convert other known types to string for logging
                                if hasattr(part, 'executable_code'): part_info['executable_code'] = str(part.executable_code)
                                if hasattr(part, 'code_execution_result'): part_info['code_execution_result'] = str(part.code_execution_result)
                                # Add other part types if needed and convert complex objects safely
                                if part_info: # Only add if we extracted something useful
                                     event_details["content_parts"].append(part_info)

                        event_list_for_debug.append(event_details) # Add structured details for this event

                        # Capture final response text from the last agent's message
                        if event.is_final_response():
                            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                                first_part = event.content.parts[0]
                                if hasattr(first_part, 'text') and first_part.text:
                                    final_response_text = first_part.text # This should be the output of the last (debug) agent

                    st.write("DEBUG: Detailed Events List:")
                    # Use st.json for potentially large/nested data, start collapsed
                    st.json(event_list_for_debug, expanded=False)
                    st.write("--- End Processing Events ---")
                    # --- END DETAILED EVENT LOGGING ---


                # --- Retrieve the final state ---
                updated_session = session_service.get_session(
                    session_id=current_session_id, app_name=APP_NAME, user_id=USER_ID
                )
                session_state = updated_session.state if updated_session and hasattr(updated_session, 'state') else {}

                # Store results in Streamlit's session state for display
                st.session_state.results = {
                    "generated_code": session_state.get("generated_code"),
                    "review_comments": session_state.get("review_comments"), # Should get debug text
                    "refactored_code": session_state.get("refactored_code"), # Should get debug text
                    "execution_summary": session_state.get("execution_summary"), # Should get debug text
                    "final_message": final_response_text # Capture the last agent's direct message
                }

                if session_state: st.success("Pipeline finished!")
                else: st.warning("Pipeline finished, but session state appears empty.")

            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
                # Keep this traceback for real errors
                st.code(traceback.format_exc())
                st.session_state.error = str(e)
                st.session_state.results = {} # Clear results on error

        elif not user_query: st.warning("Please enter a description for the code.")
        elif not ADK_AVAILABLE or not GENAI_TYPES_AVAILABLE: st.error("Cannot run pipeline: Required libraries missing.")

    if 'ran_query' in st.session_state and st.session_state.ran_query:
         st.markdown("**Last Run Request:**"); st.info(st.session_state.ran_query)

with col2:
    st.subheader("2. Agent Pipeline Results")

    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results

        # Step 1: Code Generation
        with st.expander("🖋️ Step 1: Initial Code Generation (CodeWriterAgent)", expanded=True):
            generated_code = clean_code_output(results.get("generated_code"))
            if generated_code: st.code(generated_code, language="python")
            else: st.warning("No code generated or retrieved.")

        # Step 2: Code Review
        with st.expander("🔍 Step 2: Code Review (Code Reviewer Agent)", expanded=True):
            review_comments = results.get("review_comments")
            st.text(f"State Output: {review_comments}" if review_comments else "State Output: None")

        # Step 3: Code Refactoring
        with st.expander("🔧 Step 3: Refactored Code (Code Refactorer Agent)", expanded=True):
            refactored_code = results.get("refactored_code")
            st.text(f"State Output: {refactored_code}" if refactored_code else "State Output: None")

        # Step 4: Code Execution
        with st.expander("🚀 Step 4: Code Execution (Code Interpreter Agent)", expanded=True):
            execution_summary = results.get("execution_summary")
            st.text(f"State Output: {execution_summary}" if execution_summary else "State Output: None")

        st.divider()
        st.subheader("🎉 Final Generated Code")
        final_code_to_display = clean_code_output(results.get("generated_code"))
        if final_code_to_display: st.code(final_code_to_display, language="python")
        else: st.error("Could not retrieve the generated code from session state.")

        # Added a footer for better user experience
        st.markdown("---")
        st.caption("Powered by Google ADK and Streamlit")

    elif 'error' in st.session_state and st.session_state.error:
        st.error(f"Pipeline execution failed.")
    elif not ADK_AVAILABLE:
         st.info("Enter a request and click 'Run Debug Pipeline' once ADK libraries are available.")
    else:
        st.info("Enter a request on the left to see the results here.")