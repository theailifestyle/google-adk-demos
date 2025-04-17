import streamlit as st
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search # Assuming google_search tool is available and configured
from google.genai import types
import google.generativeai as genai # Import GenAI library
import traceback
import uuid
import os
import json
from dotenv import load_dotenv
load_dotenv()
import re # For parsing subtopics

# Page setup
st.set_page_config(page_title="Parallel Research Agent Demo", layout="wide")
# Added icon to title
st.title("🧠 AI Parallel Research Assistant")
st.markdown("*Enter a main topic, generate subtopics, and let parallel agents research and synthesize the findings.*")

# --- Constants ---
APP_NAME = "parallel_research_app_streamlit_subtopics"
USER_ID = "research_user_streamlit_01"
# Using the model name confirmed by the user
GEMINI_MODEL = "gemini-2.0-flash"
# Model for subtopic generation (can be the same or different)
SUBTOPIC_GEN_MODEL = "gemini-2.0-flash" # Use the confirmed working model

# --- State Keys ---
STATE_FINAL_REPORT = "final_report"
# Dynamic state keys will be generated for topics

# --- API Key Configuration ---
# Ensure the GOOGLE_API_KEY environment variable is set
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    # Removed sidebar success message
except KeyError:
    st.error("ERROR: GOOGLE_API_KEY environment variable not set.")
    st.stop() # Stop execution if key is missing
except Exception as e:
    st.error(f"Error configuring Generative AI: {e}")
    st.stop()


# Initialize session state for Streamlit
if 'research_results' not in st.session_state:
    st.session_state.research_results = None # Will store {topic: result}
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
# Renamed state key for clarity
if 'subtopics_list' not in st.session_state:
     # Stores the 3 subtopics shown in the input boxes
     st.session_state.subtopics_list = ["", "", ""]
if 'main_topic' not in st.session_state:
     st.session_state.main_topic = "Artificial Intelligence in Healthcare" # Default main topic


# --- Function to Generate Subtopics ---
def generate_subtopics(main_topic):
    """Calls the Gemini model to generate 3 subtopics for a main topic."""
    if not main_topic or not main_topic.strip():
        return ["", "", ""] # Return empty if no main topic

    prompt = f"""Given the main research topic "{main_topic}", generate exactly 3 distinct, relevant, and concise subtopics suitable for parallel web research.

    Output ONLY the 3 subtopics, each on a new line, formatted like this:
    1. Subtopic One
    2. Subtopic Two
    3. Subtopic Three
    """
    try:
        model = genai.GenerativeModel(SUBTOPIC_GEN_MODEL)
        response = model.generate_content(prompt)

        # Basic parsing based on numbered list format
        subtopics = ["", "", ""]
        lines = response.text.strip().split('\n')
        count = 0
        for line in lines:
            # Try to extract text after "1.", "2.", "3."
            match = re.match(r"^\d+\.\s*(.*)", line.strip())
            if match and count < 3:
                subtopics[count] = match.group(1).strip()
                count += 1
        # If parsing failed, try splitting by newline as a fallback (less reliable)
        if count < 3 and len(lines) >= 3:
             subtopics = [line.strip() for line in lines[:3]] # Take first 3 lines

        # Ensure we return exactly 3, padding if necessary
        while len(subtopics) < 3:
            subtopics.append("")
        return subtopics[:3]

    except Exception as e:
        st.error(f"Error generating subtopics: {e}")
        return ["", "", ""] # Return empty on error


# --- ADK Execution Function ---
# Accepts list of subtopics
def run_parallel_research(subtopics_list):
    """
    Sets up and runs the ParallelAgent for research on custom topics and a subsequent
    agent to synthesize the results. Uses dynamically created agents and state keys.
    """
    debug_logs = [] # For optional debug output
    research_summaries = {} # Store as {topic_name: summary}
    final_report = "Report generation failed."
    # Ensure topics_list contains only non-empty strings
    topics = [topic for topic in subtopics_list if isinstance(topic, str) and topic.strip()]

    if not topics:
        st.warning("No valid subtopics provided for research.")
        return {}, "No subtopics provided.", []

    # Create a unique session ID for each run
    session_id = f"session_{uuid.uuid4()}"
    debug_logs.append(f"Created unique session ID: {session_id}")

    try:
        # --- Session Service ---
        session_service = InMemorySessionService()
        session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        debug_logs.append(f"ADK Session created: {session_id}. State is initially empty.")

        # --- Dynamically Define Researcher Sub-Agents ---
        sub_agents_list = []
        state_keys_list = []
        topic_key_map = {} # Map state key back to original topic name

        for i, topic in enumerate(topics):
            # Sanitize topic slightly for name/key generation if needed
            safe_topic_part = "".join(c if c.isalnum() else "_" for c in topic[:20])
            agent_name = f"Researcher_{safe_topic_part}_{i+1}"
            state_key = f"topic_{i+1}_result"
            debug_logs.append(f"Defining agent '{agent_name}' for topic '{topic}' with state key '{state_key}'")

            # Using the simplified instruction format that worked previously
            researcher_agent = LlmAgent(
                name=agent_name,
                model=GEMINI_MODEL, # Use ADK model here
                instruction=f"""You are an AI Research Assistant.
                Research the latest advancements in '{topic}'.
                Use the Google Search tool provided.
                Summarize your key findings concisely (1-2 sentences).
                Output *only* the summary.
                """,
                description=f"Researches {topic}.",
                tools=[google_search],
                output_key=state_key # Use dynamic state key
            )
            sub_agents_list.append(researcher_agent)
            state_keys_list.append(state_key)
            topic_key_map[state_key] = topic # Store mapping

        debug_logs.append(f"{len(sub_agents_list)} researcher agents defined.")

        # --- Create the ParallelAgent ---
        if not sub_agents_list:
             raise ValueError("No valid topics provided to create research agents.")

        parallel_research_agent = ParallelAgent(
            name="ParallelWebResearchAgent",
            sub_agents=sub_agents_list # Use dynamically created list
        )
        debug_logs.append("ParallelWebResearchAgent defined.")

        # --- Define the Report Synthesizer Agent ---
        keys_for_prompt = ", ".join([f"'{key}'" for key in state_keys_list])
        topic_names_for_prompt = ", ".join([f"'{topic_key_map[key]}'" for key in state_keys_list])

        report_agent = LlmAgent(
            name="ReportSynthesizer",
            model=GEMINI_MODEL, # Use ADK model here
            instruction=f"""You are an AI Report Writer.
            Your task is to synthesize the research findings provided in the session state.
            The findings for the subtopics {topic_names_for_prompt} are stored under the keys {keys_for_prompt} respectively.

            Combine these findings into a single, coherent summary report (2-3 paragraphs minimum).
            Structure the report logically. Start by listing the subtopics covered ({topic_names_for_prompt}).
            Highlight key trends or connections between the subtopics if possible. Assume they relate to a broader common theme.
            Ensure the language is clear and professional.

            Output *only* the final report. Do not include introductory or concluding remarks like "Here is the report:".
            """,
            description="Synthesizes research findings into a final report.",
            output_key=STATE_FINAL_REPORT
        )
        debug_logs.append(f"ReportSynthesizer Agent defined, will read keys: {keys_for_prompt} for topics: {topic_names_for_prompt}")


        # --- Run the Parallel Research ---
        runner_parallel = Runner(agent=parallel_research_agent, app_name=APP_NAME, session_service=session_service)
        debug_logs.append("Runner created for ParallelAgent.")

        # Generic trigger message
        parallel_trigger_message = "Start parallel research based on your agent instructions."
        content_parallel = types.Content(role='user', parts=[types.Part(text=parallel_trigger_message)])
        debug_logs.append(f"Running ParallelAgent with trigger: '{parallel_trigger_message}'")

        parallel_events = runner_parallel.run(user_id=USER_ID, session_id=session_id, new_message=content_parallel)

        # Process events mainly to wait for completion
        event_count_p = 0
        for event in parallel_events:
            event_count_p += 1
            debug_logs.append(f"Parallel Event {event_count_p} processed.")

        debug_logs.append(f"ParallelAgent run completed after {event_count_p} events.")

        # --- Retrieve Intermediate Results ---
        session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        debug_logs.append("Fetched session state after parallel run.")

        # Use the dynamic keys and map to retrieve results, storing by actual topic name
        for state_key in state_keys_list:
            topic_name = topic_key_map.get(state_key, state_key) # Get original topic name
            summary = session.state.get(state_key, "No result found.")
            research_summaries[topic_name] = summary # Store with topic name as key

        debug_logs.append(f"Retrieved intermediate results: {research_summaries}")

        # --- Run the Report Synthesizer ---
        # Only run if there are summaries to report on
        if not any(s != "No result found." for s in research_summaries.values()):
             final_report = "Could not generate report as no individual research summaries were found."
             debug_logs.append("Skipping report generation as no summaries were found.")
        else:
            runner_report = Runner(agent=report_agent, app_name=APP_NAME, session_service=session_service)
            debug_logs.append("Runner created for ReportAgent.")

            report_trigger_message = "Generate the final report based on the research findings in the session state."
            content_report = types.Content(role='user', parts=[types.Part(text=report_trigger_message)])
            debug_logs.append(f"Running ReportAgent with trigger: '{report_trigger_message}'")

            report_events = runner_report.run(user_id=USER_ID, session_id=session_id, new_message=content_report)

            # Process events to wait for completion and capture final response text
            event_count_r = 0
            final_report_text_temp = None
            for event in report_events:
                event_count_r += 1
                debug_logs.append(f"Report Event {event_count_r} processed.")
                # Capture final response text if possible
                if hasattr(event, 'is_final_response') and event.is_final_response():
                     if hasattr(event, 'content') and hasattr(event.content, 'parts') and event.content.parts:
                         final_report_text_temp = event.content.parts[0].text
                         debug_logs.append(f"Report final response detected: {final_report_text_temp[:100]}...")

            debug_logs.append(f"ReportAgent run completed after {event_count_r} events.")

            # --- Retrieve Final Report ---
            session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            debug_logs.append("Fetched session state after report run.")
            final_report = session.state.get(STATE_FINAL_REPORT, "Report not found in state.")
            # Fallback if state retrieval fails but event capture worked
            if final_report == "Report not found in state." and final_report_text_temp:
                 final_report = final_report_text_temp
                 debug_logs.append("Using final report text captured from event.")


    except Exception as e:
        st.error(f"An error occurred during the research process: {str(e)}")
        st.code(f"Full error: {traceback.format_exc()}", language="python")
        # Ensure partial results are still returned if available
        return research_summaries, final_report, debug_logs

    # --- Display Debug Logs ---
    with st.expander("Show Debug Logs", expanded=False):
        st.code("\n".join(debug_logs), language="text")

    return research_summaries, final_report, debug_logs

# Helper function to safely display feedback (handles potential JSON)
# Not strictly needed here, but keeping it doesn't hurt
def display_feedback(feedback_text):
    """Attempts to parse feedback as JSON, otherwise displays as text."""
    try:
        # Check if it looks like JSON before attempting to parse
        if isinstance(feedback_text, str) and feedback_text.strip().startswith(("{", "[")):
            feedback_data = json.loads(feedback_text)
            # If JSON contains a 'feedback' key, display that, otherwise pretty print
            if isinstance(feedback_data, dict) and 'feedback' in feedback_data:
                 # Check if feedback value is a list/dict for pretty printing
                 if isinstance(feedback_data['feedback'], (list, dict)):
                     st.code(json.dumps(feedback_data['feedback'], indent=2), language="json")
                 else:
                     st.text(str(feedback_data['feedback'])) # Display as text if simple value
            else:
                 st.code(json.dumps(feedback_data, indent=2), language="json")
        else:
             st.text(feedback_text) # Display as plain text
    except json.JSONDecodeError:
        st.text(feedback_text) # Display as plain text if JSON parsing fails
    except Exception as e: # Catch other potential errors during display
        st.warning(f"Could not display feedback: {e}")
        st.text(str(feedback_text)) # Fallback to string representation

# --- Streamlit UI ---

# Input for main topic
st.session_state.main_topic = st.text_input(
    "Enter Main Topic:",
    value=st.session_state.main_topic,
    key="main_topic_input"
)

# Button to generate subtopics (added icon)
if st.button("✨ Generate Subtopics"):
    with st.spinner("Generating subtopics..."):
        generated_subtopics = generate_subtopics(st.session_state.main_topic)
        # Update the session state which holds the values for the subtopic input boxes
        st.session_state.subtopics_list = generated_subtopics
        # Force rerun to display new subtopics in inputs - happens automatically on state change

st.divider() # Added divider
st.markdown("#### Generated Subtopics (edit if needed):")

# Display/Edit Subtopics
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.subtopics_list[0] = st.text_input(
        # Added icon to label
        f"1️⃣ Subtopic 1:",
        value=st.session_state.subtopics_list[0],
        key="subtopic1_input",
        label_visibility="collapsed" # Hide label above box, show in box
    )
with col2:
    st.session_state.subtopics_list[1] = st.text_input(
        f"2️⃣ Subtopic 2:",
        value=st.session_state.subtopics_list[1],
        key="subtopic2_input",
        label_visibility="collapsed"
    )
with col3:
     st.session_state.subtopics_list[2] = st.text_input(
        f"3️⃣ Subtopic 3:",
        value=st.session_state.subtopics_list[2],
        key="subtopic3_input",
        label_visibility="collapsed"
    )


st.divider() # Added divider
st.markdown("Click below to run parallel research on the 3 subtopics above and generate the final report.")

# Button to run the main research process (added icon)
if st.button("🚀 Run Parallel Research & Generate Report", type="primary"):
    # Get current subtopics from session state (potentially edited by user)
    current_subtopics = st.session_state.subtopics_list
    valid_subtopics = [t for t in current_subtopics if t.strip()]

    if not valid_subtopics:
        st.warning("Please generate or enter at least one subtopic before running.")
    else:
        with st.spinner("Running parallel research agents and generating report..."):
            try:
                # Call run_parallel_research with the list of valid subtopics
                research_summaries, final_report, _ = run_parallel_research(valid_subtopics)

                # Store results in Streamlit's session state
                st.session_state.research_results = research_summaries
                st.session_state.final_report = final_report

            except Exception as e:
                # Error already displayed in the function, just log here if needed
                print(f"Error caught in Streamlit button click: {e}")
                st.session_state.research_results = None
                st.session_state.final_report = "Failed to generate report due to error."

# Display results if available
if st.session_state.research_results:
    # Use header and add icon
    st.header("📊 Individual Research Summaries", divider='rainbow')
    num_results = len(st.session_state.research_results)
    if num_results > 0:
        cols = st.columns(num_results)
        i = 0
        # Iterate through the results dictionary (keyed by actual topic name)
        for topic, summary in st.session_state.research_results.items():
             if i < len(cols):
                 with cols[i]:
                     st.markdown(f"**{topic}:**") # Use the actual topic name
                     # Use st.success or st.warning for variety, or keep st.info
                     st.info(summary) # Can add icon= "📄" or similar
                 i += 1
             else:
                 st.markdown(f"**{topic}:**")
                 st.info(summary)
    else:
        st.warning("No individual research summaries were generated.")

    st.divider()

if st.session_state.final_report and st.session_state.final_report != "No topics provided.":
    # Use header and add icon
    st.header("📝 Synthesized Report", divider='rainbow')
    st.markdown(st.session_state.final_report) # Display the final report as markdown

