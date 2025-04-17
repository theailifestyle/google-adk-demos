import streamlit as st
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from dotenv import load_dotenv
load_dotenv()
import traceback
import uuid # Use UUID for unique session IDs
import json # To safely parse potential JSON in feedback

# Page setup
st.set_page_config(page_title="Pitch Refiner v2 - Google ADK Demo", layout="wide")
st.title("Pitch/Elevator Speech Refiner")
# Updated markdown description for Critic -> Writer
st.markdown("*Transform your rough ideas into compelling, concise pitches with AI assistance (v2: Critic-First Loop)*")

# --- Constants ---
APP_NAME = "pitch_refiner_app_v2_critic_first_minimal" # Updated app name
USER_ID = "dev_user_01"
# IMPORTANT: Replace with a valid Gemini model name available in your environment
GEMINI_MODEL = "gemini-1.5-flash-latest" # Or "gemini-1.0-pro", etc.

# --- State Keys ---
STATE_CURRENT_PITCH = "current_pitch"
STATE_FEEDBACK = "feedback"

# Initialize session state if needed
if 'history' not in st.session_state:
    st.session_state.history = []

# Pitch input UI
col1, col2 = st.columns([3, 1])

with col1:
    default_pitch = "We're building an app that uses AI to analyze financial data and predict market trends."
    pitch_idea = st.text_area("Enter your rough pitch idea:", value=default_pitch, height=150)

with col2:
    # Target audience selection
    audience_options = [
        "Investors",
        "Customers",
        "General Audience",
        "Technical Team",
        "Executive Leadership"
    ]
    target_audience = st.selectbox("Target Audience:", audience_options)

    # Number of iterations (Each iteration = Critic + Writer)
    # Note: The loop will run Critic, Writer, Critic, Writer... ending after Writer.
    # Updated slider label
    num_iterations = st.slider("Refinement Iterations (Critique + Write cycles):", min_value=1, max_value=3, value=2)

# Function to setup and run agent
def setup_and_run_agent(initial_pitch, target_audience, num_iterations):
    """Sets up and runs the ADK agents for pitch refinement."""
    debug_logs = [] # Initialize list to store debug messages

    # Create a unique session ID for each run to avoid state conflicts
    session_id = f"session_{uuid.uuid4()}"
    debug_logs.append(f"Created unique session ID: {session_id}")

    # --- Session and Runner ---
    session_service = InMemorySessionService()
    # Session state starts empty
    session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    debug_logs.append(f"ADK Session created for app: {APP_NAME}, user: {USER_ID}, session: {session_id}. State is initially empty.")

    # --- Agent Definitions ---

    # Critic Agent (LlmAgent) - Runs FIRST in the loop now
    # Instruction still asks to read from state, but maybe it gets initial message implicitly?
    critic_agent = LlmAgent(
        name="PitchCritic",
        model=GEMINI_MODEL,
        instruction=f"""
        You are an expert Pitch Coach specialized in elevator speech refinement.

        **Your Task:** Review the pitch provided. If state variable '{STATE_CURRENT_PITCH}' exists, use that. Otherwise, use the initial input message.
        Provide specific, actionable feedback (2-3 points) to make it more compelling
        for the target audience: {target_audience}.

        **Focus Areas:**
        - Clarity: Is the value proposition immediately clear?
        - Impact: Does it create interest and engagement?
        - Conciseness: Is it focused and free of unnecessary details?
        - Audience fit: Is it appropriate for {target_audience}?

        **Audience Considerations ({target_audience}):**
         - Investors: Look for market potential, ROI, competitive advantage.
         - Customers: Look for benefits, problem-solving, emotional appeal.
         - General Audience: Look for accessible language, relatable examples.
         - Technical Team: Look for technical differentiators, feasibility.
         - Executive Leadership: Look for strategic alignment, scalability, business impact.

        **Output:** Output ONLY your feedback (e.g., a list of points) without explanations or commentary. Do not output the pitch itself.
        """,
        description="Reviews and critiques the pitch.",
        output_key=STATE_FEEDBACK  # Saves critique to state
    )
    debug_logs.append("Critic Agent defined (runs first).")

    # Writer Agent (LlmAgent) - Runs SECOND in the loop now
    writer_agent = LlmAgent(
        name="PitchWriter",
        model=GEMINI_MODEL,
        instruction=f"""
        You are an expert Pitch Writer specialized in creating compelling elevator speeches.

        **Your Task:**
        1. Read the current pitch from the '{STATE_CURRENT_PITCH}' state variable (this should exist after the critic runs, even if it's the first draft).
        2. Read the feedback provided in the '{STATE_FEEDBACK}' state variable.
        3. Refine the pitch based *only* on the provided feedback to make it clear, concise (30-45 words),
           and highlight the unique value proposition for the target audience: {target_audience}.

        **Audience Focus ({target_audience}):**
         - Investors: Focus on market potential, ROI, and competitive advantage.
         - Customers: Emphasize benefits, problem-solving aspects, and emotional appeal.
         - General Audience: Use accessible language and relatable examples.
         - Technical Team: Include relevant technical differentiators and implementation feasibility.
         - Executive Leadership: Highlight strategic alignment, scalability, and business impact.

        **Output:** Output ONLY the single, refined pitch without explanations or commentary.
        """,
        description="Refines the pitch based on feedback from session state.",
        output_key=STATE_CURRENT_PITCH  # Saves refined pitch back to state, overwriting previous
    )
    debug_logs.append("Writer Agent defined (runs second).")


    # --- Create the LoopAgent ---
    # Order is now Critic -> Writer. The loop finishes after the Writer.
    loop_agent = LoopAgent(
        name="PitchRefinerLoop",
        sub_agents=[critic_agent, writer_agent], # CRITIC runs first now
        max_iterations=num_iterations
    )
    debug_logs.append(f"Loop Agent defined with max_iterations: {num_iterations}")

    runner = Runner(agent=loop_agent, app_name=APP_NAME, session_service=session_service)
    debug_logs.append("ADK Runner initialized.")

    # --- Agent Interaction ---
    # Pass the initial pitch via the first message to the runner.
    # Based on user observation, the Critic might pick this up even though its prompt refers to state.
    initial_message_text = f"Refine this pitch idea for {target_audience}: {initial_pitch}"
    content = types.Content(role='user', parts=[types.Part(text=initial_message_text)])
    debug_logs.append(f"Starting runner.run for session {session_id} with initial message: '{initial_message_text[:100]}...'")

    # Track agent outputs in sequence with agent names
    agent_outputs = []
    final_pitch = "Error: Pitch not generated." # Default value
    final_feedback = "Error: Feedback not generated." # Default value

    events = runner.run(user_id=USER_ID, session_id=session_id, new_message=content)

    debug_logs.append(f"runner.run finished for session {session_id}. Processing events...")
    event_count = 0
    for event in events:
        event_count += 1
        debug_logs.append(f"Processing event {event_count}...")
        if event.is_final_response():
            response_text = event.content.parts[0].text
            debug_logs.append(f"Final response detected: '{response_text[:100]}...'")

            # Track which agent produced this output (Critic is even, Writer is odd)
            if len(agent_outputs) % 2 == 0: # Critic ran (index 0, 2, ...)
                agent_name = "PitchCritic"
                final_feedback = response_text # Store the latest feedback
                debug_logs.append(f"Attributing to {agent_name} (Feedback)")
            else: # Writer ran (index 1, 3, ...)
                agent_name = "PitchWriter"
                final_pitch = response_text # Store the latest pitch
                debug_logs.append(f"Attributing to {agent_name} (Pitch)")

            agent_outputs.append((agent_name, response_text))
        elif event.is_error():
             # Keep actual errors visible
             st.error(f"Error during agent execution: {event.error}")
             debug_logs.append(f"ERROR event detected: {event.error}")
             break # Stop processing events on error

    debug_logs.append(f"Finished processing {event_count} events.")
    debug_logs.append(f"Agent outputs collected: {len(agent_outputs)} steps.")

    # --- Get Final State (Verification) ---
    try:
        updated_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        debug_logs.append(f"Fetched final session state for {session_id}.")
        if hasattr(updated_session, 'state'):
            state_dict = updated_session.state
            final_pitch_from_state = state_dict.get(STATE_CURRENT_PITCH)
            final_feedback_from_state = state_dict.get(STATE_FEEDBACK)
            debug_logs.append(f"Final pitch from state check: '{final_pitch_from_state}'")
            debug_logs.append(f"Final feedback from state check: '{final_feedback_from_state}'")
            # Compare with values captured during event processing
            if final_pitch != final_pitch_from_state:
                 debug_logs.append("WARNING: Final pitch captured during events differs from final session state.")
            if final_feedback != final_feedback_from_state:
                 debug_logs.append("WARNING: Final feedback captured during events differs from final session state.")
        else:
             debug_logs.append(f"WARNING: Final session {session_id} has no state attribute.")
    except Exception as e:
        st.warning(f"Could not retrieve/verify final session state: {e}")
        debug_logs.append(f"ERROR fetching/verifying final session state: {e}")


    # --- Display Debug Logs ---
    with st.expander("Show Debug Logs", expanded=False):
        st.code("\n".join(debug_logs), language="text")

    # --- Return Results ---
    # final_pitch holds the last output from PitchWriter
    # final_feedback holds the last output from PitchCritic
    return final_pitch, final_feedback, agent_outputs

# Helper function to safely display feedback (handles potential JSON)
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


# Button to run the agent
if st.button("Refine My Pitch (v2)"):
    # Updated spinner text for Critic -> Writer
    with st.spinner("Working on your pitch... (Critic -> Writer loop)"):
        try:
            final_pitch, final_feedback, agent_outputs = setup_and_run_agent(pitch_idea, target_audience, num_iterations)

            if final_pitch is not None and final_pitch != "Error: Pitch not generated.":
                st.session_state.history.append({
                    "original_idea": pitch_idea,
                    "target_audience": target_audience,
                    "final_pitch": final_pitch, # Last pitch from Writer
                    "final_feedback": final_feedback, # Last feedback from Critic
                    "agent_outputs": agent_outputs,
                    "iterations": num_iterations
                })
            else:
                st.error("Pitch refinement process did not complete successfully. Check logs if needed.")

        except Exception as e:
            st.error(f"An unexpected error occurred when trying to run the agent: {str(e)}")
            st.code(f"Full error: {traceback.format_exc()}", language="python")

# Display results
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.subheader("Your Refined Pitch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Idea")
        st.info(latest["original_idea"])
    with col2:
        # Updated label: This is the final output when ending with Writer
        st.markdown("#### Final Pitch")
        st.success(latest["final_pitch"])
        # Updated caption
        st.caption(f"Tailored for: {latest['target_audience']} (After {latest['iterations']} critique/write cycles)")

    # Word count and metrics
    original_word_count = len(latest["original_idea"].split())
    final_pitch_text = latest["final_pitch"] if isinstance(latest["final_pitch"], str) else ""
    final_word_count = len(final_pitch_text.split())
    delta_wc = final_word_count - original_word_count if final_pitch_text else -original_word_count

    st.markdown("#### Pitch Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        # Updated metric label
        st.metric("Word Count (Final Pitch)", final_word_count, delta=delta_wc if final_pitch_text else None)
    with metric_col2:
        ideal_range = "30-45 words"
        status = "❓"
        if final_pitch_text:
            status = "✅ Ideal" if 30 <= final_word_count <= 45 else "⚠️ Not ideal"
        st.metric("Ideal Range", ideal_range, delta=status, delta_color="off")
    with metric_col3:
        # Updated metric label
        st.metric("Critique/Write Cycles", latest["iterations"])

    # Display the *last* feedback provided (which led to the final pitch)
    # Updated section header
    st.markdown("#### Final Feedback Provided")
    display_feedback(latest["final_feedback"]) # Use helper function
    # Updated caption
    st.caption("(This feedback was used to generate the final pitch above)")


    # Evolution of the pitch
    st.subheader("Pitch Evolution")

    # Group agent outputs by iteration (Critic + Writer = 1 iteration)
    iterations_display = []
    current_iter_display = {}

    for i, (agent_name, output) in enumerate(latest["agent_outputs"]):
        # Corrected grouping logic for Critic -> Writer
        if agent_name == "PitchCritic":
            # Start of a new iteration display group
            if current_iter_display and current_iter_display.get("writer_draft") is not None:
                 iterations_display.append(current_iter_display)
            current_iter_display = {"critic_feedback": output, "writer_draft": None}
        elif agent_name == "PitchWriter":
            if "critic_feedback" in current_iter_display:
                 current_iter_display["writer_draft"] = output
            else:
                 # Should not happen if Critic always runs first in a cycle
                 print(f"Display Warning: Writer output found without preceding Critic feedback at step {i}")
                 current_iter_display = {"critic_feedback": "Missing/Error", "writer_draft": output}
                 iterations_display.append(current_iter_display) # Log potentially broken iteration
                 current_iter_display = {} # Reset

    # Add the last completed iteration (should have both parts if loop completes)
    if current_iter_display and current_iter_display.get("critic_feedback") is not None and current_iter_display.get("writer_draft") is not None:
        iterations_display.append(current_iter_display)
    elif current_iter_display:
         # Handle cases where loop might end unexpectedly
         print(f"Display Warning: Last iteration data incomplete: {current_iter_display}")
         # Decide whether to display partial data or not
         # iterations_display.append(current_iter_display)


    # Display iterations
    if not iterations_display:
         st.info("No complete refinement cycles were recorded (check for errors during execution).")

    for iter_num, data in enumerate(iterations_display):
        if isinstance(data, dict):
            # Expand the last one by default
            is_expanded = (iter_num == len(iterations_display) - 1)
            with st.expander(f"Cycle {iter_num + 1}", expanded=is_expanded):
                # Updated headers for Critic -> Writer
                st.markdown("##### 1. Feedback Received (Critic)")
                display_feedback(data.get("critic_feedback", "N/A")) # Use helper
                st.markdown("---")
                st.markdown("##### 2. Resulting Draft (Writer)")
                writer_draft = data.get("writer_draft", "N/A")
                # The final draft of the last iteration IS the final pitch displayed above
                if iter_num == len(iterations_display) - 1:
                    st.success(writer_draft)
                    st.caption("(This is the final pitch)")
                else:
                    st.info(writer_draft)
        else:
            print(f"Display Error: Iteration {iter_num+1} data is not in the expected format.")


# History section
if len(st.session_state.history) > 1:
    with st.expander("Previous Pitch Results"):
        for i, item in enumerate(reversed(st.session_state.history[:-1])):
            history_index = len(st.session_state.history) - 1 - i
            st.markdown(f"---")
            st.markdown(f"##### Result {history_index}: For {item['target_audience']} ({item['iterations']} cycles)")
            # Updated labels for clarity (Critic->Writer ends with Final Pitch)
            st.success(f"**Final Pitch:** {item['final_pitch']}")
            st.markdown("**Final Feedback Given:**") # This led to the final pitch
            display_feedback(item['final_feedback']) # Use helper function
            st.caption(f"*Original idea:* {item['original_idea'][:60]}..." if len(item['original_idea']) > 60 else item['original_idea'])
