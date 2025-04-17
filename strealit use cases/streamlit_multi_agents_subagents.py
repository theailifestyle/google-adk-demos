import streamlit as st
import asyncio
import os
from typing import Optional, Dict, Any

# --- ADK Imports ---
# Core components for agent, model interaction, session management, and execution
from google.adk.agents import Agent
from google.adk.tools import google_search 
from google.adk.models.lite_llm import LiteLlm # Using LiteLLM for flexibility
from google.adk.sessions import InMemorySessionService, Session # In-memory session storage
from google.adk.runners import Runner # Executes agent interactions
from google.genai import types as genai_types # Google AI types (Content, Part)
# Tool and callback related imports
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool # Helper to wrap Python functions as tools
from google.adk.tools import agent_tool


# Standard Python libraries
import warnings
import logging
import re # Import regex for French detection

# --- Asyncio Configuration for Streamlit ---
# Required to run async ADK functions within Streamlit's synchronous environment
import nest_asyncio
nest_asyncio.apply()

# --- Basic Configuration ---
warnings.filterwarnings("ignore") # Suppress common warnings
logging.basicConfig(level=logging.ERROR) # Reduce log verbosity

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ADK Tutor Bot Team", layout="wide")
st.title("🧑‍🏫 ADK Multi-Agent Tutor Bot Team")
st.caption("A Streamlit app demonstrating Google's ADK for a tutoring scenario.")

# --- Session State for Modifiable Configs ---
# Initialize default configurations only if they don't exist in Streamlit's session state.
# This allows users to modify them via the UI and have the changes persist across reruns
# until the "Apply Changes" button is clicked.


# Default instruction for the Triage Agent
default_tutor_instruction = (
    "You are a Triage Router Agent. Your task is to route user queries to the correct handler and present search results properly. "
    "Handlers available: "
    "1. The 'math_agent' sub-agent: For all questions about mathematics. If the topic is math, **invoke the 'math_agent' to provide the final answer. Its response will be the final response.** " # Changed phrasing
    "2. The 'spanish_agent' sub-agent: For all questions about the Spanish language. If the topic is Spanish, **invoke the 'spanish_agent' to provide the final answer. Its response will be the final response.** " # Changed phrasing
    "3. The 'SearchAgent' tool: For all questions about current events, news, politics, or topics requiring a web search for recent information. Use this tool if the topic requires search. "
    "Analyze the user's query: "
    "- If math-related -> **Invoke the 'math_agent'.** " # Simplified action
    "- If Spanish language-related -> **Invoke the 'spanish_agent'.** " # Simplified action
    "- If current events/news/search-related -> Use the 'SearchAgent' tool. [Rest of search tool instructions...] "
    "- If NONE of these topics, OR a simple greeting/farewell -> Respond ONLY with the exact phrase: 'This service handles Math, Spanish language, or Current Affairs/Search requests.' "
    "Strictly follow these routing rules. Do not answer questions directly yourself unless you are presenting the result from the SearchAgent tool or relaying the final answer from a sub-agent." # Adjusted constraint
)

# Default instruction for the math specialist agent (Answers concepts, uses tool ONLY for simple addition)
default_math_instruction = (
    "You are the Math Agent, an expert in mathematics. Your goal is to answer math-related questions accurately. "
    "First, analyze the user's request. Is it a request for a conceptual explanation, history, definition, or a complex problem (algebra, calculus, etc.)? "
    "If YES, you MUST answer the question directly using your own knowledge and expertise. Provide a clear explanation or solution. "
    "Is the request ONLY a simple numerical addition calculation like 'number + number' (e.g., '4+5', '10.2 + 8')? "
    "If YES, and ONLY in this specific case, you MUST use the 'solve_math_problem' tool and state the result. "
    "DO NOT use the 'solve_math_problem' tool for anything other than simple addition calculations. "
    "NEVER delegate a question back to the tutor agent or any other agent. You must handle all math queries given to you, either by answering directly or using the simple addition tool when appropriate."
)

# Default instruction for the Spanish specialist agent (Answers concepts, uses tool ONLY for specific translations)
default_spanish_instruction = (
    "You are the Spanish Language Agent, an expert in Spanish grammar, vocabulary, and culture. "
    "Your primary role is to answer questions about the Spanish language (like 'Explain subjunctive mood', 'What are common Spanish greetings?', 'Tell me about dialects in Spain') using your own knowledge. "
    "You have a specific tool called 'translate_to_spanish' which can ONLY translate a few specific English words ('hello', 'goodbye', 'thank you', 'cat', 'dog') based on its limited dictionary. "
    "ONLY use the 'translate_to_spanish' tool if the user asks to translate one of those exact words. "
    "For ALL other Spanish language questions (grammar explanations, vocabulary help for other words, cultural information, sentence translations), answer directly using your expertise in Spanish. DO NOT use the tool for these. "
    "If the user asks to translate one of the specific words handled by the tool, use the tool and provide the translation. "
    "If the tool fails or the word is not in its dictionary, state that the specific word could not be translated by the tool."
)
# Define default search instruction globally
default_search_instruction = (
    "You are a specialist agent whose ONLY purpose is to use the 'Google Search' tool "
    "to find information related to the user's query. Execute the search based on the query "
    "and return the findings."
)

# --- Session State for Modifiable Configs ---
if "agent_configs" not in st.session_state:
    st.session_state.agent_configs = {
        "tutor": {"instruction": default_tutor_instruction},
        "math": {"instruction": default_math_instruction},
        "spanish": {"instruction": default_spanish_instruction},
        "search": {"instruction": default_search_instruction} # <-- ADDED THIS LINE
    }

# (Guardrail config remains the same)
if "guardrail_configs" not in st.session_state:
     st.session_state.guardrail_configs = {
         "blocked_keyword": "FORBIDDEN_WORD",
         "blocked_language_tool": "French"
     }
# --- API Key Configuration ---
# Load API keys securely from Streamlit secrets or environment variables
st.sidebar.header("API Key Configuration")
keys_loaded = False
try:
    # Try loading from Streamlit secrets (recommended for deployment)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") # Use .get() for optional key
    keys_loaded = True
    # st.sidebar.success("API Keys loaded from Streamlit secrets.")
except (FileNotFoundError, KeyError):
    # Fallback to environment variables (common for local development)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("🔴 **Error: GOOGLE_API_KEY not found.** Please set it in Streamlit secrets or environment variables.")
        st.stop() # Stop execution if Google key is missing
    if not OPENAI_API_KEY:
         st.warning("🟡 **Warning: OPENAI_API_KEY not found.** Sub-agents using GPT may fail. Set the key or change their model.")
         # Provide a placeholder if missing, so LiteLLM doesn't raise an immediate error if selected
         os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_PLACEHOLDER"
    else:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Ensure env var is set for LiteLLM
        keys_loaded = True
    # st.sidebar.info("API Keys loaded from environment variables.")

# Set environment variables if loaded (needed by ADK/LiteLLM)
if GOOGLE_API_KEY: os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure ADK to use Google Generative AI APIs directly (not Vertex AI)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Model Constants ---
# Define the models to be used
MODEL_GEMINI_FLASH = "gemini-2.0-flash"
MODEL_GPT_4O = "openai/gpt-4o" # LiteLLM format for OpenAI models

# Display models being used in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Models Used:")
st.sidebar.markdown(f"- **Tutor Agent (Root):** `{MODEL_GEMINI_FLASH}`")

# Determine the model for sub-agents based on OpenAI key availability
SUB_AGENT_MODEL_STR = MODEL_GEMINI_FLASH # Default to Gemini
SUB_AGENT_MODEL_OBJ = LiteLlm(model=MODEL_GEMINI_FLASH) # Always use LiteLlm wrapper

if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_PLACEHOLDER":
    SUB_AGENT_MODEL_STR = MODEL_GPT_4O
    SUB_AGENT_MODEL_OBJ = LiteLlm(model=MODEL_GPT_4O) # Use GPT if key is valid
    st.sidebar.markdown(f"- **Sub Agents (Math/Spanish):** `{SUB_AGENT_MODEL_STR}`")
else:
     st.sidebar.markdown(f"- **Sub Agents (Math/Spanish):** `{SUB_AGENT_MODEL_STR}` (OpenAI key missing/invalid, using Gemini)")

# --- Tool Definitions ---
# Define the Python functions that will act as tools for the agents
# --- Tool Definitions ---

def solve_math_problem(query: str) -> str:
    """
    Solves ONLY simple addition problems (e.g., 'x + y').
    (Original Simple Version)
    """
    print(f"--- Tool: solve_math_problem (SIMPLE) executing for query: '{query}' ---") # DEBUG
    parts = query.replace(" ", "").split('+') # Simple split on '+'
    # Basic check if it looks like num+num
    if len(parts) == 2:
        try:
            num1 = float(parts[0])
            # Attempt to clean up potential trailing characters like '?', '='
            num2_str = parts[1]
            cleaned_num2_str = ""
            for char in num2_str:
                if char.isdigit() or char == '.':
                    cleaned_num2_str += char
                else:
                    break # Stop at the first non-numeric/non-decimal character
            if cleaned_num2_str:
                 num2 = float(cleaned_num2_str)
                 result = num1 + num2
                 return f"The answer to the simple addition {query} is {result}."
            else:
                 raise ValueError("Second part not a valid number")
        except ValueError:
            # If parsing fails, let the agent handle it without the tool
             return f"The 'solve_math_problem' tool can only handle simple addition like '5+3'. It couldn't parse '{query}'."
    else:
        # If it's not simple addition, the tool shouldn't be used.
        return f"The 'solve_math_problem' tool is only for simple addition (e.g., '2+2'). It cannot handle '{query}'."

def translate_to_spanish(text: str, tool_context: ToolContext) -> dict:
    """
    Provides a mock translation to Spanish for specific words.
    (Original Simple Version)
    """
    print(f"--- Tool: translate_to_spanish (SIMPLE) executing for text: '{text}' ---") # DEBUG
    tool_context.state["last_translation_request_text"] = text # Keep updating state

    text_lower = text.lower().strip().rstrip('?.!')
    # Limited dictionary
    mock_translations = {
        "hello": "Hola",
        "goodbye": "Adiós",
        "thank you": "Gracias",
        "cat": "Gato",
        "dog": "Perro",
        "bonjour": "Hola (from French 'bonjour')", # Example handling
    }

    if text_lower in mock_translations:
        translation = mock_translations[text_lower]
        report = f"The dictionary translation of '{text}' to Spanish is '{translation}'."
        # Return success only if found in the simple dictionary
        return {"status": "success", "translation": translation, "report": report}
    else:
        # If the word is not in the dictionary, the tool cannot provide the translation.
        # Return an error or specific status the agent can understand.
        error_msg = f"The word '{text}' was not found in the simple translation dictionary."
        print(f"--- Tool: translate_to_spanish (SIMPLE) - {error_msg} ---")
        # Indicate failure for the agent
        return {"status": "error", "error_message": error_msg, "translation": None}
    

# --- Callback Definitions ---
# Callbacks allow intercepting and potentially modifying the agent's behavior
# at different points in the execution cycle (before model call, before tool call, etc.).


# --- Before Model Callback ---
def block_keyword_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects the latest user message for a configured blocked keyword BEFORE sending to the LLM.
    Includes DEBUG print statements.
    """
    agent_name = callback_context.agent_name
    last_user_message_text = ""

    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                part_text = getattr(content.parts[0], 'text', None)
                if part_text:
                    last_user_message_text = part_text
                    break

    keyword_to_block = st.session_state.guardrail_configs.get("blocked_keyword", "").strip().upper()

    # --- DEBUG PRINT STATEMENTS ---
    print(f"\n--- DEBUG [Callback]: Entering block_keyword_guardrail for agent {agent_name} ---")
    print(f"--- DEBUG [Callback]: Last user message: '{last_user_message_text}' ---")
    print(f"--- DEBUG [Callback]: Keyword to block: '{keyword_to_block}' ---")
    # --- END DEBUG ---

    if keyword_to_block and keyword_to_block in last_user_message_text.upper():
        print(f"--- DEBUG [Callback]: Keyword found! Blocking. ---") # DEBUG
        # st.warning(f"Guardrail triggered: Blocked keyword '{keyword_to_block}' found in input for {agent_name}.") # Optional UI feedback
        callback_context.state["guardrail_block_keyword_triggered"] = True
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
        )
    else:
        print(f"--- DEBUG [Callback]: Keyword not found. Allowing. ---") # DEBUG
        callback_context.state["guardrail_block_keyword_triggered"] = False
        return None

# --- Before Tool Callback ---
def block_french_in_tool_guardrail(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Checks if the 'translate_to_spanish' tool is being called with input text
    that appears to be French (using simple keyword check OR langdetect if implemented).
    Includes DEBUG print statements.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name

    target_tool_name = "translate_to_spanish"
    language_to_block = st.session_state.guardrail_configs.get("blocked_language_tool", "French").strip().lower()

    # --- DEBUG PRINT STATEMENTS ---
    print(f"\n--- DEBUG [Callback]: Entering block_french_in_tool_guardrail for agent {agent_name}, tool {tool_name} ---")
    print(f"--- DEBUG [Callback]: Configured language to block: '{language_to_block}' ---")
    # --- END DEBUG ---

    # Only apply this guardrail to the specified tool AND if the configured language is 'french'
    if tool_name == target_tool_name and language_to_block == "french":
        print(f"--- DEBUG [Callback]: Checking tool {target_tool_name} for French input. ---") # DEBUG
        text_argument = args.get("text", "").lower() # Using keyword method here
        print(f"--- DEBUG [Callback]: Text argument received by tool: '{text_argument}' ---") # DEBUG

        # --- Method 1: Keyword Check (Original) ---
        french_indicators = [
             "bonjour", "merci", "oui", "non", "ça va", "parlez", "français", "francais",
             "au revoir", "s'il vous plaît", "svp", "comment", "vous", "appelle",
             "est", "le", "la", "je", "tu", "il", "elle", "pourquoi", "qui", "que", "quoi"
             ] # Expanded list
        is_french = any(indicator in text_argument for indicator in french_indicators)
        print(f"--- DEBUG [Callback]: French indicators found (keyword check): {is_french} ---") # DEBUG
        # --- End Method 1 ---

        # --- Method 2: langdetect (If you installed and imported it) ---
        # Comment out Method 1 and uncomment this section if using langdetect
        # is_french = False
        # text_argument_original_case = args.get("text", "")
        # if text_argument_original_case and len(text_argument_original_case.split()) > 1:
        #     try:
        #         detected_lang = detect(text_argument_original_case)
        #         print(f"--- DEBUG [Callback]: Detected language (langdetect): '{detected_lang}' ---")
        #         if detected_lang == 'fr':
        #             is_french = True
        #     except LangDetectException:
        #         print(f"--- DEBUG [Callback]: Language detection failed. Assuming not French. ---")
        #         is_french = False
        # print(f"--- DEBUG [Callback]: Determined as French (langdetect): {is_french} ---")
        # --- End Method 2 ---

        if is_french:
            print(f"--- DEBUG [Callback]: French detected! Blocking tool. ---") # DEBUG
            # st.warning(f"Guardrail triggered: Tool '{tool_name}' blocked by {agent_name} because input appears to be French.") # Optional UI feedback
            tool_context.state["guardrail_tool_block_language_triggered"] = True
            return {
                "status": "error",
                "error_message": f"Policy restriction: Requests in French are currently disabled by a tool guardrail."
            }
        else:
             print(f"--- DEBUG [Callback]: French not detected. Allowing tool call. ---") # DEBUG
             tool_context.state["guardrail_tool_block_language_triggered"] = False
    else:
         print(f"--- DEBUG [Callback]: Guardrail not applicable (Tool: '{tool_name}', Lang Block: '{language_to_block}'). Allowing tool call. ---") # DEBUG
         tool_context.state["guardrail_tool_block_language_triggered"] = False

    return None

# --- Agent Definitions ---
# Define the agents, including their models, instructions, tools, and sub-agents.
# Use @st.cache_resource to cache agent instances. They are only recreated when
# the cache is explicitly cleared (e.g., by the "Apply Changes" button).
# --- Agent Definitions ---
# Current Affairs Agent ---
@st.cache_resource
def create_search_agent():
    """Creates the Search Agent which uses the Google Search tool."""
    # Use a distinct name for the instruction in session state
    config_key = "search" # Changed from current_affairs
    default_instruction = ( # Define default here or load from main config dict
         "You are a specialist agent whose ONLY purpose is to use the 'Google Search' tool "
         "to find information related to the user's query. Execute the search based on the query "
         "and return the findings."
    )
    # Ensure config exists in session state
    if config_key not in st.session_state.agent_configs:
         st.session_state.agent_configs[config_key] = {"instruction": default_instruction}

    print(f"--- DEBUG: Attempting to create {config_key}_agent ---")
    try:
        instruction = st.session_state.agent_configs[config_key]["instruction"]
        agent = Agent(
            # Use a Gemini model compatible with Google Search
            # Using MODEL_GEMINI_FLASH ("gemini-1.5-flash-latest") as it's known to work
            # Or use "gemini-2.0-flash" if preferred and available
            model="gemini-2.0-flash",
            name='SearchAgent', # Use the name from the example/docs
            instruction=instruction,
            description="A specialist agent that performs Google searches for current events or specific information.",
            tools=[google_search], # Assign the built-in Google Search tool
        )
        print(f"--- DEBUG: {config_key}_agent created. Model: {agent.model}. Tools registered: {[tool.name for tool in agent.tools] if agent.tools else 'None'} ---")
        return agent
    except Exception as e:
        st.error(f"Fatal Error creating Search Agent: {e}. Check model name and instruction.")
        st.stop()


# --- Math Agent ---
@st.cache_resource
def create_math_agent():
    """Creates the Math Agent using instruction from session state."""
    print("--- DEBUG: Attempting to create math_agent ---") # Keep debug if needed
    try:
        instruction = st.session_state.agent_configs["math"]["instruction"]
        agent = Agent(
            model=SUB_AGENT_MODEL_OBJ,
            name="math_agent",
            instruction=instruction,
            description="Handles math problems and questions.",
            tools=[FunctionTool(solve_math_problem)],
             # NO callbacks needed here unless specific to math
        )
        print(f"--- DEBUG: math_agent created. Tools registered: {[tool.name for tool in agent.tools] if agent.tools else 'None'} ---")
        return agent
    except Exception as e:
        st.error(f"Fatal Error creating Math Agent: {e}. Check API keys, model name '{SUB_AGENT_MODEL_STR}', and instruction.")
        st.stop()

@st.cache_resource
def create_spanish_agent():
    """Creates the Spanish Language Agent using instruction from session state."""
    print("--- DEBUG: Attempting to create spanish_agent ---") # Keep debug if needed
    try:
        instruction = st.session_state.agent_configs["spanish"]["instruction"]
        agent = Agent(
            model=SUB_AGENT_MODEL_OBJ,
            name="spanish_agent",
            instruction=instruction,
            description="Handles Spanish translation and language questions.",
            tools=[FunctionTool(translate_to_spanish)],
            # ---> ADD CALLBACK ASSIGNMENT HERE <---
            before_tool_callback=block_french_in_tool_guardrail
        )
        print(f"--- DEBUG: spanish_agent created. Tools registered: {[tool.name for tool in agent.tools] if agent.tools else 'None'} ---")
        print(f"--- DEBUG: spanish_agent callback assigned: {agent.before_tool_callback.__name__ if agent.before_tool_callback else 'None'} ---") # Verify assignment
        return agent
    except Exception as e:
        st.error(f"Fatal Error creating Spanish Agent: {e}. Check API keys, model name '{SUB_AGENT_MODEL_STR}', and instruction.")
        st.stop()

@st.cache_resource
def create_tutor_agent(_math_agent, _spanish_agent, _search_agent): # Added _search_agent parameter
    """Creates the Root Tutor Agent (Hybrid: Sub-Agents + AgentTool)."""
    print("--- DEBUG: Attempting to create tutor_agent (Hybrid version) ---")
    if not _math_agent or not _spanish_agent or not _search_agent: # Check all agents
        st.error("Cannot create Tutor Agent, one or more required agents are not available.")
        st.stop()
    try:
        instruction = st.session_state.agent_configs["tutor"]["instruction"] # Uses the UPDATED hybrid instruction

        agent = Agent(
            name="tutor_agent_hybrid_router", # New name for clarity
            model=MODEL_GEMINI_FLASH,
            description="Router agent that delegates to Math/Spanish sub-agents or uses a Search agent tool.", # Updated description
            instruction=instruction, # Instruction MUST guide the mixed logic

            # --- Keep Math/Spanish as sub_agents ---
            sub_agents=[_math_agent, _spanish_agent],

            # --- Add SearchAgent via AgentTool ---
            tools=[
                agent_tool.AgentTool(agent=_search_agent)
            ],

            output_key="last_agent_response",
            # Keyword callback remains on the root agent
            before_model_callback=block_keyword_guardrail,
            # No tool callback needed directly on the root agent
        )
        print(f"--- DEBUG: tutor_agent_hybrid_router created. Sub-agents: {[sub.name for sub in agent.sub_agents] if agent.sub_agents else 'None'}. Tools: {[tool.name for tool in agent.tools] if agent.tools else 'None'} ---")
        return agent
    except Exception as e:
        st.error(f"Fatal Error creating Tutor Agent: {e}. Check model name '{MODEL_GEMINI_FLASH}' and instruction.")
        st.stop()

# --- Create agent instances ---
# This will use cached versions unless the cache has been cleared.
math_agent = create_math_agent()
spanish_agent = create_spanish_agent()
search_agent = create_search_agent() # Create the new search agent instance
# Pass all three specialist agents to the tutor agent creator
root_tutor_agent = create_tutor_agent(math_agent, spanish_agent, search_agent)


# --- Initialize ADK Runner and Session Service ---
@st.cache_resource # Cache the runner and session service infrastructure
def initialize_adk_infra(_root_agent):
    """Initializes ADK Runner and Session Service."""
    if not _root_agent:
        st.error("Cannot initialize ADK Infra, Root Agent not available.")
        st.stop()

    # Use a simple in-memory session service for this example
    session_service = InMemorySessionService()

    # Define identifiers for the application and user/session
    app_name = "streamlit_tutor_app_editable"
    user_id = "streamlit_user_tutor"
    session_id = "streamlit_session_tutor"
    # Initial state for the session (can be empty or contain starting values)
    initial_state = {"user_preference_language": "English"} # Example state variable

    try:
        # Create the initial session in the session service
        adk_session = session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id, state=initial_state
        )
        st.sidebar.write(f"🔑 ADK Session '{session_id}' created.")
    except Exception as e:
        st.error(f"Fatal Error creating ADK session: {e}")
        st.stop()

    try:
        # Create the Runner, linking the root agent and session service
        runner = Runner(agent=_root_agent, app_name=app_name, session_service=session_service)
        st.sidebar.write("✅ ADK Runner Initialized.") # Indicate readiness
        # Return components needed elsewhere in the app
        return {
            "runner": runner, "session_service": session_service,
            "app_name": app_name, "user_id": user_id, "session_id": session_id
        }
    except Exception as e:
        st.error(f"Fatal Error creating ADK Runner: {e}")
        st.stop()

# --- Get ADK infrastructure components ---
# This uses the cached infrastructure unless cleared.
adk_infra = initialize_adk_infra(root_tutor_agent)
runner = adk_infra["runner"]
session_service = adk_infra["session_service"]
app_name = adk_infra["app_name"]
user_id = adk_infra["user_id"]
session_id = adk_infra["session_id"]

# --- Sidebar Configuration UI ---
# Allow users to modify agent instructions and guardrail parameters
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")

# Button to apply changes: Clears caches and resets chat
st.sidebar.info("Modify settings below and click Apply to rebuild agents.")
if st.sidebar.button("Apply Changes & Reset Chat", key="apply_changes"):
    # Clear the caches for agents and infrastructure to force recreation
    create_math_agent.clear()
    create_spanish_agent.clear()
    create_search_agent.clear() # Clear the new search agent
    create_tutor_agent.clear()
    initialize_adk_infra.clear()

    # Reset chat history in Streamlit's session state
    st.session_state.messages = []

    st.sidebar.success("Configuration applied! Caches cleared & chat reset.")
    st.toast("Agents rebuilt with new configuration!")
    # Force a rerun of the Streamlit script to pick up changes and rebuild
    st.rerun()

# Expanders for editing configurations stored in st.session_state
with st.sidebar.expander("Agent Instructions", expanded=False):
     # Text area for root Tutor Agent instruction
     st.session_state.agent_configs["tutor"]["instruction"] = st.text_area(
        "Tutor Agent (Root) Instruction",
        value=st.session_state.agent_configs["tutor"]["instruction"],
        height=250,
        key="tutor_instruction_input" # Unique key for the widget
    )
     # Text area for Math Agent instruction
     st.session_state.agent_configs["math"]["instruction"] = st.text_area(
        "Math Agent Instruction",
        value=st.session_state.agent_configs["math"]["instruction"],
        height=150,
        key="math_instruction_input"
    )
     # Text area for Spanish Agent instruction
     st.session_state.agent_configs["spanish"]["instruction"] = st.text_area(
        "Spanish Agent Instruction",
        value=st.session_state.agent_configs["spanish"]["instruction"],
        height=150,
        key="spanish_instruction_input"
    )
     # Text area for Search Agent instruction
     st.session_state.agent_configs["search"]["instruction"] = st.text_area(
        "Search Agent Instruction",
        value=st.session_state.agent_configs["search"]["instruction"], # Now safe
        height=150,
        key="search_instruction_input"
     )

# Expander for guardrail parameters
with st.sidebar.expander("Guardrail Parameters", expanded=False):
    # Input for the blocked keyword (model input guardrail)
    st.session_state.guardrail_configs["blocked_keyword"] = st.text_input(
        "Keyword to Block (Model Input Guardrail)",
        value=st.session_state.guardrail_configs["blocked_keyword"],
        key="blocked_keyword_input"
    )
    # Input for the language to block (tool input guardrail - Spanish tool)
    st.session_state.guardrail_configs["blocked_language_tool"] = st.text_input(
        "Language to Block (Tool Input Guardrail - Spanish Tool, e.g., 'French')",
        value=st.session_state.guardrail_configs["blocked_language_tool"],
        key="blocked_language_tool_input"
    )

# --- Chat History Initialization ---
# Initialize the chat message history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores messages as {"role": "user/assistant", "content": "..."}

# --- Display Chat History ---
# Iterate through the stored messages and display them using Streamlit's chat elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # "user" or "assistant"
        st.markdown(message["content"], unsafe_allow_html=True) # Display content (allow basic HTML for formatting)

# --- Agent Interaction Logic ---
async def get_agent_response(user_query: str) -> tuple[str, str]:
    """
    Sends the user query to the ADK runner and processes the asynchronous events
    to extract the final response text and the name of the agent that produced it.
    """
    # Create the user message content in the format expected by ADK
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_query)])

    # Initialize default response values
    final_response_text = "Agent did not produce a final response."
    final_response_author = "system" # Default author if none is found

    try:
        # Asynchronously iterate through events generated by the runner
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Check if the event is the final response from an agent
            if event.is_final_response():
                final_response_author = event.author if event.author else "unknown_agent" # Get the agent's name

                # Extract text content from the response parts
                if event.content and event.content.parts:
                    text_parts = [getattr(part, 'text', '') for part in event.content.parts if hasattr(part, 'text')]
                    final_response_text = " ".join(filter(None, text_parts)) # Join non-empty text parts
                    if not final_response_text: # Handle case where parts exist but have no text
                         final_response_text = "(Agent returned empty text content)"

                # Handle cases where the final event indicates an error or specific action
                elif event.error_message:
                    final_response_text = f"Agent Error: {event.error_message}"
                    final_response_author = event.author if event.author else "error_handler"
                elif event.actions and event.actions.escalate:
                     final_response_text = f"Action Required: Escalated. Reason: {event.error_message or 'None specified'}"
                     final_response_author = event.author if event.author else "escalation_handler"
                # Fallback if none of the above conditions extracted text
                elif final_response_text == "Agent did not produce a final response.":
                     final_response_text = "(Final response received with no displayable content or error)"

                break # Stop processing events once the final response is found
    except Exception as e:
        # Catch errors during the agent execution
        st.error(f"An error occurred during agent interaction: {e}")
        final_response_text = f"Sorry, a critical error occurred: {e}"
        final_response_author = "system_error"

    # Clean up the author name (sometimes ADK might provide a longer identifier)
    if isinstance(final_response_author, str):
         final_response_author = final_response_author.split('.')[-1] # Get the last part (agent name)

    return final_response_text, final_response_author

# --- Handle User Input ---
# Get input from the user via Streamlit's chat input widget
if prompt := st.chat_input("Ask the tutor bot (e.g., 'What is 5 + 12?', 'Translate hello to Spanish')..."):
    # 1. Add user message to Streamlit's history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show a spinner while waiting for the agent's response
    with st.spinner("Thinking..."):
        # 3. Call the async function to get the agent response
        # asyncio.run is used here because Streamlit runs synchronously
        response_text, agent_name = asyncio.run(get_agent_response(prompt))

        # Format the response to include the agent's name
        display_response = f"**[{agent_name}]** {response_text}"

        # 4. Add the assistant's response to Streamlit's history and display it
        st.session_state.messages.append({"role": "assistant", "content": display_response})
        with st.chat_message("assistant"):
            st.markdown(display_response, unsafe_allow_html=True)

# --- Display Current ADK Session State and History ---
# This section runs after every interaction to show the internal state of the ADK session.
st.sidebar.markdown("---")
st.sidebar.header("📊 Session Data")
try:
    # Retrieve the current ADK session object from the session service
    current_adk_session: Optional[Session] = session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    # Check if the session was retrieved successfully
    if current_adk_session:

        # --- Section 1: Display Session State ---
        st.sidebar.write("**Current State:**") # Header for state section
        state_dict = current_adk_session.state # Access the state dictionary

        # Display relevant state variables set by tools or callbacks
        st.sidebar.write(f"- User Pref Lang: `{state_dict.get('user_preference_language', 'N/A')}`")
        st.sidebar.write(f"- Last Translation Input: `{state_dict.get('last_translation_request_text', 'N/A')}`")
        # Display guardrail trigger status
        model_kw_triggered = state_dict.get('guardrail_block_keyword_triggered', 'N/A')
        tool_lang_triggered = state_dict.get('guardrail_tool_block_language_triggered', 'N/A')
        st.sidebar.write(f"- Model KW Guardrail Triggered: `{model_kw_triggered}`")
        st.sidebar.write(f"- Tool Lang Guardrail Triggered: `{tool_lang_triggered}`")

        # Expander to show the raw state dictionary for debugging
        with st.sidebar.expander("Raw State Dictionary", expanded=False):
             st.json(state_dict if state_dict else {"state": "empty"})
        # --- End Section 1 ---


        # --- Section 2: Display Session History (from Events) ---
        # Access the event history stored within the Session object
        session_events = getattr(current_adk_session, 'events', None) # Use getattr for safe access

        if session_events is not None and isinstance(session_events, list):
            # Display events inside an expander
            with st.sidebar.expander(f"Detailed Event History ({len(session_events)} events)", expanded=True):
                 if session_events:
                     # Loop through each event object in the history
                     for i, event_obj in enumerate(session_events):
                         # --- Extract relevant info from the event object safely using getattr ---
                         author = getattr(event_obj, 'author', 'unknown_author')
                         role = getattr(event_obj, 'role', None) # Role might not always be present
                         display_actor = f"{author}" + (f" (Role: {role})" if role else "") # Combine author and role if available

                         content_obj = getattr(event_obj, 'content', None)
                         parts_text = []
                         function_calls = []
                         function_responses = []

                         # --- Safely extract details from content parts ---
                         if content_obj and getattr(content_obj, 'parts', None):
                            for part in content_obj.parts:
                                # Try getting different payload types
                                fc = getattr(part, 'function_call', None)
                                fr = getattr(part, 'function_response', None)
                                pt = getattr(part, 'text', None)

                                # Append extracted info to respective lists
                                if fc:
                                    fc_name = getattr(fc, 'name', '(unknown func)')
                                    fc_args = getattr(fc, 'args', {})
                                    function_calls.append(f"Tool Call: {fc_name}({fc_args})")
                                elif fr:
                                    fr_name = getattr(fr, 'name', '(unknown func)')
                                    fr_response = getattr(fr, 'response', {})
                                    # Truncate long responses for display
                                    response_str = f"{fr_response}"
                                    if len(response_str) > 150:
                                        response_str = response_str[:150] + "..."
                                    function_responses.append(f"Tool Resp: {fr_name} -> {response_str}")
                                elif pt:
                                    parts_text.append(pt)

                         # --- Display formatted event information ---
                         st.markdown(f"**{i+1}. By:** `{display_actor}`")
                         if parts_text:
                             st.text("Text: " + " ".join(parts_text))
                         if function_calls:
                             st.text(" ".join(function_calls))
                         if function_responses:
                             st.text(" ".join(function_responses))

                         # Display the type of event (e.g., LlmRequestEvent, ToolRequestEvent)
                         event_type = getattr(event_obj, '__class__', None)
                         if event_type:
                             st.caption(f"Event Type: {event_type.__name__}")

                         # --- Optional Raw Event Data for Debugging ---
                         # with st.expander("Raw Event Data (DEBUG)", expanded=False):
                         #     st.write(event_obj)
                         # --- End Raw Event Data ---

                         st.markdown("---") # Separator between events
                 else:
                     st.write("Session event list is empty.")
        else:
             st.write("Could not find a valid 'events' list on the session object.")
        # --- End Section 2 ---

    else:
        # Warning if the session object couldn't be found
        st.sidebar.warning(f"ADK Session '{session_id}' not found in the service.")

# Catch potential errors during session data access or display
except AttributeError as ae:
     st.sidebar.error(f"Error accessing session attributes: {ae}. Session structure might be unexpected.")
except Exception as e:
    st.sidebar.error(f"Error retrieving/displaying session data: {e}")
    # import traceback # Uncomment for detailed debugging
    # st.sidebar.text(traceback.format_exc()) # Uncomment for detailed debugging