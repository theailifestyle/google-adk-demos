import asyncio
import os
import logging
import warnings
from typing import Optional, Dict, Any, AsyncGenerator
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# --- ADK Imports ---
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService, Session # Using InMemory as per tutorial
from google.adk.runners import Runner
from google.genai import types as genai_types
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools import agent_tool
from google.adk.events import Event # For serializing events

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO) # More verbose for backend
logger = logging.getLogger(__name__)

# --- Environment Variable Loading (Simulating Streamlit Secrets/Env Vars) ---
# Ensure API keys are set in your environment before running
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("🔴 **Error: GOOGLE_API_KEY not found in environment variables.**")
    # For a real app, you might raise an exception or exit
if not OPENAI_API_KEY:
    logger.warning("🟡 **Warning: OPENAI_API_KEY not found. Sub-agents using GPT may fail.**")
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_PLACEHOLDER" # For LiteLLM
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Model Constants ---
MODEL_GEMINI_FLASH = "gemini-1.5-flash-latest" # or "gemini-2.0-flash" if preferred
MODEL_GPT_4O = "openai/gpt-4o"

SUB_AGENT_MODEL_STR = MODEL_GEMINI_FLASH
SUB_AGENT_MODEL_OBJ = LiteLlm(model=MODEL_GEMINI_FLASH)
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_PLACEHOLDER":
    SUB_AGENT_MODEL_STR = MODEL_GPT_4O
    SUB_AGENT_MODEL_OBJ = LiteLlm(model=MODEL_GPT_4O)

# --- Agent Instructions (Copied from Streamlit app) ---
default_current_affairs_instruction = (
    "You are the Current Affairs Agent. Your primary function is to answer questions about recent events, news, politics, "
    "or any topic requiring up-to-date information. You MUST use the 'Google Search' tool provided to find current information "
    "from the internet. Synthesize the search results into a concise and helpful answer. Do not rely on your internal knowledge "
    "for time-sensitive topics; always perform a search."
)
default_tutor_instruction = (
    "You are a Triage Router Agent. Your ONLY goal is to understand the user's query and use the single most appropriate specialized tool to handle it. "
    "Available tools: "
    "1. 'math_agent': Use this tool for ALL questions about mathematics. "
    "2. 'spanish_agent': Use this tool for ALL questions about the Spanish language. "
    "3. 'SearchAgent': Use this tool for ALL questions about news, current events, or topics requiring recent information search. "
    "Analyze the user's query. Based ONLY on the topic, choose the ONE relevant tool ('math_agent', 'spanish_agent', or 'SearchAgent') and invoke it with the user's full query. "
    "After the tool finishes, present the result provided by the tool clearly to the user. If the tool call results in an error or no specific result, state that. "
    "If the query topic is unclear, does not match any tool's specialty, or is a simple greeting/farewell, you MUST respond ONLY with the exact phrase: 'This service only handles questions about Math, Spanish language, or Current Affairs/Search.' "
    "DO NOT attempt to answer any questions directly yourself, only use the appropriate tool and present its result."
)
default_math_instruction = (
    "You are the Math Agent, an expert in mathematics. Your goal is to answer math-related questions accurately. "
    "First, analyze the user's request. Is it a request for a conceptual explanation, history, definition, or a complex problem (algebra, calculus, etc.)? "
    "If YES, you MUST answer the question directly using your own knowledge and expertise. Provide a clear explanation or solution. "
    "Is the request ONLY a simple numerical addition calculation like 'number + number' (e.g., '4+5', '10.2 + 8')? "
    "If YES, and ONLY in this specific case, you MUST use the 'solve_math_problem' tool and state the result. "
    "DO NOT use the 'solve_math_problem' tool for anything other than simple addition calculations. "
    "NEVER delegate a question back to the tutor agent or any other agent. You must handle all math queries given to you, either by answering directly or using the simple addition tool when appropriate."
)
default_spanish_instruction = (
    "You are the Spanish Language Agent, an expert in Spanish grammar, vocabulary, and culture. "
    "Your primary role is to answer questions about the Spanish language (like 'Explain subjunctive mood', 'What are common Spanish greetings?', 'Tell me about dialects in Spain') using your own knowledge. "
    "You have a specific tool called 'translate_to_spanish' which can ONLY translate a few specific English words ('hello', 'goodbye', 'thank you', 'cat', 'dog') based on its limited dictionary. "
    "ONLY use the 'translate_to_spanish' tool if the user asks to translate one of those exact words. "
    "For ALL other Spanish language questions (grammar explanations, vocabulary help for other words, cultural information, sentence translations), answer directly using your expertise in Spanish. DO NOT use the tool for these. "
    "If the user asks to translate one of the specific words handled by the tool, use the tool and provide the translation. "
    "If the tool fails or the word is not in its dictionary, state that the specific word could not be translated by the tool."
)
default_search_instruction = (
    "You are a specialist agent whose ONLY purpose is to use the 'Google Search' tool "
    "to find information related to the user's query. Execute the search based on the query "
    "and return the findings."
)

agent_configs = {
    "tutor": {"instruction": default_tutor_instruction},
    "math": {"instruction": default_math_instruction},
    "spanish": {"instruction": default_spanish_instruction},
    "search": {"instruction": default_search_instruction}
}
guardrail_configs = { # Simplified for backend
    "blocked_keyword": "FORBIDDEN_WORD",
    "blocked_language_tool": "French"
}

# --- Tool Definitions (Copied) ---
def solve_math_problem(query: str) -> str:
    logger.info(f"--- Tool: solve_math_problem executing for query: '{query}' ---")
    parts = query.replace(" ", "").split('+')
    if len(parts) == 2:
        try:
            num1 = float(parts[0])
            num2_str = parts[1]
            cleaned_num2_str = "".join(filter(lambda char: char.isdigit() or char == '.', num2_str))
            if cleaned_num2_str:
                 num2 = float(cleaned_num2_str)
                 return f"The answer to {query} is {num1 + num2}."
            else: raise ValueError("Second part not a valid number")
        except ValueError:
             return f"Tool 'solve_math_problem' couldn't parse '{query}' for simple addition."
    return f"Tool 'solve_math_problem' is only for simple addition. Cannot handle '{query}'."

def translate_to_spanish(text: str, tool_context: ToolContext) -> dict:
    logger.info(f"--- Tool: translate_to_spanish executing for text: '{text}' ---")
    tool_context.state["last_translation_request_text"] = text
    text_lower = text.lower().strip().rstrip('?.!')
    mock_translations = {"hello": "Hola", "goodbye": "Adiós", "thank you": "Gracias", "cat": "Gato", "dog": "Perro"}
    if text_lower in mock_translations:
        translation = mock_translations[text_lower]
        return {"status": "success", "translation": translation, "report": f"Translated '{text}' to '{translation}'."}
    return {"status": "error", "error_message": f"Word '{text}' not in simple dictionary.", "translation": None}

# --- Callback Definitions (Copied and simplified logging) ---
def block_keyword_guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts and hasattr(content.parts[0], 'text'):
                retrieved_text = content.parts[0].text
                last_user_message_text = retrieved_text if retrieved_text is not None else ""
                break
    keyword_to_block = guardrail_configs.get("blocked_keyword", "").strip().upper()
    logger.debug(f"Callback block_keyword_guardrail for {agent_name}: User msg: '{last_user_message_text}', Blocking: '{keyword_to_block}'")
    if keyword_to_block and last_user_message_text and keyword_to_block in last_user_message_text.upper():
        callback_context.state["guardrail_block_keyword_triggered"] = True
        return LlmResponse(content=genai_types.Content(role="model", parts=[genai_types.Part(text=f"Blocked keyword '{keyword_to_block}'.")]))
    callback_context.state["guardrail_block_keyword_triggered"] = False
    return None

def block_french_in_tool_guardrail(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext) -> Optional[Dict]:
    # Simplified for brevity, actual French detection can be more complex
    tool_name = tool.name
    agent_name = tool_context.agent_name
    logger.debug(f"Callback block_french_in_tool_guardrail for {agent_name}, tool {tool_name}")
    if tool_name == "translate_to_spanish" and guardrail_configs.get("blocked_language_tool", "").lower() == "french":
        text_argument = args.get("text", "").lower()
        if "bonjour" in text_argument: # Very basic check
            tool_context.state["guardrail_tool_block_language_triggered"] = True
            return {"status": "error", "error_message": "French input blocked by tool guardrail."}
    tool_context.state["guardrail_tool_block_language_triggered"] = False
    return None

# --- Agent Definitions (Adapted) ---
# Global agent instances for simplicity in FastAPI context
math_agent_instance: Optional[Agent] = None
spanish_agent_instance: Optional[Agent] = None
search_agent_instance: Optional[Agent] = None
root_tutor_agent_instance: Optional[Agent] = None

def create_agents():
    global math_agent_instance, spanish_agent_instance, search_agent_instance, root_tutor_agent_instance
    logger.info("Creating agents...")
    try:
        math_agent_instance = Agent(
            model=SUB_AGENT_MODEL_OBJ, name="math_agent",
            instruction=agent_configs["math"]["instruction"],
            tools=[FunctionTool(solve_math_problem)]
        )
        spanish_agent_instance = Agent(
            model=SUB_AGENT_MODEL_OBJ, name="spanish_agent",
            instruction=agent_configs["spanish"]["instruction"],
            tools=[FunctionTool(translate_to_spanish)],
            before_tool_callback=block_french_in_tool_guardrail
        )
        search_agent_instance = Agent(
            model=MODEL_GEMINI_FLASH, name='SearchAgent',
            instruction=agent_configs["search"]["instruction"],
            tools=[google_search]
        )
        root_tutor_agent_instance = Agent(
            name="tutor_agent_tool_router", model=MODEL_GEMINI_FLASH,
            instruction=agent_configs["tutor"]["instruction"],
            tools=[
                agent_tool.AgentTool(agent=math_agent_instance),
                agent_tool.AgentTool(agent=spanish_agent_instance),
                agent_tool.AgentTool(agent=search_agent_instance),
            ],
            before_model_callback=block_keyword_guardrail,
        )
        logger.info("Agents created successfully.")
    except Exception as e:
        logger.error(f"Fatal Error creating agents: {e}", exc_info=True)
        raise # Propagate error to stop app startup if agents fail

# --- ADK Runner and Session Service ---
# Using InMemorySessionService as per the tutorial for now
session_service = InMemorySessionService()
logger.info("ADK Session Service initialized with InMemorySessionService.")

runner_instance: Optional[Runner] = None

def get_runner() -> Runner:
    global runner_instance
    if runner_instance is None:
        if root_tutor_agent_instance is None:
            create_agents() # Ensure agents are created
        runner_instance = Runner(
            agent=root_tutor_agent_instance,
            app_name="fastapi_adk_tutor_app", # Consistent app name
            session_service=session_service
        )
        logger.info("ADK Runner initialized.")
    return runner_instance

# --- FastAPI App Setup ---
app = FastAPI(title="ADK Agent Backend")

@app.on_event("startup")
async def startup_event():
    create_agents() # Create agents when FastAPI starts
    get_runner()    # Initialize runner

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "fastapi_user" # Default user_id
    session_id: Optional[str] = None # If None, a new one will be created/retrieved

async def event_stream_generator(user_id: str, session_id: str, query: str) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events from ADK runner."""
    current_runner = get_runner()
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
    
    logger.info(f"Running agent for user_id='{user_id}', session_id='{session_id}', query='{query}'")
    
    try:
        async for event in current_runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Serialize the event to JSON. ADK events should have a to_dict() or similar method.
            # For now, let's try a simple representation.
            # A more robust solution would involve a proper Pydantic model for Event serialization.
            event_data = {
                "event_type": event.__class__.__name__,
                "author": getattr(event, 'author', None),
                "is_final_response": event.is_final_response(),
                "content_text": None,
                "tool_calls": None,
                "tool_responses": None,
                "error_message": getattr(event, 'error_message', None)
            }
            if event.content and event.content.parts:
                text_parts = [getattr(part, 'text', '') for part in event.content.parts if hasattr(part, 'text')]
                event_data["content_text"] = " ".join(filter(None, text_parts))

                # Ensure function_call is not None before trying to access attributes
                tool_call_parts = [part.function_call for part in event.content.parts if hasattr(part, 'function_call') and part.function_call is not None]
                if tool_call_parts:
                    event_data["tool_calls"] = [{"name": tc.name, "args": tc.args} for tc in tool_call_parts]
                
                # Ensure function_response is not None before trying to access attributes
                tool_response_parts = [part.function_response for part in event.content.parts if hasattr(part, 'function_response') and part.function_response is not None]
                if tool_response_parts:
                     event_data["tool_responses"] = [{"name": tr.name, "response": tr.response} for tr in tool_response_parts]


            yield f"data: {event.json()}\n\n" # ADK Event objects have a .json() method
            await asyncio.sleep(0.01) # Small sleep to allow other tasks, if any
    except Exception as e:
        logger.error(f"Error during agent execution or event streaming: {e}", exc_info=True)
        error_event_data = {"event_type": "ErrorEvent", "error_message": str(e)}
        import json # ensure json is imported
        yield f"data: {json.dumps(error_event_data)}\n\n"


@app.post("/chat_sse")
async def chat_with_agent_sse(chat_request: ChatRequest, request: Request):
    user_id = chat_request.user_id
    # If session_id is not provided, create a new one or use a default logic
    # For testing, let's ensure a session_id is always present
    session_id = chat_request.session_id if chat_request.session_id else str(uuid.uuid4())
    
    logger.info(f"Received request for /chat_sse: user_id='{user_id}', session_id='{session_id}'")

    # Ensure the session exists or is created by ADK
    # For InMemorySessionService, first try to get the session.
    # If it doesn't exist, create it.
    try:
        current_session = session_service.get_session(
            app_name=get_runner().app_name,
            user_id=user_id,
            session_id=session_id
            # Removed create_if_not_exists
        )
        if not current_session:
            logger.info(f"Session not found for app='{get_runner().app_name}', user='{user_id}', session='{session_id}'. Creating new session.")
            current_session = session_service.create_session(
                 app_name=get_runner().app_name,
                 user_id=user_id,
                 session_id=session_id
            )
        # Use the 'session_id' variable directly for logging, as current_session object might not have a .session_id attribute
        logger.info(f"Using ADK session (ID: {session_id}) with state: {current_session.state}")
    except Exception as e:
        logger.error(f"Error getting or creating session with InMemorySessionService: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize session with InMemorySessionService")

    return StreamingResponse(
        event_stream_generator(user_id, session_id, chat_request.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not set. Please set it before running.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
