import vertexai
# No longer need datetime/ZoneInfo from the old script
# import datetime
# from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool # <-- Import AgentTool
from vertexai.preview import reasoning_engines
from vertexai import agent_engines # Make sure this import is present
import traceback # For error printing
import time # For delays

# --- START INITIALIZATION ---
# Use your actual Project ID, Location, and Bucket
PROJECT_ID = "the-ai-lifestyle"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://deploy-ai-agent-ail"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)
print(f"Vertex AI Initialized: Project={PROJECT_ID}, Location={LOCATION}, Staging Bucket={STAGING_BUCKET}")
# --- END INITIALIZATION ---

# --- START TOOL/SUB-AGENT DEFINITIONS ---

# Instead of function tools, we define the translator agents
# 1. Spanish Translator Agent
spanish_translator_agent = Agent(
    model="gemini-1.5-flash",
    name="spanish_translator",
    description="Translates given text accurately into Spanish.",
    instruction="""You are an expert translator. Your sole task is to translate the user's provided text into Spanish. Provide ONLY the Spanish translation, without any additional commentary, prefixes, or explanations.""",
    tools=[],
)
print("Spanish Translator Agent defined.")

# 2. French Translator Agent
french_translator_agent = Agent(
    model="gemini-1.5-flash",
    name="french_translator",
    description="Translates given text accurately into French.",
    instruction="""You are an expert translator. Your sole task is to translate the user's provided text into French. Provide ONLY the French translation, without any additional commentary, prefixes, or explanations.""",
    tools=[],
)
print("French Translator Agent defined.")

# 3. Old English Translator Agent
old_english_translator_agent = Agent(
    model="gemini-1.5-flash",
    name="old_english_translator",
    description="Translates given text into Old English (Anglo-Saxon).",
    instruction="""You are an expert linguist specializing in Old English (Anglo-Saxon, approximately 450-1150 AD). Your sole task is to translate the user's provided modern English text into Old English. Use the closest appropriate Old English phrasing. Provide ONLY the Old English translation, without any additional commentary, prefixes, or explanations.""",
    tools=[],
)
print("Old English Translator Agent defined.")

# --- END TOOL/SUB-AGENT DEFINITIONS ---

# --- START ROOT AGENT DEFINITION ---

# Create AgentTool instances and set skip_summarization attribute
spanish_tool = AgentTool(agent=spanish_translator_agent)
try: spanish_tool.skip_summarization = True; print("Successfully set spanish_tool.skip_summarization = True")
except AttributeError: print("Warning: Could not set spanish_tool.skip_summarization attribute.")

french_tool = AgentTool(agent=french_translator_agent)
try: french_tool.skip_summarization = True; print("Successfully set french_tool.skip_summarization = True")
except AttributeError: print("Warning: Could not set french_tool.skip_summarization attribute.")

old_english_tool = AgentTool(agent=old_english_translator_agent)
try: old_english_tool.skip_summarization = True; print("Successfully set old_english_tool.skip_summarization = True")
except AttributeError: print("Warning: Could not set old_english_tool.skip_summarization attribute.")

# Define the Root Agent that uses the AgentTools
root_translator_agent = Agent(
    name="translation_dispatcher_agent", # Use hyphens if preferred
    model="gemini-2.0-flash", # Root agent model
    description=(
        "Dispatches translation requests to specialized agents for Spanish, French, or Old English."
    ),
    instruction=(
        """You are a helpful assistant that routes translation requests. Analyze the user's request to identify the target language (Spanish, French, or Old English) and the text to be translated. Use the appropriate tool: 'spanish_translator', 'french_translator', or 'old_english_translator'. After getting the translation from the tool, present ONLY the translated text back to the user. Do not add introductory phrases. If the target language is unclear or not supported, state that you can only translate to Spanish, French, or Old English."""
    ),
    # Pass the configured AgentTool instances
    tools=[spanish_tool, french_tool, old_english_tool],
)
print("Root Translator Agent defined.")
# --- END ROOT AGENT DEFINITION ---

# --- START PREPARE FOR DEPLOYMENT ---
# Wrap the *root* agent in AdkApp
app = reasoning_engines.AdkApp(
    agent=root_translator_agent,
    enable_tracing=True, # Enable tracing for debugging in Cloud Console
)
print("Root Translator Agent wrapped in AdkApp.")
# --- END PREPARE FOR DEPLOYMENT ---

# # --- START LOCAL TEST ---
# print("\n--- Starting Local Test ---")
# try:
#     local_session = app.create_session(user_id="local_translator_user")
#     print(f"Local session created: {local_session.id}")

#     test_queries = [
#         "Translate 'good morning my friend' to Spanish",
#         "How do you say 'where is the library?' in French?",
#         "Put 'The quick brown fox jumps over the lazy dog' into Old English",
#         "Can you translate 'water' to German?"
#     ]

#     for query in test_queries:
#         print(f"\n>>> Streaming query locally: '{query}'")
#         local_response_stream = app.stream_query(
#             user_id="local_translator_user", session_id=local_session.id, message=query,
#         )

#         # Use the dictionary parsing logic for events
#         final_text_response = ""
#         event_count = 0
#         for event in local_response_stream:
#             event_count += 1
#             current_text = ""
#             try:
#                 if isinstance(event, dict) and 'content' in event and 'parts' in event['content'] and isinstance(event['content']['parts'], list) and len(event['content']['parts']) > 0:
#                     first_part = event['content']['parts'][0]
#                     if 'text' in first_part:
#                         current_text = first_part.get('text', '')
#                     elif 'function_response' in first_part:
#                         function_response_data = first_part.get('function_response', {})
#                         response_data = function_response_data.get('response', {})
#                         current_text = response_data.get('result', '')
#             except Exception as e:
#                 print(f"[L_Event {event_count}] Warning: Error processing event structure - {e}")

#             if current_text:
#                 cleaned_text = current_text.strip()
#                 if cleaned_text:
#                    final_text_response = cleaned_text

#         print(f"<<< Local Final Extracted Response for '{query}': {final_text_response}")

# except Exception as e:
#     print(f"\nError during local test: {e}")
#     traceback.print_exc()
# finally:
#     time.sleep(1)
# print("--- Finished Local Test ---\n")
# # --- END LOCAL TEST ---


# --- START DEPLOYMENT ---
print("--- Starting Deployment to Agent Engine ---")
print("This may take several minutes...")
remote_app = None # Initialize
try:
    remote_app = agent_engines.create(
        # Pass the root agent
        agent_engine=root_translator_agent,
        requirements=[
            # CORRECTED requirements list
            "google-cloud-aiplatform[adk,agent_engines]>=1.48.0",
            # Add other pip dependencies if sub-agents needed them (none here)
        ],
        # Updated display name and description
        display_name="Multi-Language Translator Agent",
        description="Agent Engine that translates text to Spanish, French, or Old English using sub-agents."
    )
    print(f"Deployment successful! Remote App Resource Name: {remote_app.resource_name}")
    print("IMPORTANT: Grant IAM permissions before testing remotely.")
    print("It might take a few minutes for permissions and the agent engine to become fully available.")
except Exception as e:
    print(f"Error during deployment: {e}")
    traceback.print_exc()
    remote_app = None # Ensure remote_app is None if deployment fails
print("--- Finished Deployment ---")
# --- END DEPLOYMENT ---


# --- START REMOTE TEST ---
if remote_app:
    print("\n--- Starting Remote Test ---")
    print("Waiting a few seconds before remote test...")
    time.sleep(10)
    try:
        print("Creating remote session...")
        remote_session_info = remote_app.create_session(user_id="remote_translator_user")
        remote_session_id = remote_session_info["id"] # Extract the ID
        print(f"Remote session created: {remote_session_id}")

        # Use translation queries for remote test
        remote_test_queries = [
             "Translate 'the weather is nice today' into French",
             "In Spanish, how do you say 'I need help'?",
             "Render 'winter is coming' in Old English",
             "Say 'thank you' in German",
        ]

        for query in remote_test_queries:
             print(f"\n>>> Streaming query remotely: '{query}'")
             remote_response_stream = remote_app.stream_query(
                 user_id="remote_translator_user",
                 session_id=remote_session_id,
                 message=query,
             )

             # Use the dictionary parsing logic for remote events too
             remote_final_text_response = ""
             event_count = 0
             for event in remote_response_stream:
                 event_count += 1
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
                      print(f"[R_Event {event_count}] Warning: Error processing event structure - {e}")

                 if current_text:
                     cleaned_text = current_text.strip()
                     if cleaned_text:
                         remote_final_text_response = cleaned_text

             print(f"<<< Remote Final Extracted Response for '{query}': {remote_final_text_response}")

    except Exception as e:
        print(f"\nError during remote test: {e}")
        traceback.print_exc()
        print("Ensure IAM permissions were granted and have propagated.")
    finally:
        time.sleep(1)
    print("--- Finished Remote Test ---")
else:
    print("\nSkipping remote test because deployment failed or was not attempted.")
# --- END REMOTE TEST ---

print("\nScript finished.")