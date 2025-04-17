# Google Agent Development Kit (ADK) Tutorial Examples

This repository contains the example code demonstrated in the YouTube tutorial "Building Powerful AI Agents with Google ADK: A Complete Step-by-Step Guide".

## Overview

These examples illustrate key concepts of the Google Agent Development Kit (ADK), a Python framework for building, evaluating, and deploying AI agents. They cover:

* Basic agent creation and tool definition
* Workflow orchestration using Sequential, Parallel, and Loop agents
* Advanced concepts like Agents-as-Tools and built-in tools (Search, Code Execution)
* Deployment to Google Cloud (Vertex AI Agent Engine)
* Building simple UIs for agents using Streamlit

## Prerequisites

* Python (3.9+ recommended)
* Virtual Environment (`venv`)
* Google Cloud Project (for deployment examples)
* API Keys (Google AI Gemini API Key) stored securely in `.env` files or environment variables

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/googleADK1.git
   cd googleADK1
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**
   * Create a `.env` file in the project root and add your API keys:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```
   * Some scripts may expect environment variables instead of `.env` files.

## Examples

### 1. Multi-Tool Agent Quickstart (`multi_tool_agent_quickstart/`)

* **File:** `agent.py`
* **Concept:** Demonstrates building your first simple agent (`weather_time_agent`) using basic Python functions (`get_weather`, `get_current_time`) as tools. Illustrates agent definition, instructions, and tool docstrings.
* **Corresponds to Tutorial:** Setup, Building Your First Agent, Defining Tools & Agent Logic.
* **To Run:** Typically run via `adk dev <path_to_agent_py>` or integrated into other scripts.

### 2. Core ADK Samples (`samples from adk documention/`)

These scripts showcase core ADK features, often run directly.

* **`sequentialagent.py`**: Demonstrates `SequentialAgent` for a code generation pipeline (Write -> Review -> Refactor). Uses `LlmAgent` and session state.
  * **Concept:** Sequential Workflow Orchestration.
  * **Corresponds to Tutorial:** SequentialAgent for Pipelines.
  * **To Run:** `python "samples from adk documention/sequentialagent.py"`

* **`parallelagent.py`**: Demonstrates `ParallelAgent` for concurrent web research on multiple topics using `LlmAgent` and the built-in `google_search` tool.
  * **Concept:** Parallel Workflow Orchestration, Built-in Tools.
  * **Corresponds to Tutorial:** ParallelAgent for Concurrent Tasks, Built-in Tools.
  * **To Run:** `python "samples from adk documention/parallelagent.py"`

* **`loopagent.py`**: Demonstrates `LoopAgent` for iterative document writing and refinement. Uses `LlmAgent` and session state.
  * **Concept:** Iterative Workflow Orchestration.
  * **Corresponds to Tutorial:** LoopAgent for Iterative Processes.
  * **To Run:** `python "samples from adk documention/loopagent.py"`

* **`deploy_agent.py`**: Demonstrates defining agents-as-tools (specialized translators) and deploying the root agent to **Vertex AI Agent Engine**. Requires Google Cloud setup.
  * **Concept:** Agents-as-Tools, Deployment.
  * **Corresponds to Tutorial:** Agents-as-Tools, Deployment Option 1: Vertex AI Agent Engine.
  * **To Run:** `python "samples from adk documention/deploy_agent.py"` (Requires GCP configuration)

### 3. Streamlit UI Examples (`strealit use cases/`)

These examples wrap ADK agents in Streamlit web interfaces. Run them using `streamlit run <path_to_script.py>`.

* **`streamlit_code_gen_and_enhancer_sequential_agent.py`**: UI for the sequential code pipeline, including code execution using `built_in_code_execution`.

* **`streamlit_research_parallel_agent.py`**: UI for parallel research, including dynamic subtopic generation and report synthesis.

* **`streamlit_pitch_optimizer_loop_agent.py`**: UI for the iterative pitch refinement loop (Critic -> Writer).

* **`streamlit_multi_agents_agents_as_tools.py`**: UI demonstrating a multi-agent "Tutor Bot" where specialists (Math, Spanish, Search) are used as *tools* (`AgentTool`). Includes configurable instructions and guardrails.

* **`streamlit_multi_agents_subagents.py`**: UI demonstrating a *hybrid* multi-agent "Tutor Bot" using `sub_agents` for Math/Spanish and `AgentTool` for Search.

* **`streamlit_agent_engine_remote_agent.py`**: UI to connect to and interact with an agent already *deployed* on Vertex AI Agent Engine (like the one from `deploy_agent.py`).

### 4. Agent Management CLI (`agent_management_cli.py`)

* **Concept:** A command-line tool to manage agents deployed on Vertex AI (List, Update, Delete).
* **Corresponds to Tutorial:** Deployment (Management aspect).
* **To Run:** `python agent_management_cli.py` (Requires GCP configuration)

## Resources

* **YouTube Tutorial:** [Building Powerful AI Agents with Google ADK: A Complete Step-by-Step Guide](https://www.youtube.com/your_video_link)
* **Google ADK Documentation:** https://developers.google.com/vertex-ai/docs/agent-builder/agents/adk/overview
* **Google ADK GitHub:** https://github.com/google/agent-development-kit
* **ADK Samples:** https://github.com/google/agent-development-kit/tree/main/samples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
