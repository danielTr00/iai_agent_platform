# IAI Agent Platform

## Repository Structure

This repository contains the IAI Agent Platform codebase with the following structure:

- **config/**: Configuration files and environment variables
- **files/**: Storage for various data files used by the platform
- **interface/**: User interface components
  - `research_agent_interface.py`: Interface for the research agent functionality
- **scenario_coordination/**: Core functionality modules
  - `agents/`: Agent implementation files
    - `research_agent_class.py`: Implementation of the research agent
  - `custom_tools/`: Custom tool implementations
    - `rag_tool.py`: Retrieval-Augmented Generation tool
  - `utils/`: Utility functions and helpers
- **vector_db/**: Vector database storage for embeddings

## Technologies

The platform is built using the following key technologies:

- **LangChain**: Framework for developing applications powered by language models
- **LangGraph**: Library for building stateful, multi-actor applications with LLMs
- **Vector Databases**: For storing and retrieving embeddings
- **Multiple LLM Support**: Compatible with various providers (Groq, OpenAI, Anthropic, etc.)

## Dependencies

The main dependencies include:
- Python 3.9+
- LangChain and LangGraph libraries
- Various LLM provider SDKs (Groq, OpenAI, Anthropic, etc.)
- Vector database libraries
- Web search tools

See `requirements.txt` for a complete list of dependencies.

## Prerequisites

Before setting up the platform, you'll need:

1. **Docker**: Required to run the containerized version of the application
   - [Install Docker](https://docs.docker.com/get-started/get-docker/)

2. **Ollama** (Optional): For running local embedding models and LLMs
   - [Install Ollama](https://ollama.com/download)

3. **Tavily API Key** (Required for web search functionality):
   - Register for a free tier account at [Tavily](https://tavily.com/)
   - Create an API key in your Tavily dashboard

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
Create a `.env` file in the config directory with your API keys:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
# Add other API keys as needed
```

## Research Agent Interface

The platform provides a Gradio-based web interface for interacting with the research agent. The interface allows users to ask questions, upload documents for knowledge base creation, and customize agent settings.

### Interface Components

The interface is organized into two main tabs:

1. **Chat Tab**:
   - Chat interface for asking questions and receiving researched answers
   - Examples of common questions
   - Option to clear chat history

2. **Settings Tab**:
   - Configuration options for the research agent
   - Model and platform selection
   - API key management
   - Document upload for knowledge base creation

### Agent Parameters

The research agent interface accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform` | string | "groq" | LLM platform to use (groq, ollama, openai, huggingface, anthropic, azure, google, xai) |
| `model` | string | "llama-3.1-8b-instant" | The specific model to use for the selected platform |
| `ollama_base_url` | string | "http://localhost:11434" | Base URL for Ollama API when using local models |
| `max_searches` | integer | 2 | Maximum number of web searches to perform per query |
| `system_prompt` | string | None | Custom system prompt to override the default |
| `llm_api_key` | string | None | API key for the selected LLM platform |
| `tavily_api_key` | string | None | API key for Tavily search integration |
| `embedding_model_name` | string | "nomic-embed-text" | Model to use for text embeddings |
| `history_file` | string | None | Path to store conversation history |
| `agent_name` | string | "Research Agent" | Name of the agent instance |
| `vector_db_path` | string | None | Path to store vector database files |

### Supported Models

The interface supports multiple LLM platforms with the following models:

- **Groq**: llama-3.1-8b-instant, mixtral-8x7b-32768, llama-3.3-70b-versatile, qwen-2.5-32b, deepseek-r1-distill-qwen-32b
- **Ollama**: llama3.2, granite3.1-moe:1b, nemotron-mini
- **OpenAI**: gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Azure**: gpt-4, gpt-35-turbo
- **Google**: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
- **XAI**: grok-1

### Embedding Models

The following embedding models are supported for vector database operations:

- nomic-embed-text (default)
- snowflake-arctic-embed:110m
- all-minilm

### Knowledge Base Management

The interface allows users to:

1. Upload documents (PDF, DOCX, TXT) to create a knowledge base
2. View currently tracked documents
3. Export and import agent settings
4. Clear conversation history

### Running the Interface

To launch the research agent interface:

```python
from interface.research_agent_interface import ResearchAgentInterface

# Create and launch the interface
interface = ResearchAgentInterface(
    platform="groq",
    model="llama-3.1-8b-instant",
    max_searches=2
)

# Launch the interface on localhost
interface.launch()
```

## Current State

The repository currently includes:
- Research agent implementation with RAG capabilities
- Vector database integration
- Custom tools for enhanced functionality
- Interface components for user interaction

## License

This project is licensed under the MIT License.
