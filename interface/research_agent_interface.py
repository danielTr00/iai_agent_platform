import gradio as gr
import sys
import os
import shutil
from pathlib import Path
import traceback
import docx2txt  
import json  
from datetime import datetime

# Add the parent directory to the path so we can import the research agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_coordination.agents.research_agent_class import BasicAgent

class ResearchAgentInterface:
    """Interface for the Research Agent using Gradio"""
    
    def __init__(self, 
                 platform="groq", 
                 model="llama-3.1-8b-instant", 
                 ollama_base_url="http://localhost:11434",
                 max_searches=2,
                 system_prompt=None,
                 llm_api_key=None,
                 tavily_api_key=None,
                 embedding_model_name="nomic-embed-text",
                 history_file=None,
                 agent_name="Research Agent",
                 vector_db_path=None):
        """Initialize the interface with a research agent"""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set up chat history directory
        self.chat_history_dir = os.path.join(
            project_root,
            "chat_history"
        )
        os.makedirs(self.chat_history_dir, exist_ok=True)
        
        # Set default history file path
        self.history_file = history_file or os.path.join(
            self.chat_history_dir,
            "conversations.json"
        )
        
        self.agent = BasicAgent(
            platform=platform,
            model=model,
            ollama_base_url=ollama_base_url,
            max_searches=max_searches,
            system_prompt=system_prompt,
            llm_api_key=llm_api_key,
            tavily_api_key=tavily_api_key,
            embedding_model_name=embedding_model_name,
            history_file=self.history_file,
            vector_db_path=vector_db_path,
            name=agent_name
        )
        
        # Set up file storage directory
        self.files_dir = "/Users/Uni/Desktop/Coding/studienarbeit/venv/Scenario_Agency/files"
        os.makedirs(self.files_dir, exist_ok=True)
        
        # Available embedding models
        self.available_embedding_models = ["nomic-embed-text", "snowflake-arctic-embed:110m" , "all-minilm"]
        
        # Define available models for each platform
        self.platform_models = {
            "groq": ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "qwen-2.5-32b", "deepseek-r1-distill-qwen-32b"],
            "ollama": ["llama3.2", "granite3.1-moe:1b", "nemotron-mini"],
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "azure": ["gpt-4", "gpt-35-turbo"],
            "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "xai": ["grok-1"],
        }
        
        # Create the interface
        self.interface = self._build_interface()
    
    def _process_query(self, message, history):
        """Process a user query and return the response"""
        try:
            # Run the research agent with the user's query
            response = self.agent.run(message)
            
            # Update history with the new message pair
            history = history + [[message, response]]
            
            return history
        except Exception as e:
            # Handle any errors that occur during processing
            error_message = f"Error processing your request: {str(e)}"
            if "Service Unavailable" in str(e):
                error_message += "\n\nThe LLM service is currently unavailable. Try switching to a different platform like 'ollama' in the settings."
            
            # Update history with the error message
            history = history + [[message, error_message]]
            
            return history
    
    def _load_chat_history(self):
        """Load chat history from file for display in the interface"""
        try:
            if not os.path.exists(self.history_file):
                return []
                
            with open(self.history_file, 'r') as f:
                history_data = json.load(f)
            
            # Check if there are any conversations
            if not history_data.get("conversations"):
                return []
            
            # Get the most recent conversation
            last_conversation = history_data["conversations"][-1]
            
            # Format messages for Gradio chat interface
            chat_history = []
            messages = last_conversation.get("messages", [])
            
            # Group messages into pairs (user, assistant)
            for i in range(0, len(messages), 2):
                if i+1 < len(messages):  # Ensure we have both user and assistant messages
                    user_msg = messages[i]["content"]
                    assistant_msg = messages[i+1]["content"]
                    chat_history.append([user_msg, assistant_msg])
            
            print(f"Loaded {len(chat_history)} message pairs from history")
            return chat_history
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading chat history: {str(e)}")
            return []
    
    def _clear_chat_history(self):
        """Clear the chat history file and return empty history"""
        try:
            # Create empty history structure
            empty_history = {"conversations": []}
            
            # Write to file
            with open(self.history_file, 'w') as f:
                json.dump(empty_history, f)
                
            return "Chat history cleared successfully."
        except Exception as e:
            return f"Error clearing chat history: {str(e)}"
    
    def _handle_file_upload(self, files):
        """Handle file uploads and add them to the vector store"""
        if not files:
            return "No files uploaded."
        
        results = []
        
        # Get list of already tracked documents
        tracked_documents = self.agent.rag_tool.get_tracked_documents()
        if tracked_documents:
            results.append(f"Currently tracked documents: {', '.join(tracked_documents)}")
        
        for file in files:
            try:
                # Get the filename
                filename = os.path.basename(file.name)
                
                # Check if the file is already tracked
                if self.agent.rag_tool.is_document_tracked(filename):
                    results.append(f"Skipping {filename} as it's already in the knowledge base.")
                    continue
                
                # Create the destination path
                dest_path = os.path.join(self.files_dir, filename)
                
                # Copy the file to the destination
                shutil.copy(file.name, dest_path)
                
                # Add the file to the vector store
                try:
                    self.agent.rag_tool.add_documents([dest_path])
                    results.append(f"Successfully added {filename} to knowledge base.")
                except Exception as e:
                    # If the current embedding model fails, try the fallback
                    error_msg = str(e)
                    results.append(f"Error with primary embedding model: {error_msg}")
                    
                    # Try with a fallback embedding model
                    try:
                        current_model = self.agent.rag_tool.embedding_model.model
                        fallback_model = "nomic-embed-text" if current_model != "nomic-embed-text" else "snowflake-arctic-embed:110m"
                        
                        results.append(f"Trying with fallback embedding model: {fallback_model}")
                        
                        # Recreate the agent with the fallback model
                        self.agent = BasicAgent(
                            platform=self.agent.platform,
                            model=self.agent.model,
                            ollama_base_url=self.agent.ollama_base_url,
                            max_searches=self.agent.max_searches,
                            system_prompt=self.agent.system_prompt,
                            embedding_model_name=fallback_model
                        )
                        
                        # Try adding the document again
                        self.agent.rag_tool.add_documents([dest_path])
                        results.append(f"Successfully added {filename} to knowledge base using fallback model.")
                    except Exception as fallback_error:
                        results.append(f"Error with fallback embedding model: {str(fallback_error)}")
                        results.append(f"Could not process {filename}. Please try a different file or embedding model.")
            except Exception as e:
                results.append(f"Error processing {os.path.basename(file.name)}: {str(e)}")
                traceback.print_exc()
        
        return "\n".join(results)
    
    def _export_settings(self):
        """Export agent settings as a dictionary"""
        # Get the original system prompt from the interface instead of the agent
        # This ensures we only save the clean system prompt without conversation history
        system_prompt_value = ""
        for component in self.interface.blocks.values():
            if hasattr(component, "label") and component.label == "System Prompt":
                system_prompt_value = component.value
                break
        
        # No default value - use exactly what's in the field
        
        settings = {
            "name": self.agent.name,
            "platform": self.agent.platform,
            "model": self.agent.model,
            "ollama_base_url": self.agent.ollama_base_url,
            "max_searches": self.agent.max_searches,
            "system_prompt": system_prompt_value,  # Use exactly what's in the field
            "embedding_model_name": self.agent.rag_tool.embedding_model.model,
            "vector_db_path": self.agent.rag_tool.vector_db_path
        }
        
        # Only include API keys if they exist and are not None
        if hasattr(self.agent, 'llm_api_key') and self.agent.llm_api_key:
            settings["llm_api_key"] = self.agent.llm_api_key
        if hasattr(self.agent, 'tavily_api_key') and self.agent.tavily_api_key:
            settings["tavily_api_key"] = self.agent.tavily_api_key
            
        return settings
    
    def _import_settings(self, file):
        """Import agent settings from a JSON file"""
        try:
            # Read the JSON file
            with open(file.name, 'r') as f:
                settings = json.load(f)
            
            # Extract settings
            name_val = settings.get("name", "Research Agent")
            platform_val = settings.get("platform", "ollama")
            model_val = settings.get("model", "llama3.2")
            ollama_url_val = settings.get("ollama_base_url", "http://localhost:11434")
            max_searches_val = int(settings.get("max_searches", 2))
            
            # Clean the system prompt - extract only the base instructions
            system_prompt_val = settings.get("system_prompt", "")
            
            # Look for common markers of conversation history
            conversation_markers = [
                "Your last exchange with the user:",
                "User:",
                "You:",
                "Human:",
                "Assistant:"
            ]
            
            # Find the first occurrence of any marker
            first_marker_pos = len(system_prompt_val)
            for marker in conversation_markers:
                pos = system_prompt_val.find(marker)
                if pos > 0 and pos < first_marker_pos:
                    first_marker_pos = pos
            
            # Only keep the part before any conversation history
            if first_marker_pos < len(system_prompt_val):
                system_prompt_val = system_prompt_val[:first_marker_pos].strip()
            
            embedding_model_val = settings.get("embedding_model_name", "nomic-embed-text")
            vector_db_path_val = settings.get("vector_db_path", "")
            
            # Create a new agent with the imported settings
            self.agent = BasicAgent(
                platform=platform_val,
                model=model_val,
                ollama_base_url=ollama_url_val,
                max_searches=max_searches_val,
                system_prompt=system_prompt_val,
                embedding_model_name=embedding_model_val,
                vector_db_path=vector_db_path_val,
                name=name_val
                # API keys are not included in the export/import for security
            )
            
            result_message = f"Settings imported successfully from {os.path.basename(file.name)}"
            
            # Update model dropdown based on platform
            model_choices = self.platform_models.get(platform_val, [])
            model_dropdown_update = gr.update(choices=model_choices, value=model_val)
            
            return result_message, name_val, platform_val, model_dropdown_update, model_val, ollama_url_val, max_searches_val, system_prompt_val, embedding_model_val, vector_db_path_val
        except Exception as e:
            error_message = f"Error importing settings: {str(e)}"
            return error_message, None, None, None, None, None, None, None, None, None
    
    def _build_interface(self):
        """Build the Gradio interface"""
        # Create settings components
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("# Research Agent")
                    gr.Markdown("Ask questions and get researched answers with citations")
                with gr.Column(scale=1, min_width=200):
                    with gr.Row():
                        download_button = gr.Button("📥 Download Settings", size="sm")
                        upload_settings_button = gr.Button("📤 Upload Settings", size="sm")
                    settings_file_upload = gr.File(
                        file_types=[".json"],
                        file_count="single",
                        label="Upload Settings File",
                        visible=False
                    )
            
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    # Load existing chat history
                    initial_chat_history = self._load_chat_history()
                    
                    with gr.Row():
                        clear_history_button = gr.Button("🗑️ Clear Chat History", size="sm")
                    
                    # Create a standard Chatbot component instead of ChatInterface
                    # This gives us more control over the initial state
                    chatbot = gr.Chatbot(
                        value=initial_chat_history,
                        label="Research Agent Chat",
                        height=500
                    )
                    
                    msg = gr.Textbox(
                        placeholder="Ask me any question, and I'll research the answer for you.",
                        container=True,
                        scale=7
                    )
                    
                    submit = gr.Button("Submit", scale=1)
                    
                    # Set up examples
                    examples = gr.Examples(
                        examples=[
                            "What was the temperature in Berlin yesterday?",
                            "Who won the last World Cup in football?",
                            "What are the latest developments in AI?",
                            "What is the population of Tokyo?",
                        ],
                        inputs=msg
                    )
                    
                    # Connect the components
                    submit.click(
                        fn=self._process_query,
                        inputs=[msg, chatbot],
                        outputs=[chatbot],
                        queue=True
                    ).then(
                        fn=lambda: "",
                        inputs=None,
                        outputs=msg
                    )
                    
                    msg.submit(
                        fn=self._process_query,
                        inputs=[msg, chatbot],
                        outputs=[chatbot],
                        queue=True
                    ).then(
                        fn=lambda: "",
                        inputs=None,
                        outputs=msg
                    )
                    
                    # Connect clear history button
                    clear_history_result = gr.Textbox(visible=False)
                    clear_history_button.click(
                        fn=self._clear_chat_history,
                        inputs=None,
                        outputs=clear_history_result
                    ).then(
                        fn=lambda: None,  # This is a hack to refresh the page
                        inputs=None,
                        outputs=None,
                        js="() => { window.location.reload(); }"
                    )
                
                with gr.TabItem("Settings"):
                    with gr.Row():
                        platform = gr.Dropdown(
                            choices=list(self.platform_models.keys()),
                            value=self.agent.platform,
                            label="Platform",
                            info="KI-Plattform, die für die Anfragen verwendet wird"
                        )
                        
                        model_dropdown = gr.Dropdown(
                            choices=self.platform_models.get(self.agent.platform, []),
                            value=self.agent.model,
                            label="Model",
                            info="KI-Modell, das für die Anfragen verwendet wird"
                        )
                    
                    with gr.Row():
                        agent_name = gr.Textbox(
                            value=self.agent.name,
                            label="Agent Name",
                            placeholder="Enter a name for your agent",
                            info="Custom name for your research agent"
                        )
                        
                        ollama_url = gr.Textbox(
                            value=self.agent.ollama_base_url,
                            label="Ollama Base URL (only used with Ollama platform)",
                            info="Die URL, unter der dein Ollama-Server erreichbar ist (Standard: http://localhost:11434)"
                        )
                        max_searches = gr.Slider(
                            minimum=2,
                            maximum=12,
                            value=self.agent.max_searches,
                            step=1,
                            label="Maximum Number of Searches",
                            info="Maximale Anzahl an Websuchen, die der Agent pro Anfrage durchführen darf"
                        )
                    
                    with gr.Row():
                        system_prompt = gr.Textbox(
                            value=self.agent.system_prompt if hasattr(self.agent, 'system_prompt') else None,
                            label="System Prompt (optional)",
                            placeholder="Enter custom system prompt or leave empty for default",
                            lines=3,
                            info="Anweisungen, die dem KI-Modell zu Beginn jeder Konversation gegeben werden"
                        )
                        
                    with gr.Row():
                        # Add file upload for system prompt import
                        prompt_file_upload = gr.File(
                            file_types=[".txt", ".docx"],
                            file_count="single",
                            label="Import System Prompt from File"
                        )
                        import_prompt_button = gr.Button("Import Prompt")
                    
                    with gr.Row():
                        llm_api_key = gr.Textbox(
                            value="",  # Don't show API key for security
                            label="LLM API Key (for Groq/OpenAI)",
                            placeholder="Enter API key or leave empty to use environment variable",
                            type="password",
                            info="API-Schlüssel für externe KI-Dienste wie Groq, OpenAI, etc."
                        )
                        tavily_api_key = gr.Textbox(
                            value="",  # Don't show API key for security
                            label="Tavily API Key",
                            placeholder="Enter API key or leave empty to use environment variable",
                            type="password",
                            info="API-Schlüssel für die Tavily-Suchmaschine, die für Websuchen verwendet wird"
                        )
                    
                    with gr.Row():
                        embedding_model = gr.Dropdown(
                            choices=self.available_embedding_models,
                            value=self.agent.rag_tool.embedding_model.model,
                            label="Embedding Model Name",
                            info="Modell zur Umwandlung von Dokumenten in Vektoren für die Ähnlichkeitssuche"
                        )
                    
                    with gr.Row():
                        vector_db_path = gr.Textbox(
                            label="Vector DB Path (optional)", 
                            value=self.agent.vector_db_path if hasattr(self.agent, 'vector_db_path') else "",
                            placeholder="/Users/Uni/Desktop/Coding/studienarbeit/venv/Scenario_Agency/vector_db/Research Agent_5345"
                        )
                    
                    # Add file upload section
                    gr.Markdown("## Upload Files to Knowledge Base")
                    gr.Markdown("Upload files to add to the agent's knowledge base. Supported formats: PDF, DOCX, PPTX, TXT")
                    
                    with gr.Row():
                        file_upload = gr.File(
                            file_types=[".pdf", ".docx", ".pptx", ".txt"],
                            file_count="multiple",
                            label="Upload Files"
                        )
                                    
                    # Single apply button for all settings
                    apply_button = gr.Button("Apply Settings")
                    result = gr.Textbox(label="Result", lines=3)
                    
                    # Function to extract text from uploaded file and update system prompt
                    def import_system_prompt(file):
                        if file is None:
                            return "No file uploaded.", None
                        
                        try:
                            file_path = file.name
                            file_ext = os.path.splitext(file_path)[1].lower()
                            
                            # Extract text based on file type
                            if file_ext == ".txt":
                                with open(file_path, "r", encoding="utf-8") as f:
                                    prompt_text = f.read()
                            elif file_ext == ".docx":
                                prompt_text = docx2txt.process(file_path)
                            else:
                                return f"Unsupported file type: {file_ext}", None
                            
                            return f"Successfully imported prompt from {os.path.basename(file_path)}", prompt_text
                        except Exception as e:
                            return f"Error importing prompt: {str(e)}", None
                    
                    # Connect the import button
                    import_prompt_button.click(
                        fn=import_system_prompt,
                        inputs=prompt_file_upload,
                        outputs=[result, system_prompt]
                    )
                    
                    # Combined function to update settings and handle file uploads
                    def update_all_settings(platform_val, model_val, ollama_url_val, max_searches_val, 
                                           system_prompt_val, llm_api_key_val, tavily_api_key_val, 
                                           embedding_model_val, agent_name_val, vector_db_path_val, files):
                        # Add timestamp of when apply was clicked
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        results = []
                        try:
                            # Update agent settings
                            llm_key = llm_api_key_val if llm_api_key_val.strip() else None
                            tavily_key = tavily_api_key_val if tavily_api_key_val.strip() else None
                            # Use the system prompt exactly as provided, even if empty
                            system_prompt_final = system_prompt_val
                            agent_name_final = agent_name_val if agent_name_val.strip() else "Research Agent"
                            vector_db_path_final = vector_db_path_val if vector_db_path_val.strip() else None
                            
                            self.agent = BasicAgent(
                                platform=platform_val,
                                model=model_val,
                                ollama_base_url=ollama_url_val,
                                max_searches=int(max_searches_val),
                                system_prompt=system_prompt_final,
                                llm_api_key=llm_key,
                                tavily_api_key=tavily_key,
                                embedding_model_name=embedding_model_val,
                                vector_db_path=vector_db_path_final,
                                name=agent_name_final
                            )
                            results.append("Settings updated successfully!")
                        except Exception as e:
                            results.append(f"Error updating settings: {str(e)}")
                            
                        # Process file uploads if any
                        if files and len(files) > 0:
                            for file in files:
                                try:
                                    # Get file extension
                                    filename = os.path.basename(file.name)
                                    file_ext = os.path.splitext(file.name)[1].lower()
                                    
                                    # Check if the file is already tracked
                                    if self.agent.rag_tool.is_document_tracked(filename):
                                        results.append(f"Document '{filename}' is already in the knowledge base, skipping...")
                                        continue
                                        
                                    # Create a temporary copy of the file
                                    temp_path = os.path.join(self.files_dir, filename)
                                    shutil.copy(file.name, temp_path)
                                    
                                    # Use the RAGTool to load and process the file
                                    self.agent.rag_tool.add_documents([temp_path])
                                    results.append(f"Added {filename} to knowledge base")
                                except Exception as e:
                                    results.append(f"Error adding document: {str(e)}")
                        
                        return "\n".join(results)
                    
                    # Function to update model choices based on platform selection
                    def update_model_choices(platform_val):
                        model_choices = self.platform_models.get(platform_val, [])
                        return gr.Dropdown(choices=model_choices, allow_custom_value=True)
                    
                    # Function to sync model dropdown and textbox
                    def sync_model_value(model_dropdown_val):
                        """Sync the model dropdown value to the model textbox"""
                        return model_dropdown_val
                    
                    # Connect platform dropdown to update model choices
                    platform.change(
                        fn=update_model_choices,
                        inputs=platform,
                        outputs=model_dropdown
                    )
                    
                    # Connect model dropdown to model textbox
                    model_dropdown.change(
                        fn=sync_model_value,
                        inputs=model_dropdown,
                        outputs=model_dropdown  # Changed from 'model' to 'model_dropdown'
                    )
                    
                    # Connect the apply button to the combined function
                    apply_button.click(
                        fn=update_all_settings,
                        inputs=[
                            platform, model_dropdown, ollama_url, max_searches, 
                            system_prompt, llm_api_key, tavily_api_key, 
                            embedding_model, agent_name, vector_db_path, file_upload
                        ],
                        outputs=result
                    )
                    
                    # Function to download settings as JSON
                    def download_settings():
                        import tempfile
                        
                        # Get the system prompt from the agent instead of the interface component
                        # This ensures we get the exact value that's being used
                        current_system_prompt = self.agent.system_prompt
                        
                        # Safely handle vector_db_path value
                        vector_db_path_value = None
                        if vector_db_path.value is not None:
                            vector_db_path_value = vector_db_path.value.strip() or None
                        
                        # Create settings dictionary with current values
                        settings = {
                            "name": self.agent.name,
                            "platform": self.agent.platform,
                            "model": self.agent.model,
                            "ollama_base_url": self.agent.ollama_base_url,
                            "max_searches": self.agent.max_searches,
                            "system_prompt": current_system_prompt,
                            "embedding_model_name": self.agent.rag_tool.embedding_model.model,
                            "vector_db_path": self.agent.rag_tool.vector_db_path
                        }
                        
                        # Create a temporary file
                        temp_dir = tempfile.gettempdir()
                        file_path = os.path.join(temp_dir, "research_agent_settings.json")
                        
                        # Write settings to the file
                        with open(file_path, "w") as f:
                            json.dump(settings, f, indent=2)
                            
                        return file_path
                    
                    # Function to trigger file upload dialog
                    def trigger_upload():
                        return gr.update(visible=True)
                    
                    # Connect download button
                    download_button.click(
                        fn=download_settings,
                        inputs=None,
                        outputs=gr.File()
                    )
                    
                    # Connect upload settings button to show file upload
                    upload_settings_button.click(
                        fn=trigger_upload,
                        inputs=None,
                        outputs=settings_file_upload
                    )
                    
                    # Function to handle file upload and deletion
                    def handle_file_change(file):
                        if file is None or not hasattr(file, 'name'):
                            # File was deleted or not provided
                            return "No settings file uploaded or file was deleted.", None, None, None, None, None, None, None, None, None
                        else:
                            # File was uploaded
                            return self._import_settings(file)
                    
                    # Connect settings file upload
                    settings_file_upload.change(
                        fn=handle_file_change,
                        inputs=settings_file_upload,
                        outputs=[
                            result,           # Result message
                            agent_name,       # Agent name
                            platform,         # Platform dropdown
                            model_dropdown,   # Model dropdown
                            model_dropdown,   # Model textbox
                            ollama_url,       # Ollama URL
                            max_searches,     # Max searches slider
                            system_prompt,    # System prompt
                            embedding_model,  # Embedding model dropdown
                            vector_db_path    # Vector DB Path
                        ]
                    )
            
        return interface
    
    def launch(self, **kwargs):
        """Launch the interface"""
        self.interface.launch(**kwargs)


# Run the interface if this file is executed directly
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config/app.env'))
    
    # Ensure docx2txt is installed
    try:
        import docx2txt
    except ImportError:
        import subprocess
        import sys
        print("Installing required package: docx2txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "docx2txt"])
        import docx2txt
    
    # Create and launch the interface
    def main():
        """Run the research agent interface"""
        interface = ResearchAgentInterface(
            platform="ollama" if os.path.exists("/usr/local/bin/ollama") else "groq",
            model="llama3.2" if os.path.exists("/usr/local/bin/ollama") else "llama-3.1-8b-instant",
            embedding_model_name="nomic-embed-text",  # Use nomic-embed-text as default since it's more reliable
            # API keys and system prompt will be loaded from environment variables if not provided
            agent_name="Research Agent"
        )
        interface.launch(share=False)
    
    main()
