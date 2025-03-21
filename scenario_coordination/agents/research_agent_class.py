from dotenv import load_dotenv
import os
import sys
from datetime import datetime
from typing import TypedDict, List, Optional, Literal
import time
import json

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END
from langgraph.pregel import RetryPolicy
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
# LLM imports
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


from scenario_coordination.utils import agent_logger
from scenario_coordination.custom_tools.rag_tool import RAGTool, DEFAULT_VECTORSTORE_DIR

class OverallState(TypedDict):
    messages: List[BaseMessage]
    current_knowledge: List[str]
    needs_additional_search: List[bool]


class BasicAgent:
    def __init__(
        self,
        name: str = "BasicAgent", 
        platform: Literal["groq", "ollama", "openai", "huggingface", "anthropic", "azure", "google", "xai"] = "groq",
        model: str = "llama-3.1-8b-instant",
        llm_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        max_searches: int = 3,
        log_dir: Optional[str] = None,
        embedding_model_name: str = "all-minilm:latest",
        history_file: Optional[str] = None,
        vector_db_path: Optional[str] = None
    ):
        # Initialize logger
        self.logger = agent_logger.AgentLogger(log_dir=log_dir)
        self.name = name
        
        # Get project root directory (two levels up from the agents directory)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set history file path
        self.history_file = history_file or os.path.join(
            project_root,
            "chat_history",
            "conversations.json"
        )
        
        # Ensure chat history directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        # Initialize chat history if it doesn't exist
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump({"conversations": []}, f)
        
        # Set API keys from parameters or environment
        if llm_api_key:
            if platform == "groq":
                os.environ["GROQ_API_KEY"] = llm_api_key
            elif platform == "openai" or platform == "azure":
                os.environ["OPENAI_API_KEY"] = llm_api_key
            elif platform == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = llm_api_key
            elif platform == "huggingface":
                os.environ["HUGGINGFACE_API_KEY"] = llm_api_key
            elif platform == "google":
                os.environ["GOOGLE_API_KEY"] = llm_api_key
            elif platform == "xai":
                os.environ["XAI_API_KEY"] = llm_api_key
        
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        # Store configuration
        self.platform = platform
        self.model = model
        self.max_searches = max_searches
        self.ollama_base_url = ollama_base_url
        
        # Set vector database path
        self.vector_db_path = vector_db_path
        
        # Get current date for time-based queries
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Set default system prompt if none provided
        if system_prompt is None or system_prompt == "":
            self.system_prompt = f"You are a helpful agent with tools that just calls tools when there is no other way. Please use as few internet searches as possible and as many retrievals from your knowledge as possible. Important: today's date: {self.current_date}"
        else:
            self.system_prompt = system_prompt
            
        # Log initial configuration
        self.logger.log_config({
            "platform": platform,
            "model": model,
            "ollama_base_url": ollama_base_url,
            "max_searches": max_searches,
            "system_prompt": self.system_prompt,
            "current_date": self.current_date,
            "vector_db_path": self.vector_db_path
        })
        
        # Initialize LLM based on platform
        self.llm = self._initialize_llm()
        
        # Initialize tools
        self.search_tool = TavilySearchResults(max_results=3)
        self.rag_tool = RAGTool(
            embedding_model_name=embedding_model_name,
            ollama_base_url=ollama_base_url,
            agent_name=name,
            persistent_dir=self.vector_db_path,
            base_vectorstore_dir=DEFAULT_VECTORSTORE_DIR
        )
        
        # Create React agent with both tools
        self.agent = create_react_agent(
            self.llm,
            tools=[self.search_tool, self.rag_tool],
            prompt=self.system_prompt
        )
        
        # Create the state graph
        self.graph = self._build_graph()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the specified platform."""
        if self.platform == "groq":
            return ChatGroq(model=self.model)
        elif self.platform == "ollama":
            return ChatOllama(model=self.model, base_url=self.ollama_base_url, keep_alive=1000)
        elif self.platform == "openai":
            return ChatOpenAI(model=self.model)
        elif self.platform == "huggingface":
            return ChatHuggingFace(model=self.model)
        elif self.platform == "anthropic":
            return ChatAnthropic(model=self.model)
        elif self.platform == "azure":
            return AzureChatOpenAI(
                azure_deployment=self.model,
                openai_api_version="2023-05-15"
            )
        elif self.platform == "google":
            return ChatGoogleGenerativeAI(model=self.model)
        elif self.platform == "xai":
            return ChatXAI(model=self.model)
        else:
            raise ValueError(f"Unsupported platform: {self.platform}")
    
    def format_llm_input(self, content: str) -> str:
        """Format input for direct LLM calls with system prompt."""
        return f"{self.system_prompt}\n\n{content}"

    def format_agent_input(self, content: str, last_exchange=None) -> dict:
        """Format input for agent calls with system prompt and optional last exchange."""
        system_content = self.system_prompt
        
        if last_exchange:
            system_content += f"\n\nYour last exchange with the user:\nUser: {last_exchange['user']}\nYou: {last_exchange['assistant']}"
        
        return {
            "messages": [
                SystemMessage(content=system_content),
                HumanMessage(content=content)
            ]
        }
    
    def get_last_exchange(self) -> Optional[dict]:
        """Get the last exchange from the chat history."""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                
            if not history.get("conversations"):
                return None
                
            # Get the most recent conversation
            last_conversation = history["conversations"][-1]
            
            if len(last_conversation.get("messages", [])) < 2:
                return None
                
            # Get the last user and assistant messages
            messages = last_conversation["messages"]
            user_messages = [m for m in messages if m["role"] == "user"]
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            
            if not user_messages or not assistant_messages:
                return None
                
            return {
                "user": user_messages[-1]["content"],
                "assistant": assistant_messages[-1]["content"]
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.log_event(
                event_type="history_error",
                node_name="get_last_exchange",
                content={"error": str(e)}
            )
            return None
    
    def save_exchange(self, question: str, answer: str) -> None:
        """Save the current exchange to the chat history."""
        try:
            # Load existing history
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = {"conversations": []}
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Check if there's an existing conversation today
            today = datetime.now().date().isoformat()
            
            # Find today's conversation or create a new one
            today_conversation = None
            for conv in history["conversations"]:
                conv_date = datetime.fromisoformat(conv["timestamp"]).date().isoformat()
                if conv_date == today:
                    today_conversation = conv
                    break
            
            if not today_conversation:
                # Create a new conversation for today
                today_conversation = {
                    "id": f"conv_{int(time.time())}",
                    "timestamp": timestamp,
                    "messages": []
                }
                history["conversations"].append(today_conversation)
            
            # Add the new messages
            today_conversation["messages"].append({
                "role": "user",
                "content": question,
                "timestamp": timestamp
            })
            
            today_conversation["messages"].append({
                "role": "assistant",
                "content": answer,
                "timestamp": timestamp
            })
            
            # Save the updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.logger.log_event(
                event_type="history_saved",
                node_name="save_exchange",
                content={"question": question, "answer_length": len(answer)}
            )
        except Exception as e:
            self.logger.log_event(
                event_type="history_error",
                node_name="save_exchange",
                content={"error": str(e)}
            )

    def call_agent(self, state: OverallState) -> OverallState:
        """Call the agent with the current state."""
        # Get the initial message content properly
        initial_message = state['messages'][0].content if isinstance(state['messages'][0], HumanMessage) else str(state['messages'][0])
        
        # Create proper system message with context
        system_message = SystemMessage(content=f"Question: {initial_message}; Current knowledge which was already researched and create your searchquery based on this: {state['current_knowledge']}")
        
        # Log the system message
        self.logger.log_event(
            event_type="system_message",
            node_name="agent",
            content={
                "message": system_message.content,
                "current_knowledge": state["current_knowledge"]
            },
            metadata={
                "timestamp": datetime.now().isoformat(),
                "message_type": "system"
            }
        )
        
        # Invoke agent with proper message structure
        try:
            time.sleep(2)
            response = self.agent.invoke({"messages": [system_message]})
            
            # Log the agent's response
            self.logger.log_event(
                event_type="agent_response",
                node_name="agent",
                content={
                    "response": str(response),
                    "is_tool_call": isinstance(response, ToolMessage)
                },
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "response_type": type(response).__name__
                }
            )
            
            # Handle response properly
            if isinstance(response, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
                state["messages"].append(response)
            elif isinstance(response, dict) and "messages" in response:
                state["messages"].extend(response["messages"])
            
        except Exception as e:
            # Log any errors that occur
            self.logger.log_event(
                event_type="agent_error",
                node_name="agent",
                content={
                    "error": str(e),
                    "system_message": system_message.content
                },
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        return state
    
    def summarize_knowledge_node(self, state: OverallState) -> OverallState:
        """Summarize the knowledge from the last tool message."""
        self.logger.log_event(
            event_type="node_start",
            node_name="summarize",
            content={"current_knowledge": state["current_knowledge"]}
        )
        
        # Extract messages properly
        messages = [msg for msg in state['messages'] if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage))]
        
        # Find last tool message
        last_tool_message = next((msg for msg in reversed(messages) 
                                if isinstance(msg, ToolMessage)), None)
        
        if last_tool_message:
            summary_content = f"Summarize the following in 5-10 bullet points or in a table with key information(numbers and keyfacts) for the userquestion -{state['messages'][0].content}- and add the source url at the end as a link: {last_tool_message.content}"
            time.sleep(2)
            # Using LLM directly with system prompt
            summary = self.llm.invoke(self.format_llm_input(summary_content))
            time.sleep(2)
            state["current_knowledge"].append(summary.content)
            
            self.logger.log_event(
                event_type="knowledge_summary",
                node_name="summarize",
                content={
                    "tool_message": last_tool_message.content,
                    "summary": summary.content
                }
            )
        
        return state
    
    def reflection_node(self, state: OverallState) -> OverallState:
        """Reflect on the current knowledge and decide if more searches are needed."""
        self.logger.log_event(
            event_type="node_start",
            node_name="reflection",
            content={
                "knowledge_count": len(state["current_knowledge"]),
                "current_knowledge": state["current_knowledge"]
            }
        )
        
        # Check for sufficient knowledge
        has_sufficient_knowledge = (
            len(state['current_knowledge']) >= 5 and  # Need at least 5 pieces of knowledge
            all(len(k.strip()) > 50 for k in state['current_knowledge'])  # Each piece should be substantial
        )
        
        # Check for too many searches
        too_many_searches = len(state['needs_additional_search']) >= self.max_searches
        
        # Check for repeated knowledge (simple version)
        has_repeated_knowledge = (
            state['current_knowledge'] and 
            any(knowledge in state['current_knowledge'][-1] 
                for knowledge in state['current_knowledge'][:-1])
        )
        
        # Early exit conditions
        if has_sufficient_knowledge:
            print("Stopping search: Have sufficient detailed knowledge")
            state["needs_additional_search"].append(False)
            return state
        
        if too_many_searches:
            print(f"Stopping search: Reached maximum number of searches ({self.max_searches})")
            state["needs_additional_search"].append(False)
            return state
            
        if has_repeated_knowledge:
            print("Stopping search: Detected repeated knowledge")
            state["needs_additional_search"].append(False)
            return state
        
        # Construct a more detailed prompt for the LLM
        reflection_prompt = f"""
        Question: {state['messages'][0].content}
        Current Knowledge: {state['current_knowledge']}

        Based on this information, do we need additional searches to properly answer the question?
        Consider:
        1. Do we have specific, relevant facts that address the question?
        2. Is the information complete enough to give a confident answer?
        3. Are there important aspects of the question not covered by current knowledge?

        Respond with ONLY 'true' nothin else if we really need more searches, or 'false' if we have sufficient information.
        """
        time.sleep(2)
        # Get LLM's reflection with system prompt
        reflection = self.llm.invoke(self.format_llm_input(reflection_prompt)).content.lower().strip()
        
        self.logger.log_event(
            event_type="reflection_decision",
            node_name="reflection",
            content={
                "needs_more_search": reflection == "true",
                "reflection_response": reflection
            }
        )
        
        # Strict boolean check
        needs_more = reflection == "true"
        state["needs_additional_search"].append(needs_more)
        
        print(f"Decision - needs more search: {needs_more}")
        
        return state
    
    def final_answer_node(self, state: OverallState) -> OverallState:
        """Generate the final answer based on the accumulated knowledge."""
        self.logger.log_event(
            event_type="node_start",
            node_name="final",
            content={
                "current_knowledge": state["current_knowledge"],
                "question": state["messages"][0].content
            }
        )
        
        current_knowledge = state["current_knowledge"]
        question = state["messages"][0].content
        
        answer_prompt = f"answer this question exactly in detail:{question}; by using the folowing current knowledge with used sources:{current_knowledge}"
        time.sleep(2)
        # Apply system prompt to final answer generation
        full_answer = self.llm.invoke(self.format_llm_input(answer_prompt))
        state["messages"].append(AIMessage(content=full_answer.content))
        
        self.logger.log_final_answer(
            answer=full_answer.content,
            metadata={
                "total_knowledge_items": len(current_knowledge),
                "final_message_count": len(state["messages"])
            }
        )
        
        return state
    
    def reflection_decision(self, state: OverallState) -> str:
        """Decide whether to continue searching or generate the final answer."""
        print("\n=== REFLECTION DECISION START ===")
        print(f"needs_additional_search list: {state['needs_additional_search']}")
        decision = "again" if state["needs_additional_search"][-1] == True else "end"
        print(f"Decision: {decision}")
        print("=== REFLECTION DECISION END ===\n")
        return decision
    
    def add_to_knowledge_base(self, documents):
        """
        Add documents to the agent's knowledge base.
        
        Args:
            documents: List of documents to add. Can be Document objects, 
                      dictionaries with 'page_content' and optional 'metadata',
                      or strings (which will be converted to Document objects)
        """
        self.rag_tool.add_documents(documents)
        
    def clear_knowledge_base(self):
        """Clear all documents from the agent's knowledge base."""
        self.rag_tool.clear()
    
    def _build_graph(self) -> StateGraph:
        """Build the state graph for the research agent."""
        # Create retry policy for API errors
        retry = RetryPolicy(
            max_attempts=5,
            initial_interval=1.0,  # Start with a 1 second delay
            backoff_factor=2.0,    # Double the delay after each retry
            jitter=True            # Add randomness to retry intervals to prevent thundering herd
        )
        checkpointer = InMemorySaver()
        # Create state graph
        graph = StateGraph(OverallState)
        
        # Add nodes with retry policies
        graph.add_node("agent", self.call_agent, retry=retry)
        graph.add_node("summarize", self.summarize_knowledge_node, retry=retry)
        graph.add_node("reflection", self.reflection_node, retry=retry)
        graph.add_node("final", self.final_answer_node, retry=retry)
        
        # Add edges
        graph.add_edge("agent", "summarize")
        graph.add_edge("summarize", "reflection")
        
        graph.add_conditional_edges(
            "reflection", 
            self.reflection_decision, 
            {
                "again": "agent",
                "end": "final"
            }
        )
        
        graph.set_entry_point("agent")
        graph.add_edge("final", END)
        
        return graph.compile(checkpointer=checkpointer)
    
    def run(self, question: str) -> str:
        """
        Run the research agent with the given question.
        
        Args:
            question: The question to research
            
        Returns:
            The final answer as a string
        """
        # Get current timestamp from metadata for time-based queries
        current_time = datetime.now()
        
        # Get last exchange if available
        last_exchange = self.get_last_exchange()
        
        # Update initial state with the formatted agent input
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "current_knowledge": [],
            "needs_additional_search": [False]
        }
        
        self.logger.log_event(
            event_type="conversation_start",
            node_name="main",
            content={
                "question": question,
                "timestamp": current_time.isoformat(),
                "has_history": last_exchange is not None,
                "metadata": {
                    "is_time_based_query": any(word in question.lower() for word in ["yesterday", "today", "last week", "last month", "gestern", "heute"])
                }
            }
        )
        
        # Update system prompt with last exchange if available
        if last_exchange:
            self.system_prompt += f"\n\nYour last exchange with the user:\nUser: {last_exchange['user']}\nYou: {last_exchange['assistant']}"
        
        try:
            # Run the flow
            last_output = None
            for output in self.graph.stream(initial_state):
                last_output = output
                for key, value in output.items():
                    try:
                        # Safely extract message
                        message = "No message available"
                        if isinstance(value, dict) and "messages" in value and value["messages"]:
                            messages = value["messages"]
                            if isinstance(messages, list) and messages:
                                last_message = messages[-1]
                                message = str(last_message)
                        
                        # Safely extract current_knowledge
                        current_knowledge = []
                        if isinstance(value, dict) and "current_knowledge" in value:
                            knowledge = value["current_knowledge"]
                            if isinstance(knowledge, list):
                                current_knowledge = knowledge
                        
                        # Safely extract needs_additional_search
                        needs_search = False
                        if isinstance(value, dict) and "needs_additional_search" in value:
                            search_list = value["needs_additional_search"]
                            if isinstance(search_list, list) and search_list:
                                needs_search = bool(search_list[-1])
                        
                        self.logger.log_event(
                            event_type="node_output",
                            node_name=key,
                            content={
                                "message": message,
                                "current_knowledge": current_knowledge,
                                "needs_additional_search": needs_search
                            },
                            metadata={
                                "timestamp": datetime.now().isoformat(),
                                "node_type": key
                            }
                        )
                    except Exception as e:
                        # Log any errors but continue execution
                        self.logger.log_event(
                            event_type="node_output_error",
                            node_name=key,
                            content={"error": str(e)}
                        )
            
            # Get final answer and save logs
            final_answer = "No final answer generated"
            
            # Process the last output if available
            if last_output and isinstance(last_output, dict):
                for key, value in last_output.items():
                    if isinstance(value, dict) and "messages" in value and value["messages"]:
                        messages = value["messages"]
                        if isinstance(messages, list) and messages:
                            last_message = messages[-1]
                            if hasattr(last_message, "content"):
                                final_answer = last_message.content
                            else:
                                final_answer = str(last_message)
                            break
            
            # If we still don't have a good answer, try using the agent directly
            if final_answer == "No final answer generated":
                try:
                    direct_response = self.agent.invoke({"messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]})
                    
                    if isinstance(direct_response, dict) and "messages" in direct_response and direct_response["messages"]:
                        final_answer = direct_response["messages"][-1].content
                    elif hasattr(direct_response, "content"):
                        final_answer = direct_response.content
                    else:
                        final_answer = str(direct_response)
                except Exception as e:
                    self.logger.log_event(
                        event_type="direct_agent_error",
                        node_name="main",
                        content={"error": str(e)}
                    )
            
            # Save the exchange to history
            self.save_exchange(question, final_answer)
            
            # Update performance metrics with safe defaults
            metrics = {
                "total_searches": 1,
                "total_messages": 2,
                "knowledge_items": 0,
                "execution_time": (datetime.now() - current_time).total_seconds()
            }
            
            # Try to extract more accurate metrics if possible
            if last_output and isinstance(last_output, dict):
                for key, value in last_output.items():
                    if isinstance(value, dict):
                        if "needs_additional_search" in value and isinstance(value["needs_additional_search"], list):
                            metrics["total_searches"] = max(1, len(value["needs_additional_search"]) - 1)
                        
                        if "messages" in value and isinstance(value["messages"], list):
                            metrics["total_messages"] = len(value["messages"])
                        
                        if "current_knowledge" in value and isinstance(value["current_knowledge"], list):
                            metrics["knowledge_items"] = len(value["current_knowledge"])
                        break
            
            self.logger.update_performance_metrics(metrics)
            
            # Save the complete log
            log_file = self.logger.save_log()
            
            return final_answer
            
        except Exception as e:
            # Log the error and return a fallback response
            self.logger.log_event(
                event_type="graph_execution_error",
                node_name="main",
                content={"error": str(e)}
            )
            
            # Try to use the agent directly as a fallback
            try:
                self.logger.log_event(
                    event_type="fallback_attempt",
                    node_name="main",
                    content={"message": "Attempting direct agent fallback"}
                )
                
                direct_response = self.agent.invoke({"messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=question)]})
                
                fallback_answer = "No answer generated"
                if isinstance(direct_response, dict) and "messages" in direct_response and direct_response["messages"]:
                    fallback_answer = direct_response["messages"][-1].content
                elif hasattr(direct_response, "content"):
                    fallback_answer = direct_response.content
                else:
                    fallback_answer = str(direct_response)
                
                self.save_exchange(question, fallback_answer)
                
                # Update minimal performance metrics
                self.logger.update_performance_metrics({
                    "total_searches": 0,
                    "total_messages": 2,
                    "knowledge_items": 0,
                    "execution_time": (datetime.now() - current_time).total_seconds()
                })
                
                # Save the complete log
                log_file = self.logger.save_log()
                
                return fallback_answer
                
            except Exception as fallback_error:
                # Log the fallback error
                self.logger.log_event(
                    event_type="fallback_error",
                    node_name="main",
                    content={"error": str(fallback_error)}
                )
                
                # Last resort fallback
                error_message = f"I encountered an error while processing your question: {str(e)}. Fallback also failed: {str(fallback_error)}"
                self.save_exchange(question, error_message)
                
                # Save the complete log
                log_file = self.logger.save_log()
                
                return error_message

from dotenv import load_dotenv
load_dotenv('/Users/Uni/Desktop/Coding/studienarbeit/venv/Scenario_Agency/config/app.env')
