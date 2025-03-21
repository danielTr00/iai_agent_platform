import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
import logging

DEFAULT_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

class AgentLogger:
    """Logger class for tracking and saving agent conversations and operations"""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the logger
        
        Args:
            log_dir: Directory to store log files. If None, uses default logs directory
        """
        self.log_dir = os.path.abspath(log_dir if log_dir is not None else DEFAULT_LOG_DIR)
        self.conversation_id = str(uuid4())
        self.start_time = datetime.now().isoformat()
        self.log_data = {
            "conversation_id": self.conversation_id,
            "start_time": self.start_time,
            "end_time": None,
            "config": {},
            "events": [],
            "final_answer": None,
            "performance_metrics": {
                "total_tokens": 0,
                "total_searches": 0,
                "total_time": 0,
                "search_times": [],
                "llm_times": []
            }
        }
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(self.log_dir, f"{self.conversation_id}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"agent_{self.conversation_id[:8]}")
        self.logger.info(f"Initializing logger. Logs will be saved to: {self.log_dir}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log agent configuration"""
        self.log_data["config"] = config
        self.logger.info(f"Agent configured with: {config}")
    
    def log_event(self, 
                  event_type: str, 
                  node_name: str, 
                  content: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an event in the conversation
        
        Args:
            event_type: Type of event (e.g., 'node_start', 'tool_call', 'llm_response')
            node_name: Name of the node where event occurred
            content: Event content/data
            metadata: Additional metadata about the event
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "node_name": node_name,
            "content": content,
            "metadata": metadata or {}
        }
        self.log_data["events"].append(event)
        self.logger.info(f"Event: {event_type} in {node_name}")
    
    def log_tool_call(self,
                      tool_name: str,
                      inputs: Dict[str, Any],
                      outputs: Any,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a tool call"""
        self.log_event(
            event_type="tool_call",
            node_name="agent",
            content={
                "tool_name": tool_name,
                "inputs": inputs,
                "outputs": outputs
            },
            metadata=metadata
        )
    
    def log_llm_interaction(self,
                          prompt: str,
                          response: str,
                          metadata: Dict[str, Any]) -> None:
        """Log an LLM interaction"""
        self.log_event(
            event_type="llm_interaction",
            node_name="agent",
            content={
                "prompt": prompt,
                "response": response
            },
            metadata=metadata
        )
    
    def log_final_answer(self, answer: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log the final answer"""
        self.log_data["final_answer"] = {
            "content": answer,
            "metadata": metadata or {}
        }
        self.logger.info(f"Final answer logged")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics"""
        self.log_data["performance_metrics"].update(metrics)
    
    def save_log(self) -> str:
        """
        Save the log data to a JSON file
        
        Returns:
            Path to the saved log file
        """
        self.log_data["end_time"] = datetime.now().isoformat()
        
        # Calculate total time
        start = datetime.fromisoformat(self.log_data["start_time"])
        end = datetime.fromisoformat(self.log_data["end_time"])
        self.log_data["performance_metrics"]["total_time"] = (end - start).total_seconds()
        
        # Save to JSON file
        log_file = os.path.join(self.log_dir, f"conversation_{self.conversation_id}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Log saved to {log_file}")
        return log_file
