from langchain_core.tools import BaseTool, ToolException
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Union, ClassVar, Type
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from pydantic import Field
import json
import time

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Get the project root directory (3 levels up from this file)
# This file is in scenario_coordination/custom_tools/
# We need to go up to the Scenario_Agency directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Default vector store directory
DEFAULT_VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vector_db")
# Ensure the directory exists
os.makedirs(DEFAULT_VECTORSTORE_DIR, exist_ok=True)

class RAGTool(BaseTool):
    """Tool for managing and querying a vector database for RAG applications."""
    
    name: ClassVar[str] = "search_document_knowledge"
    description: ClassVar[str] = "Search through uploaded documents for relevant information based on your query."
    
    embedding_model: OllamaEmbeddings = Field(default=None, exclude=True)
    k: int = Field(default=3, description="Number of documents to retrieve in similarity search")
    vectordb: Optional[Chroma] = Field(default=None, exclude=True)
    text_splitter: RecursiveCharacterTextSplitter = Field(default=None, exclude=True)
    persistent_dir: Optional[str] = Field(default=None, description="Directory for persistent vector store")
    is_persistent: bool = Field(default=False, description="Whether the vector store is persistent")
    collection_name: str = Field(default="agent_knowledge_base", description="Name of the collection in the vector store")
    agent_name: Optional[str] = Field(default=None, description="Name of the agent using this knowledge base")
    tracked_documents_file: str = Field(default="", exclude=True)
    tracked_documents: List[str] = Field(default_factory=list, exclude=True)
    
    @property
    def vector_db_path(self) -> Optional[str]:
        """Return the path to the vector database."""
        return self.persistent_dir
    
    def __init__(
        self,
        embedding_model_name: str = "snowflake-arctic-embed:22m",
        ollama_base_url: str = "http://localhost:11434",
        k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persistent_dir: Optional[str] = None,
        collection_name: str = "agent_knowledge_base",
        agent_name: Optional[str] = None,
        base_vectorstore_dir: str = DEFAULT_VECTORSTORE_DIR
    ):
        """
        Initialize the RAG tool with a vector database.
        
        Args:
            embedding_model_name: Name of the Ollama embedding model to use
            ollama_base_url: Base URL for Ollama API
            k: Number of documents to retrieve in similarity search
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            persistent_dir: Directory for persistent vector store. If None but agent_name is provided, 
                            a directory will be created in base_vectorstore_dir
            collection_name: Name of the collection in the vector store
            agent_name: Name of the agent using this knowledge base
            base_vectorstore_dir: Base directory for storing vector databases
        """
        super().__init__()
        
        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(
            model=embedding_model_name,
            base_url=ollama_base_url
        )
        
        # Set parameters
        self.k = k
        self.agent_name = agent_name
        self.collection_name = collection_name
        
        # Set up persistence
        base_vectorstore_dir = DEFAULT_VECTORSTORE_DIR
        
        if persistent_dir:
            # Use the specified directory
            self.persistent_dir = persistent_dir
            self.is_persistent = True
        elif agent_name:
            # Generate a 4-digit ID for the agent
            agent_id = str(hash(agent_name) % 10000).zfill(4)
            
            # Create a directory with agent name and ID
            self.persistent_dir = os.path.join(
                base_vectorstore_dir, 
                f"{agent_name}_{agent_id}"
            )
            self.is_persistent = True
        else:
            # Use default directory
            self.persistent_dir = os.path.join(base_vectorstore_dir, "default")
            self.is_persistent = True
        
        # Initialize text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize document tracking
        self.tracked_documents_file = os.path.join(self.persistent_dir, f"{collection_name}_document_tracking.json")
        self.tracked_documents = self._load_tracked_documents()
        
        # Initialize vector database
        self._initialize_vectordb()
    
    def _load_tracked_documents(self) -> List[str]:
        """Load the list of tracked document filenames from disk."""
        if os.path.exists(self.tracked_documents_file):
            try:
                with open(self.tracked_documents_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load tracked documents file: {str(e)}")
                return []
        return []
    
    def _save_tracked_documents(self):
        """Save the list of tracked document filenames to disk."""
        try:
            with open(self.tracked_documents_file, 'w') as f:
                json.dump(self.tracked_documents, f)
        except IOError as e:
            print(f"Warning: Could not save tracked documents file: {str(e)}")
    
    def is_document_tracked(self, filename: str) -> bool:
        """Check if a document is already tracked by filename."""
        return filename in self.tracked_documents
    
    def track_document(self, filename: str):
        """Add a document filename to the tracking list."""
        if not self.is_document_tracked(filename):
            self.tracked_documents.append(filename)
            self._save_tracked_documents()
    
    def get_tracked_documents(self) -> List[str]:
        """Get the list of tracked document filenames."""
        return self.tracked_documents.copy()
    
    def _initialize_vectordb(self):
        """Initialize vector database, either in-memory or persistent."""
        try:
            if self.is_persistent:
                # Create persistent directory if it doesn't exist
                if self.persistent_dir:
                    os.makedirs(self.persistent_dir, exist_ok=True)
                
                # Check if the directory exists and has content
                if os.path.exists(self.persistent_dir) and os.listdir(self.persistent_dir):
                    # Directory exists and has content, try to load existing database
                    try:
                        self.vectordb = Chroma(
                            persist_directory=self.persistent_dir,
                            embedding_function=self.embedding_model,
                            collection_name=self.collection_name
                        )
                        print(f"Successfully connected to existing vector store at {self.persistent_dir}")
                    except Exception as e:
                        print(f"Error connecting to existing vector store: {str(e)}")
                        print(f"Creating a new vector store at {self.persistent_dir}")
                        # If loading fails, create a new database
                        if os.path.exists(self.persistent_dir):
                            # Rename the existing directory to avoid conflicts
                            backup_dir = f"{self.persistent_dir}_backup_{int(time.time())}"
                            shutil.move(self.persistent_dir, backup_dir)
                            print(f"Moved existing directory to {backup_dir}")
                        
                        # Create a new directory
                        os.makedirs(self.persistent_dir, exist_ok=True)
                        
                        # Initialize a new vector database
                        self.vectordb = Chroma(
                            persist_directory=self.persistent_dir,
                            embedding_function=self.embedding_model,
                            collection_name=self.collection_name
                        )
                else:
                    # Directory doesn't exist or is empty, create a new database
                    self.vectordb = Chroma(
                        persist_directory=self.persistent_dir,
                        embedding_function=self.embedding_model,
                        collection_name=self.collection_name
                    )
            else:
                # Initialize in-memory vector database
                self.vectordb = Chroma(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name
                )
                
            # Initialize document tracking after vector store is set up
            self.tracked_documents_file = os.path.join(self.persistent_dir, "agent_knowledge_base_document_tracking.json")
            self.tracked_documents = self._load_tracked_documents()
            
        except Exception as e:
            print(f"Error initializing vector database: {str(e)}")
            # Fall back to in-memory database if persistent fails
            self.vectordb = Chroma(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            self.is_persistent = False
            self.tracked_documents = []
    
    def add_documents(self, documents: List[Union[Document, Dict[str, Any], str, os.PathLike]]):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents to add. Can be:
                - Document objects
                - Dictionaries with 'page_content' and optional 'metadata'
                - Strings (converted to Document objects)
                - File paths (PDF, DOCX, PPTX, TXT files)
        """
        if self.vectordb is None:
            self._initialize_vectordb()
            
        processed_docs = []
        tracked_filenames = []
        
        for doc in documents:
            # Handle string content
            if isinstance(doc, str) and not os.path.exists(doc):
                # It's a text string, not a file path
                processed_docs.append(Document(
                    page_content=doc,
                    metadata={
                        "source": "text_input",
                        "created_at": datetime.now().isoformat(),
                        "doc_id": str(uuid.uuid4())
                    }
                ))
            
            # Handle dictionary with page_content
            elif isinstance(doc, dict) and "page_content" in doc:
                metadata = doc.get("metadata", {})
                if "source" in metadata:
                    # Track the document by filename if source is provided
                    filename = os.path.basename(metadata["source"])
                    tracked_filenames.append(filename)
                
                processed_docs.append(Document(
                    page_content=doc["page_content"],
                    metadata=metadata
                ))
            
            # Handle Document objects
            elif isinstance(doc, Document):
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    # Track the document by filename if source is provided
                    filename = os.path.basename(doc.metadata["source"])
                    tracked_filenames.append(filename)
                
                processed_docs.append(doc)
            
            # Handle file paths
            elif isinstance(doc, (str, os.PathLike)) and os.path.exists(doc):
                file_path = Path(doc)
                filename = file_path.name
                tracked_filenames.append(filename)
                
                # Check if this file is already tracked
                if self.is_document_tracked(filename):
                    print(f"Skipping {filename} as it's already in the vector store")
                    continue
                
                # Load the file
                try:
                    loaded_docs = self._load_file(file_path)
                    processed_docs.extend(loaded_docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        # Process and chunk the documents
        chunked_docs = []
        
        for doc in processed_docs:
            # Apply metadata filtering to ensure compatibility with Chroma
            if hasattr(doc, "metadata"):
                # Instead of using filter_complex_metadata, we'll handle metadata manually
                # Remove any complex objects that might cause serialization issues
                filtered_metadata = {}
                for key, value in doc.metadata.items():
                    # Only include simple types that can be easily serialized
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered_metadata[key] = value
                doc.metadata = filtered_metadata
            
            # Split the document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}
                chunk_metadata.update({
                    "chunk": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        # Add chunked documents to vector database
        if chunked_docs:
            self.vectordb.add_documents(chunked_docs)
            
            # Track the documents by filename
            for filename in tracked_filenames:
                self.track_document(filename)
            
            # Persist if using persistent storage
            if self.is_persistent:
                try:
                    self.persist()
                except Exception as e:
                    print(f"Warning: Could not persist vector database: {str(e)}")
                    print("Documents were added to in-memory database but may not be saved to disk.")
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a file using the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """
        file_extension = file_path.suffix.lower()
        file_name = file_path.name
        file_size = os.path.getsize(file_path)
        
        # Base metadata for all documents from this file
        base_metadata = {
            "source": str(file_path),
            "file_name": file_name,
            "file_extension": file_extension,
            "file_size": file_size,
            "created_at": datetime.now().isoformat(),
            "doc_id": str(uuid.uuid4())
        }
        
        # Add agent name if available
        if self.agent_name:
            base_metadata["agent_name"] = self.agent_name
        
        try:
            # Choose loader based on file extension
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                
                # Add page numbers to metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update(base_metadata)
                    doc.metadata["page_number"] = i + 1
                    doc.metadata["total_pages"] = len(docs)
                
                return docs
                
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update(base_metadata)
                
                return docs
                
            elif file_extension == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file_path))
                docs = loader.load()
                
                # Add slide numbers to metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update(base_metadata)
                    doc.metadata["slide_number"] = i + 1
                    doc.metadata["total_slides"] = len(docs)
                
                return docs
                
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path))
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update(base_metadata)
                
                return docs
                
            else:
                # Default to text loader for unknown file types
                try:
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    
                    # Add metadata to each document
                    for doc in docs:
                        doc.metadata.update(base_metadata)
                        doc.metadata["file_type"] = "unknown"
                    
                    return docs
                except Exception as e:
                    # If text loader fails, create a document with error message
                    return [Document(
                        page_content=f"Error loading file {file_name}: {str(e)}",
                        metadata=base_metadata
                    )]
                    
        except Exception as e:
            # Handle any exceptions during loading
            error_metadata = base_metadata.copy()
            error_metadata["error"] = str(e)
            error_metadata["error_type"] = type(e).__name__
            
            return [Document(
                page_content=f"Error loading file {file_name}: {str(e)}",
                metadata=error_metadata
            )]
    
    def _run(self, query: str) -> str:
        """
        Query the vector database for relevant information.
        
        Args:
            query: The query to search for in the knowledge base
            
        Returns:
            String containing relevant information from the knowledge base
        """
        if self.vectordb is None:
            return "Knowledge base is empty. No documents have been added yet."
        
        try:
            # Search for relevant documents
            docs = self.vectordb.similarity_search(query, k=self.k)
            
            # Format results
            if not docs:
                return "No relevant information found in the knowledge base."
            
            results = []
            for i, doc in enumerate(docs):
                # Extract key metadata for display
                source = doc.metadata.get("source", "Unknown")
                
                # Format different sources differently
                if "file_name" in doc.metadata:
                    file_name = doc.metadata.get("file_name")
                    file_extension = doc.metadata.get("file_extension", "")
                    
                    # Add page/slide information if available
                    if "page_number" in doc.metadata:
                        location = f"Page {doc.metadata['page_number']} of {doc.metadata.get('total_pages', '?')}"
                    elif "slide_number" in doc.metadata:
                        location = f"Slide {doc.metadata['slide_number']} of {doc.metadata.get('total_slides', '?')}"
                    else:
                        location = f"Chunk {doc.metadata.get('chunk_index', '?')} of {doc.metadata.get('total_chunks', '?')}"
                    
                    source_info = f"Source: {file_name} ({file_extension}) - {location}"
                else:
                    source_info = f"Source: {source}"
                
                # Add agent info if available
                if "agent_name" in doc.metadata:
                    source_info += f" | Agent: {doc.metadata['agent_name']}"
                
                # Format the result with metadata
                results.append(
                    f"Document {i+1}:\n"
                    f"{source_info}\n"
                    f"Content:\n{doc.page_content}\n"
                )
            
            return "\n".join(results)
        except Exception as e:
            raise ToolException(f"Error querying knowledge base: {str(e)}")
    
    def clear(self):
        """Clear all documents from the vector database."""
        if self.vectordb is not None:
            # Reset the vector database
            self._initialize_empty_vectordb()
            
            # Persist if using persistent storage
            if self.is_persistent:
                try:
                    self.persist()
                except Exception as e:
                    print(f"Warning: Could not persist empty vector database: {str(e)}")
                    print("Database was cleared in memory but changes may not be saved to disk.")
    
    def persist(self):
        """Persist the vector database to disk if using persistent storage."""
        if self.is_persistent and self.vectordb is not None:
            # Check if the vectordb has a persist method (newer versions of Chroma)
            if hasattr(self.vectordb, 'persist'):
                self.vectordb.persist()
                return f"Vector database persisted to {self.persistent_dir}"
            else:
                # For older versions of Chroma or other vector stores that don't have persist
                # The data should be automatically saved since we're using a persistent directory
                return f"Using persistent storage at {self.persistent_dir} (auto-saved)"
        return "Vector database is not persistent, nothing to persist."
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.vectordb is None:
            return {"status": "empty", "document_count": 0}
        
        try:
            # Get collection statistics
            collection = self.vectordb._collection
            count = collection.count()
            
            stats = {
                "status": "active",
                "document_count": count,
                "is_persistent": self.is_persistent,
                "collection_name": self.collection_name,
            }
            
            if self.agent_name:
                stats["agent_name"] = self.agent_name
                
            if self.is_persistent:
                stats["persistent_dir"] = self.persistent_dir
                
            return stats
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    @classmethod
    def from_persistent_dir(
        cls,
        persistent_dir: str,
        embedding_model_name: str = "snowflake-arctic-embed:22m",
        ollama_base_url: str = "http://localhost:11434",
        k: int = 3,
        collection_name: str = "agent_knowledge_base",
        agent_name: Optional[str] = None
    ):
        """
        Create a RAG tool from an existing persistent directory.
        
        Args:
            persistent_dir: Directory containing the persistent vector store
            embedding_model_name: Name of the Ollama embedding model to use
            ollama_base_url: Base URL for Ollama API
            k: Number of documents to retrieve in similarity search
            collection_name: Name of the collection in the vector store
            agent_name: Name of the agent using this knowledge base
            
        Returns:
            RAGTool instance with the loaded vector store
        """
        if not os.path.exists(persistent_dir):
            raise ValueError(f"Persistent directory does not exist: {persistent_dir}")
        
        # Try to extract agent name from directory name if not provided
        if agent_name is None:
            dir_name = os.path.basename(persistent_dir)
            # Try to extract agent name from directory name (format: agent_name_YYYYMMDD_HHMMSS)
            parts = dir_name.split('_')
            if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                # The format matches our expected pattern
                agent_name = '_'.join(parts[:-2])
        
        # Create a new RAG tool with the persistent directory
        return cls(
            embedding_model_name=embedding_model_name,
            ollama_base_url=ollama_base_url,
            k=k,
            persistent_dir=persistent_dir,
            collection_name=collection_name,
            agent_name=agent_name
        )
    
    @classmethod
    def list_available_vectorstores(cls, base_dir: str = DEFAULT_VECTORSTORE_DIR) -> List[Dict[str, Any]]:
        """
        List all available vector stores in the base directory.
        
        Args:
            base_dir: Base directory to search for vector stores
            
        Returns:
            List of dictionaries with information about each vector store
        """
        if not os.path.exists(base_dir):
            return []
        
        vectorstores = []
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # Check if it's a directory and has Chroma files
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "chroma-embeddings.parquet")):
                # Try to parse agent name and creation date from directory name
                agent_name = None
                creation_date = None
                
                parts = item.split('_')
                if len(parts) >= 3:
                    # Try to extract date from the last two parts (YYYYMMDD_HHMMSS)
                    date_str = f"{parts[-2]}_{parts[-1]}"
                    try:
                        creation_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S").isoformat()
                        agent_name = '_'.join(parts[:-2])
                    except ValueError:
                        # If date parsing fails, just use the directory name
                        agent_name = item
                else:
                    agent_name = item
                
                vectorstores.append({
                    "path": item_path,
                    "name": item,
                    "agent_name": agent_name,
                    "creation_date": creation_date,
                    "size_bytes": sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(item_path)
                        for filename in filenames
                    )
                })
        
        return vectorstores