      
"""
Simple QA module that works with enhanced_document_qa.py
This module provides a simplified interface for document-based question answering.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from enhanced_document_qa import initialize_document_qa, get_document_qa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleQA:
    """
    Simple wrapper for the enhanced document QA system.
    Provides easy-to-use methods for asking questions about documents.
    """
    
    def __init__(self, documents_dir: str = "documents", llama_model_path: str = None):
        """
        Initialize the SimpleQA system.
        
        Args:
            documents_dir: Path to directory containing documents
            llama_model_path: Optional path to Llama model file
        """
        self.documents_dir = documents_dir
        self.llama_model_path = llama_model_path
        self._qa_system = None
        self._initialized = False
        
        # Try to initialize the system
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the QA system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing SimpleQA with documents from: {self.documents_dir}")
            
            # Check if documents directory exists
            if not os.path.exists(self.documents_dir):
                logger.warning(f"Documents directory not found: {self.documents_dir}")
                os.makedirs(self.documents_dir, exist_ok=True)
                logger.info(f"Created documents directory: {self.documents_dir}")
                return False
            
            # Check for documents
            supported_files = []
            for root, dirs, files in os.walk(self.documents_dir):
                for file in files:
                    if file.lower().endswith(('.pdf', '.pptx', '.ppt', '.txt')):
                        supported_files.append(file)
            
            if not supported_files:
                logger.warning(f"No supported documents found in {self.documents_dir}")
                logger.info("Supported formats: PDF (.pdf), PowerPoint (.pptx, .ppt), Text (.txt)")
                return False
            
            # Initialize the enhanced document QA system
            self._qa_system = initialize_document_qa(
                documents_dir=self.documents_dir,
                llama_model_path=self.llama_model_path
            )
            
            self._initialized = True
            logger.info(f"SimpleQA initialized successfully with {len(supported_files)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SimpleQA: {str(e)}")
            self._initialized = False
            return False
    
    def is_ready(self) -> bool:
        """
        Check if the QA system is ready to answer questions.
        
        Returns:
            True if ready, False otherwise
        """
        if not self._initialized or not self._qa_system:
            return False
        
        # Check if the underlying system is ready
        if hasattr(self._qa_system, 'vector_store_manager'):
            return self._qa_system.vector_store_manager.is_ready()
        
        return self._qa_system is not None
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "answer": "Please provide a question for me to answer.",
                "sources": [],
                "error": "Empty question"
            }
        
        if not self._initialized or not self._qa_system:
            return {
                "success": False,
                "answer": "The QA system is not initialized. Please check if documents are available.",
                "sources": [],
                "error": "System not initialized"
            }
        
        try:
            # Use the enhanced document QA system
            response = self._qa_system.answer_question(question.strip())
            
            return {
                "success": True,
                "answer": response.get("answer", "I couldn't generate a response."),
                "sources": response.get("sources", []),
                "question": question.strip()
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "success": False,
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the QA system.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            "initialized": self._initialized,
            "ready": self.is_ready(),
            "documents_dir": self.documents_dir,
            "llama_available": self.llama_model_path is not None
        }
        
        # Count documents
        document_count = 0
        if os.path.exists(self.documents_dir):
            for root, dirs, files in os.walk(self.documents_dir):
                for file in files:
                    if file.lower().endswith(('.pdf', '.pptx', '.ppt', '.txt')):
                        document_count += 1
        
        status["document_count"] = document_count
        
        # Check if vector store manager is available
        if self._qa_system and hasattr(self._qa_system, 'vector_store_manager'):
            vm = self._qa_system.vector_store_manager
            status["vector_store_initializing"] = vm._is_initializing
            status["vector_store_error"] = vm._initialization_error
        
        return status
    
    def reload_documents(self) -> bool:
        """
        Reload documents and reinitialize the QA system.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Reloading documents and reinitializing QA system...")
        
        # Reset the system
        self._qa_system = None
        self._initialized = False
        
        # Reinitialize
        return self.initialize()
    
    def add_document(self, file_path: str, target_filename: str = None) -> bool:
        """
        Add a new document to the system.
        
        Args:
            file_path: Path to the source file
            target_filename: Optional target filename (uses original if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Source file not found: {file_path}")
                return False
            
            # Determine target filename
            if not target_filename:
                target_filename = os.path.basename(file_path)
            
            # Check file extension
            if not target_filename.lower().endswith(('.pdf', '.pptx', '.ppt', '.txt')):
                logger.error(f"Unsupported file type: {target_filename}")
                return False
            
            # Copy file to documents directory
            target_path = os.path.join(self.documents_dir, target_filename)
            
            import shutil
            shutil.copy2(file_path, target_path)
            
            logger.info(f"Added document: {target_filename}")
            
            # Reload the system
            return self.reload_documents()
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False


# Global instance for easy access
_simple_qa_instance = None

def get_qa_system(documents_dir: str = "documents", llama_model_path: str = None) -> SimpleQA:
    """
    Get or create a global SimpleQA instance.
    
    Args:
        documents_dir: Path to documents directory
        llama_model_path: Optional path to Llama model
        
    Returns:
        SimpleQA instance
    """
    global _simple_qa_instance
    
    if _simple_qa_instance is None:
        _simple_qa_instance = SimpleQA(documents_dir, llama_model_path)
    
    return _simple_qa_instance

def ask_question(question: str, documents_dir: str = "documents", llama_model_path: str = None) -> Dict[str, Any]:
    """
    Simple function to ask a question using the default QA system.
    
    Args:
        question: The question to ask
        documents_dir: Path to documents directory
        llama_model_path: Optional path to Llama model
        
    Returns:
        Dictionary containing answer and sources
    """
    qa_system = get_qa_system(documents_dir, llama_model_path)
    return qa_system.ask(question)

def get_system_status(documents_dir: str = "documents", llama_model_path: str = None) -> Dict[str, Any]:
    """
    Get the status of the QA system.
    
    Args:
        documents_dir: Path to documents directory
        llama_model_path: Optional path to Llama model
        
    Returns:
        Dictionary containing status information
    """
    qa_system = get_qa_system(documents_dir, llama_model_path)
    return qa_system.get_status()

# Example usage
if __name__ == "__main__":
    # Example of how to use the SimpleQA system
    
    # Initialize with default settings
    qa = SimpleQA()
    
    # Check status
    status = qa.get_status()
    print("QA System Status:", status)
    
    if qa.is_ready():
        # Ask a question
        response = qa.ask("What is artificial intelligence?")
        print("\nQuestion:", response.get("question"))
        print("Answer:", response.get("answer"))
        print("Sources:", response.get("sources"))
    else:
        print("QA system is not ready. Please add documents to the 'documents' directory.")

    