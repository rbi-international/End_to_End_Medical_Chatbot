"""
Template Generator for Medical Chatbot Project

This script creates the initial folder structure and empty files needed for the medical chatbot project.
It sets up a standardized project structure following best practices for Python projects.

Usage:
    python create_project_structure.py

Author: [Rohit Bharti]
Date: May 31, 2025
"""

import os
from pathlib import Path
import logging

# Configure logging to show timestamps with messages
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define all files and directories to be created
# This forms the skeleton of our project structure
list_of_files = [
    # Core application structure
    "src/__init__.py",                # Makes src a proper package
    "src/helper.py",                  # Utility functions for the application
    "src/prompt.py",                  # LLM prompt templates and management
    "src/data_processing/__init__.py", # Package for data processing modules
    "src/data_processing/pdf_extractor.py", # PDF extraction functionality
    "src/data_processing/text_chunker.py",  # Text chunking functionality
    "src/embeddings/__init__.py",     # Package for embedding-related code
    "src/embeddings/model.py",        # Embedding model implementation
    "src/database/__init__.py",       # Package for database interactions
    "src/database/pinecone_ops.py",   # Pinecone vector database operations
    "src/llm/__init__.py",            # Package for LLM operations
    "src/llm/openai_interface.py",    # OpenAI API interface
    
    # Configuration files
    ".env",                           # Environment variables (API keys, etc.)
    "pyproject.toml",                 # Project metadata and dependencies
    "requirements.txt",               # Project dependencies
    "README.md",                      # Project documentation
    
    # Application entry point
    "app.py",                         # Main Flask application
    
    # Research and testing
    "research/trials.ipynb",          # Jupyter notebook for experimentation
    "tests/__init__.py",              # Makes tests a proper package
    "tests/test_pdf_extraction.py",   # Tests for PDF extraction
    "tests/test_embeddings.py",       # Tests for embedding functionality
    "tests/test_pinecone.py",         # Tests for Pinecone operations
    "test.py"                         # General test script
]


def create_project_structure(file_list):
    """
    Creates the project directory structure and empty files based on the provided list.
    
    Args:
        file_list (list): List of file paths to create
        
    Returns:
        None
    """
    # Log the start of the process
    logging.info("Starting project structure creation")
    
    for filepath in file_list:
        # Convert to Path object for better cross-platform compatibility
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        # Create directories if they don't exist
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for file: {filename}")

        # Create file if it doesn't exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                # Add initial content for certain files
                if filename == "README.md":
                    f.write("# Medical Chatbot\n\nAn end-to-end medical chatbot using generative AI.\n")
                elif filename == "pyproject.toml":
                    f.write('[build-system]\nrequires = ["setuptools>=42.0", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n[tool.black]\nline-length = 88\n')
                elif filename == "__init__.py":
                    f.write('"""This module is part of the Medical Chatbot project."""\n')
            
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")
    
    # Log completion
    logging.info("Project structure creation completed")


if __name__ == "__main__":
    # Only execute the code if this file is run directly
    create_project_structure(list_of_files)
    
    # Provide next steps
    logging.info("Project structure created successfully!")
    logging.info("Next steps:")
    logging.info("1. Update .env with your API keys")
    logging.info("2. Install required dependencies")
    logging.info("3. Begin implementing core functionality")