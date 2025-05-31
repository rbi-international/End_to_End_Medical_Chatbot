"""
Helper Functions for Medical Chatbot

This module contains utility functions for data processing, text splitting, and embeddings
generation used in the medical chatbot application.

The functions handle PDF loading, text chunking, and embeddings creation, forming the
foundation of the knowledge base for the chatbot.
"""


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_file(data_directory):
    """
    Extract data from PDF files in the specified directory.
    
    This function uses DirectoryLoader to load all PDF files from the given directory
    and processes them using PyPDFLoader to extract text content.
    
    Args:
        data_directory (str): Path to the directory containing PDF files.
        
    Returns:
        list: A list of Document objects containing the extracted text from the PDFs.
        
    Example:
        documents = load_pdf_file("./data/medical_books")
    """
    # Create a DirectoryLoader that finds all PDF files in the directory
    # and uses PyPDFLoader to extract their content
    loader = DirectoryLoader(
        data_directory,
        glob="*.pdf",  # Only load files with .pdf extension
        loader_cls = PyPDFLoader  # Use PyPDFLoader for processing PDFs
    )

    # Load all documents from the directory
    documents = loader.load()
    
    # Log the number of pages/documents loaded
    print(f"Loaded {len(documents)} document pages from {data_directory}")
    
    return documents


def text_split(extracted_data, chunk_size=500, chunk_overlap=20):
    """
    Split the extracted text data into smaller, manageable chunks.
    
    This function uses RecursiveCharacterTextSplitter to divide documents into
    smaller chunks for better processing and semantic searching.
    
    Args:
        extracted_data (list): List of Document objects containing text.
        chunk_size (int, optional): The target size of each text chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 20.
        
    Returns:
        list: A list of smaller Document chunks.
        
    Example:
        chunks = text_split(documents, chunk_size=600, chunk_overlap=50)
    """
    # Create a text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Target size for each chunk
        chunk_overlap=chunk_overlap,  # Overlap between chunks to maintain context
        length_function=len,  # Function to measure chunk size
        is_separator_regex=False  # Don't use regex for separation
    )
    
    # Split the documents into chunks
    text_chunks = text_splitter.split_documents(extracted_data)
    
    # Log the number of chunks created
    print(f"Split into {len(text_chunks)} chunks of text (chunk_size: {chunk_size}, overlap: {chunk_overlap})")
    
    return text_chunks


def download_hugging_face_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Initialize and return a HuggingFace embeddings model.
    
    This function downloads (if needed) and initializes a sentence transformer model
    from HuggingFace for generating text embeddings.
    
    Args:
        model_name (str, optional): The name of the HuggingFace model to use.
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            
    Returns:
        HuggingFaceEmbeddings: An initialized embeddings model.
        
    Note:
        The default model 'sentence-transformers/all-MiniLM-L6-v2' returns 
        384-dimensional embeddings and offers a good balance between 
        performance and accuracy.
        
    Example:
        embeddings = download_hugging_face_embeddings()
        # Or with a custom model:
        embeddings = download_hugging_face_embeddings('sentence-transformers/all-mpnet-base-v2')
    """
    # Initialize the embeddings model from HuggingFace
    print(f"Loading embeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,  # Model to use for embeddings
        model_kwargs={'device': 'cuda'},  # Use GPU if available
        encode_kwargs={'normalize_embeddings': True}  # Normalize the embeddings
    )
    
    print(f"Embeddings model loaded successfully")
    
    return embeddings