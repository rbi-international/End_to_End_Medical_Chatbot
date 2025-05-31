"""
Vector Database Indexing for Medical Chatbot

This script extracts data from medical PDFs, splits them into chunks,
creates embeddings, and stores them in a Pinecone vector database.

This is a standalone script that should be run once to initialize the
knowledge base before starting the chatbot application.
"""

import os
import time
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PineconeApiException
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings


def create_vector_db_index():
    """
    Main function to create and populate the vector database index.
    
    This function:
    1. Loads environment variables
    2. Extracts text from PDF files
    3. Splits text into chunks
    4. Generates embeddings
    5. Creates a Pinecone index
    6. Uploads the embeddings to Pinecone
    
    Returns:
        bool: True if index creation was successful, False otherwise
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Pinecone API key from environment variables
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    
    # Check if API key is available
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not found in environment variables")
        return False
    
    # Set the API key in environment (redundant but kept for compatibility)
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    
    try:
        # Extract data from PDF files
        print("Loading PDF files...")
        extracted_data = load_pdf_file(data_directory='Data/')
        
        # Split text into manageable chunks
        print("Splitting text into chunks...")
        text_chunks = text_split(extracted_data)
        
        # Download and initialize embedding model
        print("Initializing embedding model...")
        embeddings = download_hugging_face_embeddings()
        
        # Initialize Pinecone client
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Define index name and parameters
        index_name = "medicalbot"
        dimension = 384  # Dimension of the embedding model (all-MiniLM-L6-v2)
        
        # Check if index already exists
        existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
        
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists. Deleting and recreating...")
            pc.delete_index(index_name)
            # Wait for the index to be fully deleted
            time.sleep(10)
        
        # Create a new Pinecone index
        print(f"Creating new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # Cosine similarity is suitable for text embeddings
            spec=ServerlessSpec(
                cloud="aws",  # Cloud provider
                region="us-east-1"  # Region (adjust as needed)
            )
        )
        
        # Wait for the index to be fully initialized
        print("Waiting for index to initialize...")
        time.sleep(30)
        
        # Embed each chunk and upload to Pinecone
        print("Uploading document embeddings to Pinecone...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        
        print(f"Successfully created and populated Pinecone index '{index_name}'")
        print(f"Uploaded {len(text_chunks)} document chunks to the vector database")
        return True
        
    except PineconeApiException as e:
        print(f"Pinecone API error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    # Only run the indexing process if this script is executed directly
    success = create_vector_db_index()
    
    if success:
        print("Index creation completed successfully!")
    else:
        print("Index creation failed. Please check the error messages above.")