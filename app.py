"""
Medical Chatbot Application

This is the main application file for the Medical Chatbot, which serves as a question-answering
system for medical topics. It uses a RAG (Retrieval Augmented Generation) approach to provide
accurate answers based on medical literature.

The application:
1. Connects to Pinecone vector database containing medical document embeddings
2. Retrieves relevant documents based on user queries
3. Uses OpenAI to generate contextual, accurate responses
4. Serves a web interface for user interaction

Author: Rohit Bharti
Date: May 31, 2025
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

def initialize_rag_pipeline():
    """
    Initialize the Retrieval-Augmented Generation (RAG) pipeline.
    
    This function:
    1. Loads environment variables
    2. Sets up embeddings model
    3. Connects to Pinecone vector database
    4. Initializes the LLM and RAG chain
    
    Returns:
        retrieval_chain: The configured RAG chain for question answering
        bool: Success status of initialization
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API keys from environment variables
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        
        # Validate API keys are present
        if not pinecone_api_key or not openai_api_key:
            logger.error("Missing required API keys. Please check your .env file.")
            return None, False
        
        # Set environment variables (redundant but kept for compatibility)
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize the embedding model
        logger.info("Initializing embedding model...")
        embeddings = download_hugging_face_embeddings()
        
        # Define the index name for Pinecone
        index_name = "medicalbot"
        
        # Connect to the existing Pinecone vector store
        logger.info(f"Connecting to Pinecone index: {index_name}")
        try:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            logger.error("Make sure to run store_index.py first to create the index!")
            return None, False
        
        # Create a retriever from the vector store
        # Using similarity search with top 3 results
        retriever = docsearch.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
       # Initialize the OpenAI model
        logger.info("Initializing OpenAI model...")
        from langchain_openai import ChatOpenAI  # Import the correct class
        llm = ChatOpenAI(
            temperature=0.4,  # Lower temperature for more deterministic outputs
            max_tokens=500,   # Limit response length
            model="gpt-3.5-turbo"  # Using the chat model
)
        
        # Create the prompt template for the chatbot
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create the RAG chain
        logger.info("Creating RAG chain...")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logger.info("RAG pipeline initialized successfully")
        return rag_chain, True
        
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        return None, False


# Initialize the RAG pipeline when the app starts
rag_chain, init_success = initialize_rag_pipeline()


@app.route("/")
def index():
    """
    Render the main chat interface.
    
    Returns:
        Rendered HTML template for the chat interface
    """
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    """
    Process user messages and generate responses.
    
    This endpoint:
    1. Receives the user's message
    2. Retrieves relevant medical information
    3. Generates a response using the RAG pipeline
    4. Returns the response to the user
    
    Returns:
        str: The generated response to the user's query
    """
    if not init_success:
        return "I'm having trouble connecting to my knowledge base. Please check the server logs."
    
    try:
        # Get the user's message from the request
        user_message = request.form.get("msg", "")
        
        if not user_message.strip():
            return "I didn't receive a question. Could you please ask me something?"
        
        # Log the user's message
        logger.info(f"User message: {user_message}")
        
        # Process the message through the RAG chain
        response = rag_chain.invoke({"input": user_message})
        
        # Extract and log the answer
        answer = response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
        logger.info(f"Response: {answer}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return "I'm sorry, I encountered an error while processing your question. Please try again."


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the application is running.
    
    Returns:
        JSON: Status of the application
    """
    return jsonify({"status": "healthy", "initialized": init_success})


if __name__ == '__main__':
    # Check if initialization was successful
    if not init_success:
        logger.error("Failed to initialize the RAG pipeline. Check logs for details.")
        logger.error("Make sure store_index.py has been run and API keys are correctly set.")
    
    # Run the Flask application
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)