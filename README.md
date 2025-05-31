An end-to-end medical chatbot using Retrieval Augmented Generation (RAG) with LangChain, Pinecone, and OpenAI.

![Medical Chatbot](https://cdn-icons-png.flaticon.com/512/387/387569.png)

## Overview

This application allows users to ask medical questions and receive accurate, contextual answers based on a knowledge base of medical literature. The system uses a RAG architecture to retrieve relevant medical information and generate human-like responses.

## Features

- **PDF Knowledge Base**: Extract and process information from medical documents
- **Vector Search**: Convert text into embeddings for semantic search
- **Contextual Answers**: Generate accurate, context-aware responses
- **User-Friendly Interface**: Simple chat interface for easy interaction
- **Scalable Architecture**: Modular design for future enhancements

## Tech Stack

- **Backend**: Python, Flask
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: Pinecone
- **LLM**: OpenAI GPT Models
- **Framework**: LangChain
- **Frontend**: HTML, CSS, JavaScript (Bootstrap)

## Project Structure

```
End_to_End_Medical_Chatbot/
├── app.py                 # Main Flask application
├── store_index.py         # Script to create Pinecone vector database
├── templates/
│   └── chat.html          # Chat interface template
├── static/
│   └── style.css          # CSS styling for the chat interface
├── src/
│   ├── __init__.py
│   ├── helper.py          # Utility functions for data processing
│   ├── prompt.py          # Prompt templates for the LLM
│   ├── data_processing/   # Package for data processing (future)
│   ├── embeddings/        # Package for embedding models (future)
│   ├── database/          # Package for database operations (future)
│   └── llm/               # Package for LLM operations (future)
├── Data/
│   └── *.pdf              # Medical PDF documents
├── research/
│   └── trials.ipynb       # Experimentation notebook
├── tests/                 # Testing modules (future)
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/End_to_End_Medical_Chatbot.git
   cd End_to_End_Medical_Chatbot
   ```

2. Create a conda environment:
   ```bash
   conda create -n medicalbot python=3.13.2 -y
   conda activate medicalbot
   ```

3. Install PyTorch with CUDA support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up Pinecone:
## Pinecone Vector Database

### Detailed Pinecone Setup

Pinecone is used as the vector database for storing and retrieving embeddings. Here's a comprehensive guide on setting it up:

1. **Create a Pinecone Account**:
   - Visit [Pinecone's website](https://www.pinecone.io/) and sign up for an account
   - Verify your email and complete the registration process

2. **Create an API Key**:
   - Log in to your Pinecone console
   - Navigate to "API Keys" in the left sidebar
   - Click "Create API Key"
   - Give your key a name (e.g., "medical-chatbot")
   - Copy the API key and store it securely

3. **No Need to Create Index Manually**:
   - The `store_index.py` script will create the index for you automatically
   - You only need to provide the API key in your `.env` file
   - The script will create an index named `medicalbot` with the right parameters:
     - **Dimensions**: `384` (matches the all-MiniLM-L6-v2 embedding model)
     - **Metric**: `cosine` (for semantic search)
     - **Serverless Spec**: Using AWS in us-east-1

4. **Running the Indexing Script**:
   - After setting up your `.env` file with the Pinecone API key
   - Run `python store_index.py`
   - This will create the index if it doesn't exist and populate it with embeddings
   - Wait for the process to complete (this may take several minutes depending on your document size)

5. **Understanding Serverless Plan Limits**:
   - Free tier: 100,000 vectors (sufficient for testing)
   - Vector count: Each chunk of text becomes one vector
   - Monthly active vectors: Vectors that are queried/updated during the month
   - Note your plan limits to avoid unexpected charges

6. **Monitoring Usage**:
   - In the Pinecone console, monitor:
     - Vector count
     - QPS (queries per second)
     - Latency
     - Storage used

7. **Optimizing Costs**:
   - Delete unused indexes
   - Consider pod-based deployments for production if you have predictable workloads
   - Use appropriate pod sizes based on your vector count and query needs

8. **Troubleshooting Common Issues**:
   - API key errors: Ensure the key is correctly copied to the `.env` file
   - Dimension mismatch: If you change the embedding model, update the dimensions in `store_index.py`
   - Rate limits: Free tier has QPS limitations
   - Connection timeouts: Check your network and Pinecone status

6. Set up OpenAI API:
   - Create an OpenAI account at [https://platform.openai.com/](https://platform.openai.com/)
   - Navigate to the API keys section
   - Create a new API key and copy it

7. Create a `.env` file with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. First, run the indexing script to create the vector database:
   ```bash
   python store_index.py
   ```
   
   > ⚠️ **Important**: This step must be completed first, otherwise the application will crash.

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

4. Start asking medical questions in the chat interface!

## How It Works

1. **Data Preparation**:
   - Medical PDFs are loaded and text is extracted
   - Text is split into smaller chunks for better processing

2. **Vector Database Creation**:
   - Text chunks are converted to embeddings using Sentence Transformers
   - Embeddings are stored in a Pinecone vector database

3. **Query Processing**:
   - User query is converted to an embedding
   - Similar chunks are retrieved from the vector database
   - Retrieved context and query are sent to OpenAI

4. **Response Generation**:
   - OpenAI generates a response based on the retrieved context
   - Response is displayed to the user

## Future Enhancements

- Multi-document support with source attribution
- Voice interface for questions and answers
- Improved text chunking strategies
- Integration with medical APIs for additional information
- Advanced embedding models for better retrieval
- Conversation history for follow-up questions
- Local LLM support for privacy-focused deployments

## Deployment

This application can be deployed to various cloud platforms like AWS, Azure, or Google Cloud. Deployment instructions will be added in a future update as the project evolves.

For now, the application can be run locally as described in the Usage section.

## Credits

This project was created following the excellent course "Generative AI with AI Agents (MCP) for Developers" by DS with Bappy:

- [Udemy Course](https://www.udemy.com/course/generative-ai-with-ai-agents-mcp-for-developers)
- [YouTube Channel](https://www.youtube.com/@dswithbappy)

## License

MIT License

## Author

Rohit Bharti