"""
Prompt Templates for Medical Chatbot

This module contains system prompts and templates used to format interactions
between the user and the LLM in the medical chatbot application.

These prompts are designed to guide the model's responses and ensure they're
relevant, concise, and helpful in a medical context.
"""

# Base system prompt for general medical question answering
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Enhanced medical system prompt with more specific guidance
# Can be used as an alternative to the base system prompt
medical_system_prompt = (
    "You are a helpful and knowledgeable medical assistant. "
    "Use the following pieces of retrieved medical context to answer the question. "
    "Base your answer strictly on the provided context and avoid making assumptions. "
    "If the provided context doesn't contain enough information to answer confidently, "
    "acknowledge the limitations and explain what you do know from the context. "
    "Keep your answers concise (three sentences maximum) while ensuring medical accuracy. "
    "Do not provide medical advice that goes beyond the context provided. "
    "Always remind the user to consult healthcare professionals for personalized medical advice."
    "\n\n"
    "{context}"
)

# User question template for combining with context
user_template = (
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer: "
)

# Human message template - simpler version for direct questions
human_template = "{question}"

# Complete prompt template function for flexibility
def get_prompt(context, question, system=system_prompt):
    """
    Creates a complete prompt by combining system prompt, context, and question.
    
    Args:
        context (str): The retrieved context information from the knowledge base
        question (str): The user's question
        system (str, optional): The system prompt to use. Defaults to system_prompt.
        
    Returns:
        dict: A formatted prompt dictionary with system and human messages
        
    Example:
        prompt = get_prompt(context_text, "What are the symptoms of diabetes?")
    """
    formatted_system = system.format(context=context)
    formatted_human = human_template.format(question=question)
    
    # Return as a message dictionary format used by many LLM APIs
    return {
        "system": formatted_system,
        "human": formatted_human
    }