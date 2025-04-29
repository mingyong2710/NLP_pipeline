from openai import OpenAI
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_openai_streaming_response(messages_history):
    """
    Generate a streaming response from OpenAI API using conversation history
    
    Args:
        messages_history (list): List of message dictionaries with 'role' and 'content' keys
                               representing the conversation history
    
    Yields:
        str: Chunks of the generated response
    """
    
    try:
        # Format the messages for OpenAI API
        formatted_messages = []
        for msg in messages_history:
            if isinstance(msg, tuple):
                role, content = msg
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
        
        # Call OpenAI API with the full conversation history
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=formatted_messages,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            
    except Exception as e:
        yield f"Error: {str(e)}"