import requests
import json
import os
import time
import threading

class QueryVLM:
    def __init__(self):
        # API keys should ideally be stored in environment variables for security
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAG6Ewr5aTJsFxEFjiixkP_tt0SNfF2cZQ")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "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")

    def query_gemini(self, query):
        """Queries a specific Gemini model via Google's API."""
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_api_key}'
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": query}]}]}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {e}"

    def query_openrouter(self, model_name, query):
        """Queries a specific model via the OpenRouter API."""
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json',
            # Recommended headers by OpenRouter
            'HTTP-Referer': 'http://localhost', 
            'X-Title': 'QueryVLM'
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": query}]
        }
        
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {e}"

def query_and_print(vlm_instance, model_name, query):
    """Wrapper function to handle querying and printing the response for a single model."""
    print(f"--- Querying {model_name}... ---")
    response = vlm_instance.query_openrouter(model_name, query)
    print(f"\nResponse from {model_name}:")
    print(response)
    print("-" * (len(model_name) + 20))


# Example usage
if __name__ == "__main__":
    # Initialize the VLM
    vlm = QueryVLM()
    
    # Define the models to query simultaneously through OpenRouter
    models_to_query = [
        "deepseek/deepseek-chat-v3.1:free",
        "moonshotai/kimi-k2:free",
        "qwen/qwen3-235b-a22b:free"
    ]
    
    # Example query
    test_query = "Explain what a neural network is in three sentences."
    
    threads = []
    
    # Create and start a thread for each model query
    for model in models_to_query:
        thread = threading.Thread(target=query_and_print, args=(vlm, model, test_query))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll models have responded.")