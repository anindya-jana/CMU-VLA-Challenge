import requests
import json
import os
import time
import threading

# --- This is your original code, unchanged ---
class QueryVLM:
    def __init__(self, model_name):
        self.model_name = "gemini"

    def query_gemini(self, query):
        api_key="AIzaSyAG6Ewr5aTJsFxEFjiixkP_tt0SNfF2cZQ"
         #### url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}'
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": query
                        }
                    ]
                },
            ],
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None


    def query(self, query):
        
        if self.model_name == "gemini":
            return self.query_gemini(query)
        elif self.model_name == "chatgpt":
            return self.query_chatgpt(query)
        elif self.model_name == "mistral":
            return self.query_mistral(query)
        else:
            print("Model not supported")
            return None

# --- This is the new part for calling OpenRouter models ---

def query_openrouter(model_name, query):
    """Queries a specific model via the OpenRouter API."""
    api_key = "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"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
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

def query_and_print_openrouter(model_name, query):
    """Wrapper function to handle querying and printing the response for a single OpenRouter model."""
    print(f"--- Querying {model_name}... ---")
    response = query_openrouter(model_name, query)
    print(f"\nResponse from {model_name}:")
    print(response)
    print("-" * (len(model_name) + 20))

# --- Example usage for the new OpenRouter functionality ---
if __name__ == "__main__":
    # Define the models to query simultaneously through OpenRouter
    models_to_query = [
        "deepseek/deepseek-chat-v3.1:free",
        "moonshotai/kimi-k2:free",
        "qwen/qwen3-235b-a22b:free"
    ]
    
    # Example query
    test_query = "What are the main differences between Python lists and tuples?"
    
    threads = []
    
    # Create and start a thread for each model query
    for model in models_to_query:
        thread = threading.Thread(target=query_and_print_openrouter, args=(model, test_query))
        threads.append(thread)
        thread.start()
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll OpenRouter models have responded.")

    # You can still use your original class like this:
    # print("\n--- Querying Gemini using the original class ---")
    # vlm = QueryVLM("gemini")
    # gemini_response = vlm.query("What is a black hole?")
    # if gemini_response:
    #     print(gemini_response)