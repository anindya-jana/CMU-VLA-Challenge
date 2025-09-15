import requests
import json
import os
import time

class QueryVLM:
    def __init__(self, model_name):
        self.model_name = "gemini"

    def query_gemini(self, query):
        api_key = "AIzaSyAG6Ewr5aTJsFxEFjiixkP_tt0SNfF2cZQ"
        secondary_api_key = "AIzaSyAxr9kb1s__HA20pgyOaC40Tu9XLYMIPjU"
        
        # Try with primary API key first
        result = self._make_gemini_request(query, api_key)
        
        # If primary key fails, try with secondary key
        if result is None:
            print("Primary API key failed, trying secondary API key...")
            result = self._make_gemini_request(query, secondary_api_key)
            
        return result

    def _make_gemini_request(self, query, api_key):
        """Helper method to make the actual API request"""
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
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Request failed with exception: {e}")
            return None

    def query(self, query):
        if self.model_name == "gemini":
            return self.query_gemini(query)
        else:
            print("Model not supported")
            return None