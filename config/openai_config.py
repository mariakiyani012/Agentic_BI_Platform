from openai import OpenAI
from config.settings import Config
import streamlit as st

class OpenAIClient:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                Config.validate()
                self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                st.error(f"OpenAI configuration error: {str(e)}")
                raise
    
    @property
    def client(self):
        return self._client
    
    def test_connection(self):
        """Test OpenAI API connection"""
        try:
            response = self._client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

# Global instance
openai_client = OpenAIClient()