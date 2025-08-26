from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config.openai_config import openai_client
from config.settings import Config
import streamlit as st

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.client = openai_client.client
        self.model = Config.OPENAI_MODEL
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main function"""
        pass
    
    def call_openai(self, messages: list, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Make OpenAI API call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error in {self.name}: {str(e)}")
            raise
    
    def update_progress(self, message: str):
        """Update progress in Streamlit"""
        if 'progress_placeholder' in st.session_state:
            st.session_state.progress_placeholder.info(f"🤖 {self.name}: {message}")
    
    def format_system_prompt(self, prompt_template: str, **kwargs) -> str:
        """Format system prompt with variables"""
        return prompt_template.format(**kwargs)
    
    def get_prompt_from_file(self, filename: str) -> str:
        """Load prompt from file"""
        try:
            with open(f"prompts/{filename}", 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""