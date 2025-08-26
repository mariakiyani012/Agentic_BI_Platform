import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Application Settings
    APP_NAME = os.getenv("APP_NAME", "Agentic BI Platform")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Streamlit Settings
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    
    # File Upload Settings
    MAX_FILE_SIZE_MB = 200
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    
    # Analysis Settings
    MAX_ROWS_PREVIEW = 1000
    MAX_COLUMNS_ANALYSIS = 50
    
    # Chart Settings
    DEFAULT_CHART_HEIGHT = 400
    DEFAULT_CHART_WIDTH = 600
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True