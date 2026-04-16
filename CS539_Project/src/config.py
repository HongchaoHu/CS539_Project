"""Configuration settings for the Data Analysis Agent"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the agent"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Analysis Configuration
    MAX_ROWS_TO_DISPLAY = 10
    MAX_COLUMNS_TO_ANALYZE = 50
    CORRELATION_THRESHOLD = 0.5
    
    # Visualization Configuration
    FIGURE_SIZE = (10, 6)
    DPI = 100
    STYLE = "seaborn-v0_8-darkgrid"
    
    # Output Configuration
    OUTPUT_DIR = "outputs"
    SAVE_FIGURES = True
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file. "
                "Copy .env.example to .env and add your API key."
            )
        return True


# Note: Validation is performed when the agent is initialized, not at import time
