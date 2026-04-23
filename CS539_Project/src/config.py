"""Centralized runtime configuration for the project.

Keeping environment access in one module makes future maintenance easier
because the rest of the codebase can depend on typed attributes instead of
scattered ``os.getenv`` lookups.
"""
import os
from dotenv import load_dotenv

# Load variables from a local .env file when present.
load_dotenv()


class Config:
    """Small namespace of settings used by the API, agent, and tools."""
    
    # Accept either env var name so notebook and server launches can share one setup.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Defaults below are conservative and mainly act as soft guidance for prompts.
    MAX_ROWS_TO_DISPLAY = 10
    MAX_COLUMNS_TO_ANALYZE = 50
    CORRELATION_THRESHOLD = 0.5
    
    # Visualization Configuration
    FIGURE_SIZE = (10, 6)
    DPI = 100
    STYLE = "seaborn-v0_8-darkgrid"
    
    # Output locations are referenced by both the API and the visualization tool.
    OUTPUT_DIR = "outputs"
    SAVE_FIGURES = True
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "in your .env or environment before starting the server."
            )
        return True


# Validation happens when the agent is created so imports stay side-effect light.
