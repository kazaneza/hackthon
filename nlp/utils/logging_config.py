"""
Logging configuration for the Bank of Kigali AI Assistant application.
"""

import logging
import sys

def setup_logging():
    """Configure logging for the application."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log"),
        ]
    )
    
    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    # Create logger
    logger = logging.getLogger("bank_of_kigali_assistant")
    return logger

# Application logger
logger = setup_logging()