# utils.py
import sys
from pathlib import Path

def get_resource_path(relative_path: str) -> Path:
    """
    Return the absolute path to a bundled resource (models, soundfonts, etc.).
    
    Works both:
    - In development mode (running from source)
    - In bundled mode (PyInstaller --onefile or --onedir)
    
    Usage:
        model_path = get_resource_path("models/demucs/htdemucs.pth")
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled executable (PyInstaller)
        # sys._MEIPASS is the temp folder where resources are extracted
        base_path = Path(sys._MEIPASS)
    else:
        # Running from source (normal Python execution)
        # __file__ is utils.py, so parent.parent gets to project root
        base_path = Path(__file__).parent
    
    return base_path / relative_path