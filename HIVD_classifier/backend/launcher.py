"""
HIVD Classifier - Standalone Launcher
Maltezos, V. (2026). A framework for efficient image categorisation in social research

This script launches the HIVD Classifier as a standalone application.
It starts the FastAPI server and opens the web browser automatically.
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# Determine if we're running as a bundled executable
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(os.path.dirname(sys.executable))
else:
    # Running as script
    BASE_DIR = Path(__file__).parent.parent
    APP_DIR = BASE_DIR

# Set up paths
os.chdir(APP_DIR)
sys.path.insert(0, str(BASE_DIR / 'backend'))

# Ensure data directories exist
(APP_DIR / 'image_dataset').mkdir(exist_ok=True)
(APP_DIR / 'inference_dataset').mkdir(exist_ok=True)
(APP_DIR / 'predictions').mkdir(exist_ok=True)
(APP_DIR / 'classes').mkdir(exist_ok=True)


def open_browser():
    """Open browser after a short delay to allow server to start"""
    time.sleep(2.5)
    webbrowser.open('http://127.0.0.1:8000')


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("  HIVD Classifier - Zero-Shot Image Classification System")
    print("  " + "-" * 66)
    print("  Maltezos, V. (2026). A framework for efficient image categorisation")
    print("  in social research: Addressing high intra-class visual diversity.")
    print("  Dissertationes Universitatis Helsingiensis (27/2026).")
    print("=" * 70)
    print("\n  Starting server...")
    print("  Web interface will open automatically at: http://127.0.0.1:8000")
    print("  Press Ctrl+C to stop the server\n")


def main():
    """Main entry point"""
    print_banner()
    
    # Start browser opener in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Import and run the FastAPI app
    import uvicorn
    
    # Import main module
    if getattr(sys, 'frozen', False):
        # When frozen, we need to import from the bundled location
        sys.path.insert(0, str(BASE_DIR / 'backend'))
    
    from main import app
    
    # Run the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,  # Disable reload for production
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
