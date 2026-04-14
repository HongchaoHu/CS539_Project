#!/usr/bin/env python3
"""
Data Analysis Agent Launcher
Starts the server and opens the web browser automatically
"""

import subprocess
import webbrowser
import time
import sys
import os

def main():
    print("=" * 50)
    print("Data Analysis Agent - Starting...")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("Error: app.py not found!")
        print("Please run this script from the project directory.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found!")
        print("Please create a .env file with your GEMINI_API_KEY")
        print()
    
    # Start the server
    print("Starting API server...")
    try:
        # Start server in a new process
        if sys.platform == "win32":
            # Windows - start in new window
            subprocess.Popen(
                ["python", "-m", "uvicorn", "app:app", "--port", "8080", "--reload"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Unix-like systems
            subprocess.Popen(
                ["python", "-m", "uvicorn", "app:app", "--port", "8080", "--reload"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Wait for server to start
        print("Waiting for server to initialize...")
        time.sleep(3)
        
        # Open browser
        print("Opening web browser...")
        webbrowser.open("http://localhost:8080")
        
        print()
        print("=" * 50)
        print("Application is running!")
        print("Web interface: http://localhost:8080")
        print("=" * 50)
        print()
        print("To stop the server:")
        print("- Close the server window")
        print("- Or press Ctrl+C in the server terminal")
        print()
        input("Press Enter to exit this launcher...")
        
    except Exception as e:
        print(f"Error starting server: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
