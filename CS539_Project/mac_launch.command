#!/bin/bash
echo "===================================="
echo "Data Analysis Agent - Starting..."
echo "===================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists, if not try to create it
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "Installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
else
    # Activate existing virtual environment
    source .venv/bin/activate
fi

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "Error: Python is not available in virtual environment"
    read -p "Press Enter to exit..."
    exit 1
fi

# Start the server in the background
echo "Starting API server..."
python -m uvicorn app:app --port 8080 --reload &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 3

# Open browser
echo "Opening web browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8080
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:8080 2>/dev/null || echo "Please open http://localhost:8080 in your browser"
fi

echo ""
echo "===================================="
echo "Application is running!"
echo "Web interface: http://localhost:8080"
echo "===================================="
echo ""
echo "Server PID: $SERVER_PID"
echo "To stop the server, run: kill $SERVER_PID"
echo ""
echo "Press Enter to exit this launcher (server will keep running)..."
read
