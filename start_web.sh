#!/bin/bash
# Startup script for nuts_vision web interface

echo "=========================================="
echo "nuts_vision - Web Interface Launcher"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
if pip install -q --timeout 30 -r requirements.txt; then
    echo "✅ Dependencies ready"
else
    echo "⚠️  Failed to install dependencies (check your network connection)"
    echo "   If packages are already installed, the app may still work."
    echo "   Run: pip install -r requirements.txt"
fi
echo ""

# Set environment variables if .env exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment loaded"
else
    echo "ℹ️  No .env file found, using defaults"
    echo "   Create a .env file based on .env.example for custom configuration"
fi
echo ""

# Set default port if not configured
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

# Check if the port is available, try up to 10 consecutive ports
MAX_ATTEMPTS=10
ATTEMPT=0

is_port_in_use() {
    if command -v ss &> /dev/null; then
        ss -tlnH "sport = :$1" 2>/dev/null | grep -q .
    elif command -v lsof &> /dev/null; then
        lsof -iTCP:"$1" -sTCP:LISTEN -t &> /dev/null
    elif command -v netstat &> /dev/null; then
        netstat -tln 2>/dev/null | grep -q ":$1 "
    else
        return 1
    fi
}

while is_port_in_use "$STREAMLIT_PORT"; do
    echo "⚠️  Port $STREAMLIT_PORT is in use, trying next port..."
    STREAMLIT_PORT=$((STREAMLIT_PORT + 1))
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "❌ Could not find an available port after $MAX_ATTEMPTS attempts."
        echo "   Please free port 8501 or set STREAMLIT_PORT in your .env file."
        exit 1
    fi
done

# Launch Streamlit app
echo "=========================================="
echo "🚀 Starting nuts_vision Web Interface..."
echo "=========================================="
echo ""
echo "The application will open in your browser at:"
echo "👉 http://localhost:$STREAMLIT_PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py --server.port "$STREAMLIT_PORT" --server.address localhost
