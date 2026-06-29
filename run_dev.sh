#!/bin/bash
# Startup script for Medster Local LLM with web interface

echo "🏥 Starting Medster Local LLM..."
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama doesn't seem to be running on localhost:11434"
    echo "   Please start Ollama first: ollama serve"
    echo "   And pull the model: ollama pull qwen3.6:35b-mlx"
    echo ""
fi

# Start FastAPI backend
echo "🚀 Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate

# Set PYTHONPATH to include src directory (required for medster module)
export PYTHONPATH="$PWD/src:$PYTHONPATH"

uvicorn medster.api:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start Next.js frontend
echo "🎨 Starting Next.js frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Medster Local LLM is running!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
