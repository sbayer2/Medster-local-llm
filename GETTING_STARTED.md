# Getting Started with Medster Local LLM Web Interface

This guide will help you test the new Next.js + Tailwind CSS web interface for Medster-local-LLM.

## Prerequisites

Before starting, ensure you have:

1. **Ollama running** with at least one model pulled:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # If not running, start it
   ollama serve
   
   # Pull a model if you haven't already
   ollama pull gpt-oss:20b
   # OR
   ollama pull qwen3-vl:8b
   ```

2. **Python dependencies installed**:
   ```bash
   cd /Users/sbm4_mac/Desktop/Medster-local-LLM
   uv sync
   ```

3. **Frontend dependencies installed**:
   ```bash
   cd frontend
   npm install
   ```

## Quick Start

### Option 1: Use the Startup Script (Recommended)

The easiest way to start both services:

```bash
cd /Users/sbm4_mac/Desktop/Medster-local-LLM
./run_dev.sh
```

This will automatically start:
- **Backend (FastAPI)** on `http://localhost:8000`
- **Frontend (Next.js)** on `http://localhost:3000`

Then open your browser to: **http://localhost:3000**

### Option 2: Start Services Manually

If you prefer to run services in separate terminals:

**Terminal 1 - Backend:**
```bash
cd /Users/sbm4_mac/Desktop/Medster-local-LLM
source .venv/bin/activate
uvicorn medster.api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /Users/sbm4_mac/Desktop/Medster-local-LLM/frontend
npm run dev
```

## Testing the Web Interface

### 1. Initial Connection Test

1. Open browser to `http://localhost:3000`
2. Check the status indicator in the top-left:
   - Should show "Connected" with a green pulsing dot
3. Verify the model selector in the top-right shows available models

### 2. Model Selection Test

1. Click the model selector dropdown
2. You should see two models:
   - **gpt-oss:20b** (TEXT-ONLY) - Faster inference
   - **qwen3-vl:8b** (TEXT + IMAGES) - Multimodal vision
3. Select a model
4. Verify a system message appears confirming the switch

### 3. Basic Query Test

Try a simple query to test the agent:

```
Get patient demographics for patient 12345
```

**Expected behavior:**
- Your message appears in a blue gradient bubble on the right
- Status indicator shows "Processing..." with animated dots
- System messages appear showing agent tasks (e.g., "📋 Task: Retrieve patient demographics")
- Tool execution messages appear (e.g., "🔧 Using tool: get_demographics")
- Final answer appears in a dark bubble on the left

### 4. Complex Query Test

Try a more complex clinical analysis:

```
Analyze the last 7 days of lab results for patient 12345 and identify any critical values or concerning trends
```

**Expected behavior:**
- Multiple task messages as the agent plans and executes
- Tool execution messages for lab retrieval
- Comprehensive clinical analysis in the final response
- Real-time streaming of all events

### 5. Error Handling Test

Test error handling by:

1. **Disconnect test**: Stop the backend (Ctrl+C in backend terminal)
   - Status should change to "Disconnected" with red dot
   - Send button should be disabled

2. **Reconnection test**: Restart the backend
   - Should automatically reconnect within a few seconds
   - Status returns to "Connected"

## Troubleshooting

### Backend won't start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd /Users/sbm4_mac/Desktop/Medster-local-LLM
uv sync
```

### Frontend won't start

**Error**: `Cannot find module 'react'`

**Solution**:
```bash
cd /Users/sbm4_mac/Desktop/Medster-local-LLM/frontend
npm install
```

### WebSocket connection fails

**Error**: Status shows "Disconnected"

**Check**:
1. Is the backend running on port 8000?
   ```bash
   curl http://localhost:8000
   ```
2. Check browser console (F12) for WebSocket errors
3. Verify CORS is not blocking (should be allowed for localhost:3000)

### Ollama not responding

**Error**: Agent queries timeout or fail

**Check**:
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check if your model is available
ollama list
```

### Port already in use

**Error**: `Address already in use` for port 8000 or 3000

**Solution**:
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

## API Documentation

Once the backend is running, you can view the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture Overview

```
┌─────────────────┐
│  Browser        │
│  localhost:3000 │
└────────┬────────┘
         │ WebSocket
         │ (real-time streaming)
         ▼
┌─────────────────┐
│  FastAPI        │
│  localhost:8000 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Medster Agent  │
│  (existing)     │
└────────┬────────┘
         │
         ├─────────────┐
         ▼             ▼
┌──────────────┐  ┌──────────────┐
│ Ollama       │  │ FHIR Data    │
│ Local LLMs   │  │ (Coherent)   │
└──────────────┘  └──────────────┘
```

## Next Steps

After successful testing:

1. **Customize the UI**: Edit files in `frontend/components/` and `frontend/app/`
2. **Adjust the theme**: Modify `frontend/tailwind.config.js` for colors
3. **Add features**: Extend the backend API in `src/medster/api.py`
4. **Deploy**: Build for production with `npm run build` in the frontend directory

## Support

If you encounter issues:

1. Check the browser console (F12) for frontend errors
2. Check the backend terminal for Python errors
3. Verify Ollama is running and models are available
4. Review the IMPLEMENTATION_PLAN.md for architecture details

---

**Enjoy your new Medster Local LLM web interface!** 🏥✨
