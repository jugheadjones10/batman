# Development Server Script

Launch both backend and frontend development servers for local Batman development.

## Basic Usage

```bash
./scripts/run_dev.sh
```

## What It Does

The script automatically:

1. **Checks dependencies**: Verifies `uv` and `npm` are installed
2. **Syncs Python dependencies**: Runs `uv sync`
3. **Installs frontend dependencies**: Runs `npm install` if needed
4. **Creates data directory**: Ensures `data/projects` exists
5. **Starts backend server**: FastAPI on http://localhost:8000
6. **Starts frontend server**: Vite dev server on http://localhost:5173
7. **Handles cleanup**: Gracefully stops servers on Ctrl+C

## Output

```bash
$ ./scripts/run_dev.sh

🦇 Starting Batman Development Servers...

✓ Checking dependencies...
✓ Syncing Python dependencies...
✓ Installing frontend dependencies...
✓ Creating data directories...
✓ Starting backend server...
✓ Starting frontend server...

🚀 Services running:
   Frontend: http://localhost:5173
   Backend:  http://localhost:8000
   API Docs: http://localhost:8000/docs

Press Ctrl+C to stop all servers...

[Backend] INFO: Uvicorn running on http://0.0.0.0:8000
[Frontend] VITE v5.0.0 ready in 543 ms
[Frontend] ➜ Local: http://localhost:5173/
```

## Services

### Backend Server
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Framework**: FastAPI with Uvicorn
- **Auto-reload**: Enabled

### Frontend Server
- **URL**: http://localhost:5173
- **Framework**: Vite + React
- **Hot Module Replacement**: Enabled

## Stopping Servers

Press `Ctrl+C` to stop both servers:

```bash
^C
🛑 Stopping servers...
✓ Servers stopped
```

## Manual Start (Alternative)

If you prefer manual control:

### Terminal 1 - Backend

```bash
cd backend
uv run python -m backend.app.main

# or using the batman command
uv run batman
```

### Terminal 2 - Frontend

```bash
cd frontend
npm run dev
```

## Prerequisites

### System Requirements

- **Python 3.11+**
- **Node.js 18+**
- **uv** - Python package manager
- **npm** - Node package manager

### Installation

#### macOS

```bash
brew install uv node
```

#### Linux (Ubuntu/Debian)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## Configuration

### Backend Settings

Edit `backend/app/config.py` or use environment variables:

```bash
# .env file
BATMAN_HOST=127.0.0.1
BATMAN_PORT=8000
BATMAN_DEBUG=true
BATMAN_DATA_DIR=./data
```

### Frontend Settings

Edit `frontend/vite.config.ts` for Vite configuration:

```typescript
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

## Development Workflow

### 1. Start Servers

```bash
./scripts/run_dev.sh
```

### 2. Develop

- **Frontend**: Edit files in `frontend/src/`, changes reload automatically
- **Backend**: Edit files in `backend/app/`, server restarts automatically

### 3. Test

Open http://localhost:5173 in your browser

### 4. Stop Servers

Press `Ctrl+C` when done

## Troubleshooting

### Port Already in Use

Kill existing processes:

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 5173
lsof -ti:5173 | xargs kill -9
```

### uv Not Found

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### npm Not Found

Install Node.js and npm:

```bash
# macOS
brew install node

# Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Frontend Dependencies Failed

Clean and reinstall:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Backend Import Errors

Resync dependencies:

```bash
uv sync --force
```

### FFmpeg Not Found

Install FFmpeg:

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

## Tips

### 1. Use Multiple Terminals

For more control, run servers separately:

```bash
# Terminal 1
uv run python -m backend.app.main

# Terminal 2
cd frontend && npm run dev
```

### 2. Enable Debug Logging

```bash
export BATMAN_DEBUG=true
./scripts/run_dev.sh
```

### 3. Custom Ports

Edit the script or run manually with custom ports:

```bash
# Backend
uv run uvicorn backend.app.main:app --port 8080

# Frontend
cd frontend
npm run dev -- --port 3000
```

### 4. Access from Other Devices

Bind to all interfaces:

```bash
# Backend
uv run uvicorn backend.app.main:app --host 0.0.0.0

# Frontend  
cd frontend
npm run dev -- --host 0.0.0.0
```

Then access via your machine's IP address:
- Frontend: `http://<your-ip>:5173`
- Backend: `http://<your-ip>:8000`

## Related

- **[Getting Started](../getting-started.md)** - Installation guide
- **[API Reference](../api/index.md)** - Backend API documentation
- **[CLI Tools](../cli/index.md)** - Command-line tools
