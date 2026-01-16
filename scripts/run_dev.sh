#!/bin/bash

# Batman Development Server Launcher
# This script starts both backend and frontend servers

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🦇 Starting Batman Development Servers${NC}"
echo -e "Project directory: ${PROJECT_DIR}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

# Always sync Python dependencies to ensure they're up to date
echo -e "${YELLOW}Syncing Python dependencies...${NC}"
uv sync

# Install frontend dependencies if needed
if [ -d "$PROJECT_DIR/frontend" ]; then
    if [ ! -d "$PROJECT_DIR/frontend/node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        cd "$PROJECT_DIR/frontend" && npm install
        cd "$PROJECT_DIR"
    fi
else
    echo -e "${YELLOW}Warning: frontend directory not found${NC}"
fi

# Create data directories
mkdir -p data/projects

# Start backend in background
echo -e "${GREEN}Starting backend server on http://localhost:8000${NC}"
uv run python -m backend.app.main &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend if directory exists
if [ -d "$PROJECT_DIR/frontend" ]; then
    echo -e "${GREEN}Starting frontend server on http://localhost:5173${NC}"
    cd "$PROJECT_DIR/frontend" && npm run dev &
    FRONTEND_PID=$!
    cd "$PROJECT_DIR"
else
    FRONTEND_PID=""
fi

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

echo ""
echo -e "${GREEN}🦇 Batman is ready!${NC}"
echo -e "   Frontend: ${YELLOW}http://localhost:5173${NC}"
echo -e "   Backend:  ${YELLOW}http://localhost:8000${NC}"
echo -e "   API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait

