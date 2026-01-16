.PHONY: help install dev backend frontend clean test lint format

# Default target
help:
	@echo "Batman - Video Auto-Label Application"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install    Install all dependencies"
	@echo "  dev        Run both backend and frontend in development mode"
	@echo "  backend    Run backend server only"
	@echo "  frontend   Run frontend dev server only"
	@echo "  test       Run tests"
	@echo "  lint       Run linters"
	@echo "  format     Format code"
	@echo "  clean      Clean build artifacts"

# Install all dependencies
install:
	uv sync
	cd frontend && npm install

# Run development servers (requires two terminals or use tmux/screen)
dev:
	@echo "Starting development servers..."
	@echo "Run 'make backend' in one terminal and 'make frontend' in another"

# Run backend server
backend:
	uv run python -m backend.app.main

# Run frontend dev server
frontend:
	cd frontend && npm run dev

# Run tests
test:
	uv run pytest backend/tests -v

# Run linters
lint:
	uv run ruff check backend/
	cd frontend && npm run lint

# Format code
format:
	uv run ruff format backend/
	uv run ruff check --fix backend/

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf backend/__pycache__ backend/**/__pycache__
	rm -rf frontend/dist frontend/node_modules/.cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Build frontend for production
build-frontend:
	cd frontend && npm run build

# Export dataset (requires project name)
export-dataset:
	@echo "Usage: make export-dataset PROJECT=<project_name>"
	@test -n "$(PROJECT)" || (echo "PROJECT is required" && exit 1)
	uv run python -c "from backend.app.services.dataset_exporter import DatasetExporter; import asyncio; asyncio.run(DatasetExporter('data/projects/$(PROJECT)').export([], [], []))"

# Download model weights
download-models:
	mkdir -p models
	@echo "Downloading SAM2 weights..."
	wget -O models/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
	@echo "Done!"

