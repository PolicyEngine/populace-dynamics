.PHONY: install format test docs docs-serve docs-quick clean help

# Install dependencies
install:
	pip install -e ".[dev]"

# Format code with Black
format:
	black . -l 79

# Run tests if implementation tests exist
test:
	@if [ -d tests ]; then pytest tests/ -v; else echo "No tests directory yet; skipping."; fi

# Build Quarto documentation
docs:
	quarto render docs

# Serve documentation locally
docs-serve:
	quarto preview docs --port 3004 --no-browser

# Clean build artifacts
clean:
	rm -rf docs/_book
	rm -rf docs/.quarto
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick build and serve
docs-quick: docs docs-serve

# Help
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make format       - Format code with Black"
	@echo "  make test         - Run test suite"
	@echo "  make docs         - Build Quarto documentation"
	@echo "  make docs-serve   - Serve documentation locally"
	@echo "  make docs-quick   - Build and serve documentation"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
