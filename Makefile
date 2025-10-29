.PHONY: install format test docs docs-serve clean

# Install dependencies
install:
	pip install -e ".[dev]"

# Format code with Black
format:
	black . -l 79

# Run tests (will add tests in Phase 1)
test:
	pytest tests/ -v

# Build documentation
docs:
	cd jupyterbook && myst build --html

# Serve documentation locally
docs-serve:
	cd jupyterbook && myst start

# Clean build artifacts
clean:
	rm -rf jupyterbook/_build
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
	@echo "  make docs         - Build Jupyter Book documentation"
	@echo "  make docs-serve   - Serve documentation locally"
	@echo "  make docs-quick   - Build and serve documentation"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
