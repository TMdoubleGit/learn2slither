PYTHON := python3.12
VENV := .venv
PIP := $(VENV)/bin/pip

install:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Activating and installing requirements..."
	@$(PIP) install --upgrade pip setuptools
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	fi
	@echo "✅ Setup complete."

freeze:
	@$(PIP) freeze > requirements.txt

clean:

fclean: clean
	@rm -rf $(VENV) ./data

.PHONY: install freeze clean fclean
