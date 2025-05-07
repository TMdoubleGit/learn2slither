# PYTHON := python3.12
# VENV := .venv
# PIP := $(VENV)/bin/pip

# install:
# 	@echo "Creating virtual environment..."
# 	@$(PYTHON) -m venv $(VENV)
# 	@echo "Activating and installing requirements..."
# 	@$(PIP) install --upgrade pip setuptools
# 	@if [ -f requirements.txt ]; then \
# 		$(PIP) install -r requirements.txt; \
# 	fi
# 	@echo "✅ Setup complete."

# freeze:
# 	@$(PIP) freeze > requirements.txt

# clean:

# fclean: clean
# 	@rm -rf $(VENV) ./data

# .PHONY: install freeze clean fclean



python := python3

define venvWrapper
	{\
	. .venv/bin/activate; \
	$1; \
	}
endef


install:
	@{ \
		echo "Setting up..."; \
		python3 -m venv .venv; \
		. .venv/bin/activate; \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
			echo "Installing dependencies...DONE"; \
		fi; \
	}

freeze:
	$(call venvWrapper, pip freeze > requirements.txt)

clean:


fclean: clean
	@rm -rf .venv/ .venv/bin/ .venv/include/ .venv/lib/ .venv/lib64 .venv/pyvenv.cfg .venv/share/
	@rm -rf ./data

phony: install freeze clean fclean
