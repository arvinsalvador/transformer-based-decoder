PYTHON ?= python
.PHONY: install install-dev test lint run docker-up docker-down gpu-up gpu-down
install:
	$(PYTHON) -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt
install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
run:
	$(PYTHON) -m streamlit run app/streamlit_app.py
docker-up:
	docker compose up --build -d
docker-down:
	docker compose down
gpu-up:
	docker compose -f docker-compose.gpu.yml up --build -d
gpu-down:
	docker compose -f docker-compose.gpu.yml down
