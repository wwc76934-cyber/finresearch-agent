FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/packages
COPY pyproject.toml ./
RUN pip install --no-cache-dir "pydantic>=2.7" "sqlalchemy>=2.0" "fastapi>=0.110" "uvicorn>=0.29" "langgraph>=0.2" \
    "numpy>=1.26" "pandas>=2.0" "pyyaml>=6.0" "httpx>=0.27" "psycopg[binary]>=3.1" "pytest>=8"
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "python scripts/bootstrap.py && python -c 'import sys; sys.path.insert(0, \"packages\"); from evaluation_platform.datasets import write_tasks; write_tasks()' && uvicorn apps.api_server.main:app --host 0.0.0.0 --port 8000"]
