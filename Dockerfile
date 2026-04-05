FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml uv.lock ./
COPY inference.py run_baseline.py ./
COPY access_governance_env ./access_governance_env
COPY server ./server
COPY demo ./demo

RUN pip install --upgrade pip uv && uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
