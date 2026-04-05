FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY access_governance_env ./access_governance_env
COPY demo ./demo

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "access_governance_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
