FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY mcp_server/ mcp_server/

FROM base AS orchestrator
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS mcp-server
EXPOSE 8001
CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8001"]
