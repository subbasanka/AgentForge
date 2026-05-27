from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api.routes import router
from src.observability.langfuse_client import get_langfuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting AgentForge service")
    get_langfuse()
    yield
    logger.info("Shutting down AgentForge service")
    lf = get_langfuse()
    lf.flush()


app = FastAPI(
    title="AgentForge API",
    description="AgentForge — Production-grade multi-agent task orchestration with LangGraph, MCP, and Langfuse",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
