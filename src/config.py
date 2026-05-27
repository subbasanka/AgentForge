from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # LLM
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o")
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")

    # Search APIs
    exa_api_key: str = Field(default="")
    tavily_api_key: str = Field(default="")

    # Langfuse
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="http://localhost:3000")

    # MCP
    mcp_server_url: str = Field(default="http://localhost:8001")

    # Token budgets
    supervisor_token_budget: int = Field(default=4000)
    researcher_token_budget: int = Field(default=8000)
    analyst_token_budget: int = Field(default=8000)
    writer_token_budget: int = Field(default=10000)
    reviewer_token_budget: int = Field(default=6000)

    # Retry
    max_retries: int = Field(default=3)
    retry_base_delay: float = Field(default=1.0)

    # Cost tracking (per 1M tokens)
    input_cost_per_m: float = Field(default=2.50)
    output_cost_per_m: float = Field(default=10.00)


settings = Settings()
