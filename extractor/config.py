"""Central configuration via Pydantic BaseSettings.

All values can be overridden by environment variables or a .env file.
Example: MODEL_NAME=Qwen/Qwen2.5-3B-Instruct python -m extractor.api.main
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model serving
    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model ID or local path",
    )
    model_revision: str = Field(default="main")
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # vLLM
    vllm_host: str = Field(default="localhost")
    vllm_port: int = Field(default=8000)
    vllm_api_key: str = Field(default="extractor-local")

    # FastAPI
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    log_level: str = Field(default="info")

    # Training
    output_dir: str = Field(default="checkpoints/")
    data_dir: str = Field(default="data/")
    seed: int = Field(default=42)

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"


settings = Settings()
