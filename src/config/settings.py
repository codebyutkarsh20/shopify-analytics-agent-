"""
Configuration management for the Shopify Analytics Agent.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class TelegramConfig:
    bot_token: str = ""

    def __post_init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.bot_token)


@dataclass
class AnthropicConfig:
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4000

    def __post_init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", self.api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", self.model)


@dataclass
class OpenAIConfig:
    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 4000

    def __post_init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", self.api_key)
        self.model = os.getenv("OPENAI_MODEL", self.model)
        max_tokens_str = os.getenv("OPENAI_MAX_TOKENS", "")
        if max_tokens_str:
            self.max_tokens = int(max_tokens_str)


@dataclass
class ShopifyConfig:
    access_token: str = ""
    shop_domain: str = ""

    def __post_init__(self):
        self.access_token = os.getenv("SHOPIFY_ACCESS_TOKEN", self.access_token)
        self.shop_domain = os.getenv("SHOPIFY_SHOP_DOMAIN", self.shop_domain)


@dataclass
class DatabaseConfig:
    url: str = ""

    def __post_init__(self):
        default_url = f"sqlite:///{PROJECT_ROOT / 'data' / 'shopify_agent.db'}"
        self.url = os.getenv("DATABASE_URL", default_url)


@dataclass
class MCPConfig:
    server_command: str = "npx"
    server_args: list = field(default_factory=list)

    def __post_init__(self):
        self.server_command = os.getenv("MCP_SERVER_COMMAND", self.server_command)
        args_str = os.getenv("MCP_SERVER_ARGS", "-y,shopify-mcp-server")
        self.server_args = args_str.split(",")


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = ""

    def __post_init__(self):
        self.level = os.getenv("LOG_LEVEL", self.level)
        self.log_file = os.getenv("LOG_FILE", str(PROJECT_ROOT / "logs" / "agent.log"))


@dataclass
class Settings:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    shopify: ShopifyConfig = field(default_factory=ShopifyConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm_provider: str = ""

    # Application settings
    conversation_history_limit: int = 10
    learning_pattern_threshold: int = 5
    cache_ttl_seconds: int = 300  # 5 minutes
    analytics_cache_ttl_seconds: int = 3600  # 1 hour

    # Session management
    session_hard_timeout_seconds: int = 7200      # 2 hours — definitely new session
    session_soft_timeout_seconds: int = 1800      # 30 minutes — likely new session if topic changed
    session_min_gap_seconds: int = 300            # 5 minutes — min gap for topic-shift detection
    aggregation_interval: int = 50                # Run global insight aggregation every N interactions
    max_context_tokens: int = 2000                # Token budget for learning context in system prompt

    def __post_init__(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "anthropic")

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of missing items."""
        missing = []
        if not self.telegram.bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")

        # Check for appropriate LLM API key based on provider
        if self.llm_provider == "openai":
            if not self.openai.api_key:
                missing.append("OPENAI_API_KEY")
        else:
            # Default to Anthropic
            if not self.anthropic.api_key:
                missing.append("ANTHROPIC_API_KEY")

        if not self.shopify.access_token:
            missing.append("SHOPIFY_ACCESS_TOKEN")
        if not self.shopify.shop_domain:
            missing.append("SHOPIFY_SHOP_DOMAIN")
        return missing


# Global settings singleton
settings = Settings()
