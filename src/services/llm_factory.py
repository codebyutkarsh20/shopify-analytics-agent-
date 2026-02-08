"""Factory for creating LLM service instances based on configuration."""

from typing import Optional

from src.config.settings import Settings
from src.database.operations import DatabaseOperations
from src.learning.context_builder import ContextBuilder
from src.learning.template_manager import TemplateManager
from src.learning.recovery_manager import RecoveryManager
from src.services.shopify_graphql import ShopifyGraphQLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_llm_service(
    settings: Settings,
    db_ops: DatabaseOperations,
    context_builder: Optional[ContextBuilder] = None,
    graphql_client: Optional[ShopifyGraphQLClient] = None,
    template_manager: Optional[TemplateManager] = None,
    recovery_manager: Optional[RecoveryManager] = None,
):
    """Create the appropriate LLM service based on settings.llm_provider.

    Args:
        settings: Application settings
        db_ops: Database operations instance
        context_builder: Optional context builder
        graphql_client: Optional Shopify GraphQL client
        template_manager: Optional template manager
        recovery_manager: Optional recovery manager

    Returns:
        LLMService instance (AnthropicService or OpenAIService)
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from src.services.openai_service import OpenAIService
        logger.info("Creating OpenAI LLM service", model=settings.openai.model)
        return OpenAIService(
            settings=settings,
            db_ops=db_ops,
            context_builder=context_builder,
            graphql_client=graphql_client,
            template_manager=template_manager,
            recovery_manager=recovery_manager,
        )
    else:
        from src.services.anthropic_service import AnthropicService
        logger.info("Creating Anthropic LLM service", model=settings.anthropic.model)
        return AnthropicService(
            settings=settings,
            db_ops=db_ops,
            context_builder=context_builder,
            graphql_client=graphql_client,
            template_manager=template_manager,
            recovery_manager=recovery_manager,
        )
