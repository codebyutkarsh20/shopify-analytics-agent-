"""
Shopify Analytics Agent - Main Entry Point

A Python-based Shopify analytics agent that connects to a Shopify store
via direct GraphQL API, provides natural language analysis through Telegram,
and learns from user interactions to provide increasingly relevant insights.
"""

import asyncio
import signal
import sys
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler as TelegramMessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.request import HTTPXRequest

from src.config.settings import settings
from src.database.init_db import init_database
from src.database.operations import DatabaseOperations
from src.services.chart_generator import ChartGenerator
from src.services.llm_factory import create_llm_service
from src.services.shopify_graphql import ShopifyGraphQLClient
from src.learning.pattern_learner import PatternLearner
from src.learning.context_builder import ContextBuilder
from src.learning.preference_manager import PreferenceManager
from src.learning.session_manager import SessionManager
from src.learning.template_manager import TemplateManager
from src.learning.recovery_manager import RecoveryManager
from src.learning.feedback_analyzer import FeedbackAnalyzer
from src.learning.insight_aggregator import InsightAggregator
from src.learning.vector_store import EmbeddingStore
from src.bot.handlers import MessageHandler
from src.bot.commands import BotCommands
from src.bot.telegram_adapter import TelegramAdapter
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class ShopifyAnalyticsBot:
    """Main application class that wires all components together."""

    def __init__(self):
        self.application: Application | None = None
        self._whatsapp_handler = None
        self._whatsapp_runner = None
        self._running = False
        self._db_ops: DatabaseOperations | None = None

    async def initialize(self) -> bool:
        """Initialize all services and components."""
        logger.info("Initializing Shopify Analytics Bot...")

        # Validate configuration
        missing = settings.validate()
        if missing:
            logger.error("Missing required configuration", missing=missing)
            print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
            print("Please copy .env.example to .env and fill in your credentials.\n")
            return False

        # Setup logging
        setup_logging(
            level=settings.logging.level,
            log_file=settings.logging.log_file,
        )
        logger.info("Logging configured", level=settings.logging.level)

        # Ensure data and logs directories exist
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        # Initialize database (creates tables, runs migrations, seeds templates)
        # We pass vector_store so seed templates get embedded on first boot.
        # EmbeddingStore needs db_ops, and init_database returns db_ops — so we
        # do a two-step: basic init first, then vector_store, then seed/backfill.
        logger.info("Initializing database...")
        db_ops = init_database()
        if not db_ops:
            logger.error("Failed to initialize database")
            return False
        self._db_ops = db_ops
        logger.info("Database initialized successfully")

        # Initialize vector store for semantic search (lazy-loads embedding model)
        vector_store = EmbeddingStore(db_ops, lazy_load=True)
        logger.info("Vector store initialized (model loads on first use)")

        # Backfill embeddings for any templates that don't have one yet
        try:
            vector_store.backfill_templates()
        except Exception as e:
            logger.warning("Embedding backfill skipped: %s", e)

        # Initialize core services
        context_builder = ContextBuilder(db_ops, vector_store=vector_store)

        # Initialize Shopify GraphQL client (direct API — no MCP subprocess needed)
        graphql_client = None
        if settings.shopify.shop_domain and settings.shopify.access_token:
            graphql_client = ShopifyGraphQLClient(settings)
            logger.info("ShopifyGraphQLClient initialized for direct Shopify API access")
        else:
            logger.warning("GraphQL client not initialized — missing Shopify credentials")

        # Initialize learning system components
        pattern_learner = PatternLearner(db_ops)
        preference_manager = PreferenceManager(
            db_ops, pattern_threshold=settings.learning_pattern_threshold
        )
        session_manager = SessionManager(db_ops, settings)
        template_manager = TemplateManager(db_ops, vector_store=vector_store)
        recovery_manager = RecoveryManager(db_ops)
        feedback_analyzer = FeedbackAnalyzer()
        insight_aggregator = InsightAggregator(db_ops)
        logger.info("Learning system components initialized")

        # Initialize channel adapter
        telegram_adapter = TelegramAdapter()
        logger.info("Telegram adapter initialized")

        # Initialize chart generator for visual analytics
        chart_generator = ChartGenerator()
        logger.info("Chart generator initialized")

        # Initialize LLM service (Anthropic or OpenAI based on LLM_PROVIDER env var)
        llm_service = create_llm_service(
            settings=settings,
            db_ops=db_ops,
            context_builder=context_builder,
            graphql_client=graphql_client,
            template_manager=template_manager,
            recovery_manager=recovery_manager,
            chart_generator=chart_generator,
        )

        # Initialize cross-channel linker (shared between Telegram + WhatsApp)
        from src.services.channel_linker import ChannelLinker
        channel_linker = ChannelLinker(db_ops) if settings.whatsapp.enabled else None
        if channel_linker:
            logger.info("ChannelLinker initialized for cross-channel identity linking")

        # Initialize bot handlers
        bot_commands = BotCommands(
            db_ops=db_ops,
            preference_manager=preference_manager,
            channel_linker=channel_linker,
        )
        message_handler = MessageHandler(
            claude_service=llm_service,
            pattern_learner=pattern_learner,
            preference_manager=preference_manager,
            db_ops=db_ops,
            bot_commands=bot_commands,
            session_manager=session_manager,
            feedback_analyzer=feedback_analyzer,
            template_manager=template_manager,
            insight_aggregator=insight_aggregator,
            telegram_adapter=telegram_adapter,
        )

        # Build Telegram application
        logger.info("Building Telegram bot application...")
        # Increase connection timeouts to handle slow networks
        request = HTTPXRequest(
            connection_pool_size=16,
            connect_timeout=60.0,
            read_timeout=60.0,
            write_timeout=60.0,
        )
        self.application = (
            Application.builder()
            .token(settings.telegram.bot_token)
            .request(request)
            .build()
        )

        # Register command handlers
        self.application.add_handler(
            CommandHandler("start", bot_commands.start_command)
        )
        self.application.add_handler(
            CommandHandler("help", bot_commands.help_command)
        )
        self.application.add_handler(
            CommandHandler("connect", bot_commands.connect_command)
        )
        self.application.add_handler(
            CommandHandler("settings", bot_commands.settings_command)
        )
        self.application.add_handler(
            CommandHandler("forget", bot_commands.forget_command)
        )
        self.application.add_handler(
            CommandHandler("status", bot_commands.status_command)
        )
        self.application.add_handler(
            CommandHandler("verify", bot_commands.verify_command)
        )
        self.application.add_handler(
            CommandHandler("link", bot_commands.link_command)
        )

        # Register callback query handlers (for inline keyboards)
        self.application.add_handler(
            CallbackQueryHandler(
                bot_commands.handle_forget_callback,
                pattern="^forget_",
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                message_handler.handle_feedback_callback,
                pattern="^fb_",
            )
        )

        # Register message handler (must be last - catches all text messages)
        self.application.add_handler(
            TelegramMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                message_handler.handle_message,
            )
        )

        # Register error handler
        self.application.add_error_handler(message_handler.handle_error)

        logger.info("Telegram bot application built successfully")

        # ── Initialize WhatsApp channel (optional) ─────────────────
        if settings.whatsapp.enabled:
            from src.bot.whatsapp_handler import WhatsAppWebhookHandler

            self._whatsapp_handler = WhatsAppWebhookHandler(
                llm_service=llm_service,
                pattern_learner=pattern_learner,
                preference_manager=preference_manager,
                db_ops=db_ops,
                session_manager=session_manager,
                feedback_analyzer=feedback_analyzer,
                template_manager=template_manager,
                insight_aggregator=insight_aggregator,
                channel_linker=channel_linker,
            )
            logger.info(
                "WhatsApp webhook handler initialized",
                port=settings.whatsapp.webhook_port,
                provider=settings.whatsapp.provider,
            )
        else:
            logger.info("WhatsApp channel disabled (set WHATSAPP_ENABLED=true to enable)")

        return True

    async def run(self):
        """Run the bot (Telegram polling + optional WhatsApp webhook server)."""
        success = await self.initialize()
        if not success:
            sys.exit(1)

        self._running = True
        logger.info("Starting bot polling...")
        print("\n✅ Shopify Analytics Bot is running!")

        # Start Telegram polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        print("  📱 Telegram channel: active (polling)")

        # Start WhatsApp webhook server (if enabled)
        if self._whatsapp_handler:
            self._whatsapp_runner = await self._whatsapp_handler.start()
            print(f"  💬 WhatsApp channel: active (webhook on port {settings.whatsapp.webhook_port})")
        else:
            print("  💬 WhatsApp channel: disabled")

        print("\nPress Ctrl+C to stop.\n")

        # Keep running until stopped
        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            logger.info("Received shutdown signal", signal=sig)
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start periodic cache cleanup (every 30 minutes)
        cache_cleanup_task = asyncio.create_task(
            self._periodic_cache_cleanup()
        )

        await stop_event.wait()
        cache_cleanup_task.cancel()
        await self.shutdown()

    async def _periodic_cache_cleanup(self, interval_seconds: int = 1800):
        """Periodically clean up expired analytics cache entries.

        Runs every ``interval_seconds`` (default 30 min) and deletes
        expired rows from the analytics_cache table to prevent unbounded
        growth.
        """
        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                if self._db_ops:
                    deleted = self._db_ops.clear_expired_cache()
                    if deleted:
                        logger.info("Cache cleanup: removed %d expired entries", deleted)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Cache cleanup error (non-critical)", error=str(e))

    async def shutdown(self):
        """Gracefully shut down all services."""
        logger.info("Shutting down bot...")
        self._running = False

        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

        if self._whatsapp_handler:
            await self._whatsapp_handler.shutdown()
            logger.info("WhatsApp handler stopped")
        if self._whatsapp_runner:
            await self._whatsapp_runner.cleanup()
            logger.info("WhatsApp webhook server stopped")

        logger.info("Bot shutdown complete")
        print("\n👋 Bot stopped. Goodbye!")


def main():
    """Entry point."""
    print(
        """
╔══════════════════════════════════════════╗
║   Shopify Analytics Agent               ║
║   Powered by Claude AI + GraphQL        ║
╚══════════════════════════════════════════╝
"""
    )

    bot = ShopifyAnalyticsBot()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
