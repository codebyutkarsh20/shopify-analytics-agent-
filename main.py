"""
Shopify Analytics Agent - Main Entry Point

A Python-based Shopify analytics agent that connects to a Shopify store
via MCP server, provides natural language analysis through Telegram,
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

from src.config.settings import settings
from src.database.init_db import init_database
from src.database.operations import DatabaseOperations
from src.services.mcp_service import MCPService
from src.services.shopify_service import ShopifyService
from src.services.claude_service import ClaudeService
from src.services.analytics_service import AnalyticsService
from src.learning.pattern_learner import PatternLearner
from src.learning.context_builder import ContextBuilder
from src.learning.preference_manager import PreferenceManager
from src.bot.handlers import MessageHandler
from src.bot.commands import BotCommands
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class ShopifyAnalyticsBot:
    """Main application class that wires all components together."""

    def __init__(self):
        self.mcp_service: MCPService | None = None
        self.application: Application | None = None
        self._running = False

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

        # Initialize database
        logger.info("Initializing database...")
        db_ops = init_database()
        if not db_ops:
            logger.error("Failed to initialize database")
            return False
        logger.info("Database initialized successfully")

        # Initialize MCP service
        logger.info("Starting MCP service...")
        self.mcp_service = MCPService(settings)
        try:
            await self.mcp_service.start()
            logger.info("MCP service started successfully")
        except Exception as e:
            logger.warning(
                "MCP service failed to start - bot will work but cannot fetch Shopify data",
                error=str(e),
            )
            # Continue anyway - user might need to connect store first

        # Initialize services
        shopify_service = ShopifyService(self.mcp_service)
        context_builder = ContextBuilder(db_ops)
        claude_service = ClaudeService(
            settings=settings,
            mcp_service=self.mcp_service,
            db_ops=db_ops,
            context_builder=context_builder,
        )
        analytics_service = AnalyticsService(shopify_service, db_ops)

        # Initialize learning system
        pattern_learner = PatternLearner(db_ops)
        preference_manager = PreferenceManager(
            db_ops, pattern_threshold=settings.learning_pattern_threshold
        )

        # Initialize bot handlers
        # Note: bot_commands is created first so it can be passed to message_handler
        # for handling the connect flow (credentials input)
        bot_commands = BotCommands(
            db_ops=db_ops,
            preference_manager=preference_manager,
        )
        message_handler = MessageHandler(
            claude_service=claude_service,
            pattern_learner=pattern_learner,
            preference_manager=preference_manager,
            db_ops=db_ops,
            bot_commands=bot_commands,
        )

        # Build Telegram application
        logger.info("Building Telegram bot application...")
        self.application = (
            Application.builder()
            .token(settings.telegram.bot_token)
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

        # Register callback query handler (for inline keyboards)
        self.application.add_handler(
            CallbackQueryHandler(
                bot_commands.handle_forget_callback,
                pattern="^forget_",
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

        logger.info("Bot application built successfully")
        return True

    async def run(self):
        """Run the bot."""
        success = await self.initialize()
        if not success:
            sys.exit(1)

        self._running = True
        logger.info("Starting bot polling...")
        print("\n✅ Shopify Analytics Bot is running!")
        print("Press Ctrl+C to stop.\n")

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )

        # Keep running until stopped
        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            logger.info("Received shutdown signal", signal=sig)
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        await stop_event.wait()
        await self.shutdown()

    async def shutdown(self):
        """Gracefully shut down all services."""
        logger.info("Shutting down bot...")
        self._running = False

        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

        if self.mcp_service:
            await self.mcp_service.stop()
            logger.info("MCP service stopped")

        logger.info("Bot shutdown complete")
        print("\n👋 Bot stopped. Goodbye!")


def main():
    """Entry point."""
    print(
        """
╔══════════════════════════════════════════╗
║   Shopify Analytics Agent               ║
║   Powered by Claude AI + MCP            ║
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
