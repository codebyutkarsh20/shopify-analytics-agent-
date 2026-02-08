# Shopify Analytics Agent

A Python-based Shopify analytics agent with a Telegram bot interface, powered by Claude AI and MCP. Provides natural language analytics queries, learns from user interactions, and delivers increasingly relevant insights about your Shopify store performance.

## Features

- **Natural Language Queries** - Ask questions about your store in plain English
- **Revenue & Order Analysis** - Analyze sales trends, revenue metrics, and order patterns
- **AOV & Metrics** - Track Average Order Value and other key performance indicators
- **Time-Based Comparisons** - Compare metrics across different time periods
- **Learning System** - Bot learns from your interactions to provide better recommendations
- **MCP Integration** - Direct connection to Shopify store via Model Context Protocol server

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd shopify-analytics-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
- `TELEGRAM_BOT_TOKEN` - Get from [BotFather](https://t.me/botfather) on Telegram
- `ANTHROPIC_API_KEY` - Get from [Anthropic Console](https://console.anthropic.com)
- `SHOPIFY_ACCESS_TOKEN` - Generate in your Shopify Admin
- `SHOPIFY_SHOP_DOMAIN` - Your store domain (e.g., `mystore.myshopify.com`)

### 5. Run the Bot
```bash
python main.py
```

The bot will initialize, connect to your services, and start polling for messages.

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from BotFather | `123456789:ABCdefGHIjklmno` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | `sk-ant-...` |
| `SHOPIFY_ACCESS_TOKEN` | Access token for Shopify Admin API | `shpat_...` |
| `SHOPIFY_SHOP_DOMAIN` | Your Shopify store domain | `mystore.myshopify.com` |
| `DATABASE_URL` | SQLite database location | `sqlite:///data/shopify_agent.db` |
| `LOG_LEVEL` | Logging level | `INFO`, `DEBUG`, `WARNING`, `ERROR` |
| `LOG_FILE` | Log file path | `logs/agent.log` |
| `MCP_SERVER_COMMAND` | Command to run MCP server | `npx` |
| `MCP_SERVER_ARGS` | Arguments for MCP server | `-y,shopify-mcp-server` |

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize the bot and show welcome message |
| `/connect` | Connect or reconnect to your Shopify store |
| `/help` | Display available commands and usage examples |
| `/settings` | View and manage bot preferences |
| `/forget` | Clear learned patterns and preferences |
| `/status` | Check connection status and bot health |

## Example Queries

Once connected, you can ask the bot questions like:

- "What was my revenue last month?"
- "How many orders did I get this week compared to last week?"
- "Show me the top products by revenue"
- "What's my average order value for Q4?"
- "Compare my sales for January vs February"
- "What are my busiest days?"

The bot uses Claude AI to understand your queries and the learning system to refine responses based on your preferences.

## Project Structure

```
shopify-analytics-agent/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── README.md                        # This file
│
├── src/
│   ├── bot/                         # Telegram bot handlers
│   │   ├── commands.py              # Command implementations
│   │   ├── handlers.py              # Message handlers
│   │   └── keyboards.py             # UI components
│   │
│   ├── services/                    # Business logic services
│   │   ├── claude_service.py        # Claude AI integration
│   │   ├── shopify_service.py       # Shopify API wrapper
│   │   ├── mcp_service.py           # MCP server management
│   │   └── analytics_service.py     # Analytics calculations
│   │
│   ├── learning/                    # User learning system
│   │   ├── pattern_learner.py       # Pattern detection
│   │   ├── preference_manager.py    # User preferences
│   │   └── context_builder.py       # Query context building
│   │
│   ├── database/                    # Data persistence
│   │   ├── models.py                # SQLAlchemy models
│   │   ├── operations.py            # DB operations
│   │   └── init_db.py               # Database initialization
│   │
│   ├── config/                      # Configuration
│   │   └── settings.py              # Settings management
│   │
│   └── utils/                       # Utilities
│       ├── logger.py                # Logging setup
│       ├── formatters.py            # Output formatting
│       └── date_parser.py           # Date parsing utilities
│
└── tests/                           # Test suite
    ├── test_bot.py                  # Bot tests
    ├── test_services.py             # Service tests
    ├── test_learning.py             # Learning system tests
    └── conftest.py                  # Pytest configuration
```

## Testing

Run the test suite:

```bash
pytest tests/
```

For coverage report:

```bash
pytest tests/ --cov=src
```

## Tech Stack

- **Language:** Python 3.10+
- **Telegram:** `python-telegram-bot` - Telegram bot framework
- **AI:** `anthropic` - Claude AI integration
- **Database:** SQLAlchemy + SQLite - Data persistence
- **Async:** `aiohttp`, `asyncio` - Asynchronous operations
- **Data:** pandas, numpy - Data processing and analysis
- **Logging:** structlog - Structured logging
- **Testing:** pytest, pytest-asyncio - Testing framework

## Architecture Highlights

The bot follows a modular, service-oriented architecture:

1. **MCP Service** - Manages connection to Shopify via Model Context Protocol
2. **Claude Service** - Interfaces with Claude AI for natural language understanding
3. **Analytics Service** - Processes Shopify data for insights
4. **Learning System** - Tracks user patterns and preferences to improve responses
5. **Database Layer** - Persistent storage of queries, preferences, and learned patterns

## Development

To contribute or extend the bot:

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Ensure Python syntax is valid
5. Submit a pull request

## Troubleshooting

**Bot won't start:**
- Check all required env vars are set in `.env`
- Verify Telegram bot token is valid
- Check Anthropic API key is correct

**MCP connection fails:**
- Ensure `shopify-mcp-server` is installed: `npm install -g shopify-mcp-server`
- Check Shopify credentials are valid
- Review logs in `logs/agent.log`

**Database errors:**
- Delete `data/shopify_agent.db` and restart to reinitialize
- Ensure `data/` directory has write permissions

## License

This project is part of the Shopify Analytics Agent ecosystem.
