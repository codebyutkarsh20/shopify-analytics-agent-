# Shopify Analytics Agent

A Python-based Shopify analytics agent with a Telegram bot interface, powered by multi-LLM support (Claude AI or OpenAI GPT) and direct GraphQL API access. Provides natural language analytics queries, learns from every interaction, and delivers increasingly relevant insights about your Shopify store performance.

## Features

- **Multi-LLM Support** - Choose between Anthropic Claude or OpenAI GPT via environment variable
- **Natural Language Queries** - Ask questions about your store in plain English
- **Revenue & Order Analysis** - Analyze sales trends, revenue metrics, and order patterns
- **Product & Customer Intelligence** - Track top products, customer behavior, and inventory
- **Time-Based Comparisons** - Compare metrics across different time periods
- **Direct GraphQL API** - Connects to Shopify's Admin GraphQL API directly (no MCP dependency)
- **5-Layer Learning System** - Session memory, user memory, query knowledge, tool intelligence, and response quality tracking
- **Session Management** - Automatic session detection for single-thread chat platforms
- **Error Recovery Learning** - Learns from failures and applies proven recovery strategies
- **Channel-Agnostic Architecture** - Built for Telegram with WhatsApp support ready

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

The bot will initialize the database, seed query templates, connect to your services, and start polling for messages.

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use (`anthropic` or `openai`) | `anthropic` (default) |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from BotFather | `123456789:ABCdefGHIjklmno` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required if provider=anthropic) | `sk-ant-...` |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-sonnet-4-5-20250929` (default) |
| `OPENAI_API_KEY` | Your OpenAI API key (required if provider=openai) | `sk-...` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` (default) |
| `SHOPIFY_ACCESS_TOKEN` | Access token for Shopify Admin API | `shpat_...` |
| `SHOPIFY_SHOP_DOMAIN` | Your Shopify store domain | `mystore.myshopify.com` |
| `DATABASE_URL` | SQLite database location | `sqlite:///data/shopify_agent.db` |
| `LOG_LEVEL` | Logging level | `INFO`, `DEBUG`, `WARNING`, `ERROR` |
| `LOG_FILE` | Log file path | `logs/agent.log` |

### Switching LLM Providers

By default the bot uses Anthropic Claude. To switch to OpenAI:

```bash
# In your .env file:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini        # or gpt-4o, gpt-4-turbo, etc.
```

Both providers support the full tool-calling pipeline (Shopify GraphQL queries, multi-turn conversations, error recovery learning). The learning system, session management, and all other features work identically regardless of provider.

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
- "Show me the top 10 products by revenue"
- "What's my average order value for Q4?"
- "Compare my sales for January vs February"
- "Who are my top spending customers?"
- "What products are low on inventory?"

The bot uses Claude AI to understand your queries and the learning system refines responses based on your interaction patterns.

## Architecture

### System Overview

The bot follows a modular, service-oriented architecture with a 5-layer learning system:

```
User Message (Telegram)
    |
    v
TelegramAdapter (channel abstraction)
    |
    v
MessageHandler (10-step pipeline)
    |
    +---> PatternLearner.classify_intent()     [coarse + fine intent]
    +---> SessionManager.get_or_create_session() [session boundary detection]
    +---> FeedbackAnalyzer.analyze_follow_up()  [implicit quality feedback]
    +---> ClaudeService.process_message()       [Claude AI + GraphQL tools]
    |         |
    |         +---> TemplateManager             [reuse proven queries]
    |         +---> RecoveryManager              [auto-recover from errors]
    |         +---> ContextBuilder               [rich learning context]
    |
    +---> PatternLearner.learn_from_query()     [store patterns]
    +---> PreferenceManager                     [update preferences]
    +---> InsightAggregator                     [periodic cross-user learning]
    |
    v
Response (formatted via TelegramAdapter)
```

### Message Processing Pipeline

Every incoming message goes through 10 steps:

1. **Resolve user** - Get or create user from Telegram metadata
2. **Classify intent** - Two-level classification (coarse: `revenue`, fine: `products_ranking_revenue_last_30_days`)
3. **Detect/create session** - Multi-factor session boundary detection (time gaps + topic shifts + explicit signals)
4. **Analyze feedback** - Detect implicit satisfaction signals from follow-up messages
5. **Show typing indicator** - UX feedback while processing
6. **Process through Claude** - Session-aware conversation with GraphQL tool execution
7. **Learn patterns** - Store query patterns, update template/recovery knowledge
8. **Format & send** - Channel-specific formatting (Telegram HTML) with message splitting
9. **Save conversation** - Persist with session_id, channel_type, and tool call history
10. **Aggregation check** - Periodic cross-user insight aggregation

### Learning System (5 Layers)

**Layer 1: Session Memory** - Tracks conversation sessions on single-thread platforms using time gaps (hard: 2hr, soft: 30min + topic change), topic shift detection, and explicit user signals ("new question", "different topic").

**Layer 2: User Memory** - Per-user patterns, preferences, interaction history, and preferred metrics/time ranges.

**Layer 3: Query Knowledge (Global)** - Successful GraphQL query templates indexed by intent. When a query succeeds, its tool+parameters are stored as a reusable template. Future similar intents get pre-built proven queries instead of constructing from scratch.

**Layer 4: Tool Intelligence (Global)** - Error recovery patterns with MD5 fingerprinting. When an error occurs and is later recovered, the error-to-recovery mapping is stored. Future identical errors can be auto-recovered.

**Layer 5: Response Quality** - Implicit feedback detection from user follow-ups: "thanks" = positive (score: +0.8), "wrong" = negative (score: -0.8), corrections = feedback (-0.5), refinements = partial success (+0.5). Quality scores feed back into template confidence.

### Database Schema

14 tables managed via SQLAlchemy 2.0 ORM:

| Table | Purpose |
|-------|---------|
| `users` | User accounts with interaction counts |
| `conversations` | Message history with session/channel/quality tracking |
| `sessions` | Conversation sessions with lifecycle management |
| `query_templates` | Proven query patterns indexed by intent |
| `error_recovery_patterns` | Error fingerprint to recovery action mappings |
| `global_insights` | Cross-user aggregated intelligence |
| `response_feedback` | Implicit quality feedback records |
| `channel_sessions` | Channel-specific session metadata |
| `tool_usage` | Tool execution history with intent and template tracking |
| `query_errors` | Error log with recovery linkage |
| `query_patterns` | User query pattern frequencies |
| `user_preferences` | Manual and learned user preferences |
| `shopify_stores` | Connected Shopify store credentials |
| `analytics_cache` | Cached analytics results |

## Project Structure

```
shopify-analytics-agent/
├── main.py                              # Entry point - wires all components
├── requirements.txt                     # Python dependencies
├── .env.example                         # Environment variables template
├── README.md                            # This file
├── MEMORY_SYSTEM_DESIGN.md              # Detailed learning system design doc
│
├── src/
│   ├── bot/                             # Telegram bot layer
│   │   ├── handlers.py                  # 10-step message pipeline
│   │   ├── commands.py                  # /start, /connect, /help, etc.
│   │   ├── keyboards.py                 # Inline keyboard UI components
│   │   ├── channel_adapter.py           # Abstract channel interface
│   │   └── telegram_adapter.py          # Telegram-specific formatting
│   │
│   ├── services/                        # Core services
│   │   ├── llm_service.py              # Abstract LLM base (shared logic)
│   │   ├── anthropic_service.py        # Anthropic Claude implementation
│   │   ├── openai_service.py           # OpenAI GPT implementation
│   │   ├── llm_factory.py             # Provider factory (env-driven)
│   │   ├── claude_service.py           # Backward-compat alias
│   │   └── shopify_graphql.py          # Direct Shopify GraphQL client
│   │
│   ├── learning/                        # 5-layer learning system
│   │   ├── pattern_learner.py           # Intent classification + pattern detection
│   │   ├── context_builder.py           # Rich context assembly (templates, recovery, insights)
│   │   ├── preference_manager.py        # User preference management
│   │   ├── session_manager.py           # Session lifecycle management
│   │   ├── template_manager.py          # Query template learning
│   │   ├── recovery_manager.py          # Error recovery pattern learning
│   │   ├── feedback_analyzer.py         # Implicit feedback detection
│   │   ├── insight_aggregator.py        # Cross-user intelligence aggregation
│   │   └── template_seeds.py            # 10 seed templates for common queries
│   │
│   ├── database/                        # Data persistence
│   │   ├── models.py                    # 14 SQLAlchemy ORM models
│   │   ├── operations.py               # ~40 database operations
│   │   └── init_db.py                   # DB init, migration, and seeding
│   │
│   ├── config/                          # Configuration
│   │   └── settings.py                  # Settings with session/learning params
│   │
│   └── utils/                           # Utilities
│       ├── logger.py                    # Structured logging (structlog)
│       ├── formatters.py                # Output formatting + Markdown-to-HTML
│       └── date_parser.py              # Natural language date parsing
│
└── tests/                               # Test suite
    ├── test_bot.py                      # Date parser, formatter tests
    ├── test_learning.py                 # PatternLearner, ContextBuilder, PreferenceManager tests
    └── conftest.py                      # Pytest configuration
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

For coverage report:

```bash
pytest tests/ --cov=src
```

## Tech Stack

- **Language:** Python 3.10+
- **Telegram:** `python-telegram-bot` 20.7 - Async Telegram bot framework
- **AI:** `anthropic` - Claude AI with tool use for natural language understanding
- **Shopify:** Direct GraphQL API via `httpx` - No MCP dependency
- **Database:** SQLAlchemy 2.0 + SQLite (WAL mode) - ORM with 14 tables
- **Async:** `aiohttp`, `asyncio`, `httpx` - Async HTTP and event loop
- **Logging:** `structlog` - Structured JSON logging
- **Testing:** `pytest`, `pytest-asyncio` - Async-compatible test framework

## Extending to WhatsApp

The channel-agnostic architecture makes adding WhatsApp straightforward:

1. Create `src/bot/whatsapp_adapter.py` extending `ChannelAdapter`
2. Implement channel-specific methods (user resolution, formatting, limits)
3. Add a WhatsApp webhook handler
4. Wire into `main.py` alongside the Telegram setup

The `SessionManager`, `TemplateManager`, `RecoveryManager`, and all learning components work across channels without modification.

## Troubleshooting

**Bot won't start:**
- Check all required env vars are set in `.env`
- Verify Telegram bot token is valid with BotFather
- Check Anthropic API key is correct

**Shopify queries fail:**
- Verify `SHOPIFY_ACCESS_TOKEN` has the required API scopes (read_products, read_orders, read_customers)
- Check `SHOPIFY_SHOP_DOMAIN` format (e.g., `mystore.myshopify.com`)
- Review logs in `logs/agent.log` for GraphQL error details

**Database errors:**
- Delete `data/shopify_agent.db` and restart to reinitialize
- Ensure `data/` directory has write permissions
- The migration system handles `mcp_tool_usage` to `tool_usage` rename automatically

**Empty or poor responses:**
- The learning system improves over time — initial queries may be less accurate
- Check that query templates are seeded (happens automatically on first run)
- Try explicit queries like "show me revenue for last 7 days" for best results

## License

This project is part of the Shopify Analytics Agent ecosystem.
