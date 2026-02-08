"""Abstract LLM service base class for multi-provider support.

Provides shared logic for system prompt building, message history,
tool execution, and error learning. Provider-specific API calls,
tool formatting, and response parsing are delegated to subclasses.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from src.config.settings import Settings
from src.database.operations import DatabaseOperations
from src.learning.context_builder import ContextBuilder
from src.learning.template_manager import TemplateManager
from src.learning.recovery_manager import RecoveryManager
from src.services.shopify_graphql import ShopifyGraphQLClient
from src.utils.logger import get_logger
from src.utils.timezone import now_ist

logger = get_logger(__name__)


@dataclass
class ToolCall:
    """Normalized tool call representation across providers."""
    id: str
    name: str
    input: Dict[str, Any]


class LLMService(ABC):
    """Abstract base class for LLM service providers.

    Subclasses must implement provider-specific methods for:
    - API client creation
    - Tool definition formatting
    - API calls
    - Response parsing
    - Message formatting for tool results
    """

    def __init__(
        self,
        settings: Settings,
        db_ops: DatabaseOperations,
        context_builder: Optional[ContextBuilder] = None,
        graphql_client: Optional[ShopifyGraphQLClient] = None,
        template_manager: Optional[TemplateManager] = None,
        recovery_manager: Optional[RecoveryManager] = None,
    ):
        self.settings = settings
        self.db_ops = db_ops
        self.context_builder = context_builder
        self.graphql_client = graphql_client
        self.template_manager = template_manager
        self.recovery_manager = recovery_manager

        # Track tool calls per message for learning
        self.last_tool_calls: List[Dict] = []

        # Initialize provider-specific client
        self.client = self._create_api_client()
        self.model = self._get_model_name()
        self.max_tokens = self._get_max_tokens()

        logger.info(
            "LLM service initialized",
            provider=self.provider_name,
            model=self.model,
        )

    # ─── Abstract methods (subclasses implement) ───────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'anthropic', 'openai')."""
        ...

    @abstractmethod
    def _create_api_client(self) -> Any:
        """Create and return the provider's API client."""
        ...

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name from settings."""
        ...

    @abstractmethod
    def _get_max_tokens(self) -> int:
        """Return max tokens from settings."""
        ...

    @abstractmethod
    def _format_tools_for_provider(self, base_tools: List[Dict]) -> List[Dict]:
        """Convert base tool schemas to provider-specific format.

        Args:
            base_tools: List of tool dicts with 'name', 'description', 'input_schema' keys

        Returns:
            Provider-formatted tool definitions
        """
        ...

    @abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict],
    ) -> Any:
        """Make the actual API call to the LLM provider.

        Args:
            system_prompt: System prompt string
            messages: Conversation messages
            tools: Provider-formatted tool definitions

        Returns:
            Raw provider response object
        """
        ...

    @abstractmethod
    def _parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        """Parse provider response into text and tool calls.

        Args:
            response: Raw provider response

        Returns:
            Tuple of (response_text, list_of_tool_calls)
        """
        ...

    @abstractmethod
    def _is_tool_use_response(self, response: Any) -> bool:
        """Check if the response indicates tool use is requested."""
        ...

    @abstractmethod
    def _build_assistant_message(self, response: Any) -> Dict:
        """Build assistant message dict for conversation history.

        Must include the full response content so tool results
        can reference the correct tool call IDs.
        """
        ...

    @abstractmethod
    def _build_tool_results_message(
        self,
        tool_results: List[Dict],
    ) -> Any:
        """Build tool results message(s) for conversation history.

        Returns:
            For Anthropic: single dict {"role": "user", "content": [...]}
            For OpenAI: list of dicts [{"role": "tool", ...}, ...]
        """
        ...

    # ─── Shared logic (provider-agnostic) ──────────────────────────

    async def process_message(
        self,
        user_id: int,
        message: str,
        intent=None,
        session_id: int = None,
    ) -> str:
        """Process a user message and return the LLM's response.

        Handles the full conversation flow including multi-turn tool use.
        Provider-specific details are delegated to abstract methods.

        Args:
            user_id: User ID for conversation context
            message: User's message/query
            intent: Optional intent classification for template learning
            session_id: Optional session ID for session-aware message loading

        Returns:
            Final response text
        """
        try:
            logger.info(
                "Processing message",
                user_id=user_id,
                message_length=len(message),
                session_id=session_id,
                provider=self.provider_name,
            )

            # Reset tool call tracking
            self.last_tool_calls = []
            session_key = f"{user_id}_{session_id or 'nosession'}"

            # Build conversation history and system prompt
            messages = self._build_messages(user_id, message, session_id)
            system_prompt = self._build_system_prompt(user_id)

            # Get tools in provider format
            base_tools = self._get_base_tool_definitions()
            tools = self._format_tools_for_provider(base_tools) if base_tools else []

            logger.info(
                "Tools available",
                tool_count=len(tools),
                tool_names=[t.get("name", t.get("function", {}).get("name", "?")) for t in base_tools],
            )

            # Multi-turn tool use loop
            response_text = ""
            tool_use_count = 0
            max_tool_iterations = 10

            while tool_use_count < max_tool_iterations:
                logger.debug(
                    "Calling LLM API",
                    provider=self.provider_name,
                    messages_count=len(messages),
                )

                # Call provider API
                response = self._call_api(system_prompt, messages, tools)

                # Parse response into text + tool calls
                text, tool_calls = self._parse_response(response)

                if text:
                    response_text = text

                # If no tool use requested, we're done
                if not self._is_tool_use_response(response) or not tool_calls:
                    logger.info(
                        "LLM response complete",
                        provider=self.provider_name,
                    )
                    break

                # Add assistant message to history
                messages.append(self._build_assistant_message(response))

                # Execute all tool calls
                tool_results = []
                for tc in tool_calls:
                    logger.info(
                        "Executing tool call",
                        tool_name=tc.name,
                        tool_input=tc.input,
                    )

                    tool_start = time.time()

                    try:
                        tool_result = await self._execute_tool(tc.name, tc.input)
                        tool_results.append({
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "content": tool_result,
                            "is_error": False,
                        })

                        # Track for learning
                        tool_call_info = {
                            "tool": tc.name,
                            "params": tc.input,
                            "success": True,
                            "time_ms": int((time.time() - tool_start) * 1000),
                        }
                        self.last_tool_calls.append(tool_call_info)

                        # Record success for template learning
                        if self.template_manager and intent:
                            try:
                                self.template_manager.record_successful_query(
                                    user_id=user_id,
                                    user_message=message,
                                    tool_name=tc.name,
                                    tool_params=tc.input,
                                    execution_time_ms=tool_call_info["time_ms"],
                                    intent_category=intent.fine if hasattr(intent, 'fine') else str(intent),
                                )
                            except Exception as tmpl_err:
                                logger.warning("Template recording failed", error=str(tmpl_err))

                        # Recovery recording
                        if self.recovery_manager:
                            try:
                                self.recovery_manager.record_recovery(
                                    session_key, tc.name, tc.input,
                                )
                            except Exception as rec_err:
                                logger.warning("Recovery recording failed", error=str(rec_err))

                    except Exception as tool_err:
                        logger.warning(
                            "Tool call failed",
                            tool_name=tc.name,
                            error=str(tool_err),
                        )
                        tool_results.append({
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "content": json.dumps({"error": str(tool_err)}),
                            "is_error": True,
                        })

                        self.last_tool_calls.append({
                            "tool": tc.name,
                            "params": tc.input,
                            "success": False,
                            "error": str(tool_err),
                        })

                        # Error recording for recovery learning
                        if self.recovery_manager:
                            try:
                                error_type = self._classify_error(str(tool_err), tc.name)
                                self.recovery_manager.record_error(
                                    session_key, tc.name, tc.input,
                                    error_type, str(tool_err),
                                )
                            except Exception as rec_err:
                                logger.warning("Error recording failed", error=str(rec_err))

                        # Template failure recording
                        if self.template_manager and intent:
                            try:
                                self.template_manager.record_failed_query(
                                    intent.fine if hasattr(intent, 'fine') else str(intent),
                                    tc.name,
                                )
                            except Exception as tmpl_err:
                                logger.warning("Template failure recording failed", error=str(tmpl_err))

                    tool_use_count += 1

                # Add tool results to message history
                results_msg = self._build_tool_results_message(tool_results)
                if isinstance(results_msg, list):
                    messages.extend(results_msg)
                else:
                    messages.append(results_msg)

            logger.info(
                "Message processed successfully",
                user_id=user_id,
                response_length=len(response_text),
                tool_use_count=tool_use_count,
                provider=self.provider_name,
            )

            return response_text

        except Exception as e:
            logger.error(
                "Failed to process message",
                user_id=user_id,
                provider=self.provider_name,
                error=str(e),
                exc_info=True,
            )
            raise

    @property
    def last_tool_calls_json(self) -> Optional[str]:
        """Get last tool calls as JSON string for storage."""
        if self.last_tool_calls:
            return json.dumps(self.last_tool_calls, default=str)
        return None

    def _get_base_tool_definitions(self) -> List[Dict]:
        """Return provider-neutral tool definitions.

        These use the Anthropic-style schema (name, description, input_schema)
        as the canonical format. Provider subclasses convert as needed.
        """
        tools = []
        if self.graphql_client:
            tools.append(self._get_analytics_tool_definition())
            tools.append(self._get_raw_graphql_tool_definition())
        return tools

    def _get_analytics_tool_definition(self) -> Dict[str, Any]:
        """Return the shopify_analytics tool definition."""
        return {
            "name": "shopify_analytics",
            "description": (
                "Query Shopify products, orders, or customers with sorting, filtering, and pagination. "
                "Use for: rankings (top products, biggest orders, best customers), "
                "date-filtered queries, sorting by any field (price, revenue, date), "
                "and paginating through large result sets. "
                "Queries Shopify's GraphQL Admin API directly. "
                "All timestamps in the response are already in IST (UTC+5:30) — display as-is, do NOT convert."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource": {
                        "type": "string",
                        "enum": ["products", "orders", "customers"],
                        "description": "The type of Shopify resource to query",
                    },
                    "sort_key": {
                        "type": "string",
                        "description": (
                            "Field to sort by. "
                            "Products: TITLE, PRICE, BEST_SELLING, CREATED_AT, UPDATED_AT, INVENTORY_TOTAL. "
                            "Orders: CREATED_AT, TOTAL_PRICE, ORDER_NUMBER, PROCESSED_AT. "
                            "Customers: NAME, TOTAL_SPENT, ORDERS_COUNT, CREATED_AT, LAST_ORDER_DATE."
                        ),
                    },
                    "reverse": {
                        "type": "boolean",
                        "description": "Sort descending (highest/newest first) when true. Default: true",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (1-250). Default: 10",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Shopify search/filter string. Examples: "
                            "'status:active', 'created_at:>2024-01-01', "
                            "'tag:sale', 'financial_status:paid'"
                        ),
                    },
                    "after": {
                        "type": "string",
                        "description": "Pagination cursor from previous response's end_cursor for next page",
                    },
                },
                "required": ["resource"],
            },
        }

    def _get_raw_graphql_tool_definition(self) -> Dict[str, Any]:
        """Return the shopify_graphql tool definition."""
        return {
            "name": "shopify_graphql",
            "description": (
                "Execute a custom GraphQL query against Shopify's Admin API. "
                "Use when shopify_analytics doesn't cover your needs — "
                "for example: shop info, inventory levels, discount codes, "
                "draft orders, metafields, collections, fulfillments, refunds, "
                "or any complex/nested query. You write the full GraphQL query. "
                "READ-ONLY: mutations are blocked for safety. "
                "The Shopify Admin API uses Relay-style connections (edges/node pattern). "
                "All timestamps in the response are already in IST (UTC+5:30) — display as-is, do NOT convert."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The full GraphQL query string. Must start with 'query' or '{'. "
                            "Use Shopify Admin API schema. Example: "
                            '\'{ shop { name email myshopifyDomain plan { displayName } } }\''
                        ),
                    },
                    "variables": {
                        "type": "object",
                        "description": "Optional GraphQL variables as a JSON object",
                    },
                },
                "required": ["query"],
            },
        }

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool call via direct Shopify GraphQL API."""
        start_time = time.time()

        try:
            if not self.graphql_client:
                raise ValueError("GraphQL client not initialized — check Shopify credentials")

            logger.info(
                "Executing tool via GraphQL",
                tool_name=tool_name,
                tool_input=tool_input,
            )

            if tool_name == "shopify_analytics":
                result_json = await self._execute_analytics_tool(tool_input)
            elif tool_name == "shopify_graphql":
                result_json = await self._execute_raw_graphql(tool_input)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Tool execution successful",
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )
            return result_json

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                error=str(e),
                execution_time_ms=execution_time_ms,
                exc_info=True,
            )

            # Log tool usage
            try:
                self.db_ops.log_tool_usage(
                    user_id=0,
                    tool_name=tool_name,
                    parameters=json.dumps(tool_input),
                    success=False,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as log_error:
                logger.warning("Failed to log tool usage", error=str(log_error))

            # Log error for learning
            try:
                error_type = self._classify_error(str(e), tool_name)
                lesson = self._generate_error_lesson(tool_name, tool_input, str(e), error_type)
                self.db_ops.log_query_error(
                    user_id=0,
                    tool_name=tool_name,
                    query_text=json.dumps(tool_input, default=str)[:2000],
                    error_message=str(e)[:1000],
                    error_type=error_type,
                    lesson=lesson,
                )
            except Exception as learn_error:
                logger.warning("Failed to log query error", error=str(learn_error))

            raise ValueError(f"Tool execution failed: {tool_name} - {str(e)}")

    async def _execute_analytics_tool(self, tool_input: Dict[str, Any]) -> str:
        """Execute the shopify_analytics tool."""
        resource = tool_input.get("resource", "products")
        sort_key = tool_input.get("sort_key", "CREATED_AT")
        reverse = tool_input.get("reverse", True)
        limit = tool_input.get("limit", 10)
        query = tool_input.get("query")
        after = tool_input.get("after")

        logger.info(
            "Executing analytics query",
            resource=resource, sort_key=sort_key,
            reverse=reverse, limit=limit, query=query,
        )

        if resource == "products":
            result = await self.graphql_client.query_products(
                sort_key=sort_key, reverse=reverse, limit=limit,
                query=query, after=after,
            )
        elif resource == "orders":
            result = await self.graphql_client.query_orders(
                sort_key=sort_key, reverse=reverse, limit=limit,
                query=query, after=after,
            )
        elif resource == "customers":
            result = await self.graphql_client.query_customers(
                sort_key=sort_key, reverse=reverse, limit=limit,
                query=query, after=after,
            )
        else:
            raise ValueError(f"Unknown resource: {resource}")

        return json.dumps(result)

    async def _execute_raw_graphql(self, tool_input: Dict[str, Any]) -> str:
        """Execute a raw GraphQL query."""
        query = tool_input.get("query", "")
        variables = tool_input.get("variables")
        logger.info("Executing raw GraphQL", query_length=len(query))
        result = await self.graphql_client.execute_raw_query(query, variables)
        return json.dumps(result)

    def _build_system_prompt(self, user_id: int) -> str:
        """Build the system prompt with context, tool guidance, and formatting rules."""
        try:
            system_parts = [
                "You are a Shopify Analytics Assistant. Your role is to help merchants analyze their store data, understand sales trends, and make data-driven decisions.",
                "",
                f"Current date and time (IST, Indian Standard Time, UTC+5:30): {now_ist().isoformat()}",
                f"Store domain: {self.settings.shopify.shop_domain}",
                "",
                "IMPORTANT — TIMEZONE HANDLING:",
                "- All timestamps in tool results are ALREADY converted to IST (Indian Standard Time, UTC+5:30).",
                "- Do NOT convert or adjust any timestamps — display them exactly as received.",
                "- When showing a timestamp like '2026-02-09T00:15:00+05:30', display it as 'Feb 9, 2026 at 12:15 AM IST'.",
                "- Always append 'IST' when showing times to the user.",
            ]

            # Context from context builder
            if self.context_builder:
                try:
                    context = self.context_builder.build_context(user_id)

                    if context.get("preferences"):
                        pref_str = ", ".join(
                            f"{k}: {v}" for k, v in context["preferences"].items()
                        )
                        system_parts.append(f"\nUser preferences: {pref_str}")

                    if context.get("recent_patterns"):
                        patterns_str = ", ".join(
                            f"{p['value']} (frequency: {p['frequency']})"
                            for p in context["recent_patterns"]
                        )
                        system_parts.append(f"\nRecent query patterns: {patterns_str}")

                    if context.get("past_errors"):
                        system_parts.append("")
                        system_parts.append(
                            "PAST QUERY ERRORS — Learn from these and DO NOT repeat the same mistakes:"
                        )
                        for i, err in enumerate(context["past_errors"], 1):
                            tool = err.get("tool_name", "unknown")
                            error_type = err.get("error_type", "unknown")
                            error_msg = err.get("error_message", "")[:200]
                            lesson = err.get("lesson", "")
                            query_snippet = err.get("query_text", "")[:150]
                            system_parts.append(f"  {i}. [{tool}] {error_type}: {error_msg}")
                            if query_snippet:
                                system_parts.append(f"     Failed query: {query_snippet}")
                            if lesson:
                                system_parts.append(f"     Lesson: {lesson}")
                        system_parts.append(
                            "When constructing queries, review the above errors and "
                            "ensure you use correct field names, valid sort keys, "
                            "and proper GraphQL syntax to avoid these same issues."
                        )

                except Exception as e:
                    logger.warning("Failed to build context", error=str(e))

            # Tool guidance
            if self.graphql_client:
                system_parts.extend([
                    "",
                    "AVAILABLE TOOLS:",
                    "",
                    "1. shopify_analytics — Structured analytics queries with sorting, filtering, and pagination",
                    "   - Query products, orders, or customers",
                    "   - Sort products by: TITLE, PRICE, BEST_SELLING, CREATED_AT, INVENTORY_TOTAL",
                    "   - Sort orders by: CREATED_AT, TOTAL_PRICE, ORDER_NUMBER",
                    "   - Sort customers by: NAME, TOTAL_SPENT, ORDERS_COUNT, LAST_ORDER_DATE",
                    "   - Filter with Shopify queries: 'status:active', 'created_at:>2024-01-01', 'financial_status:paid'",
                    "   - Supports cursor-based pagination for large datasets",
                    "",
                    "2. shopify_graphql — Custom GraphQL queries against Shopify Admin API",
                    "   - Use for ANYTHING not covered by shopify_analytics",
                    "   - Shop info, inventory levels, collections, discounts, metafields, fulfillments, etc.",
                    "   - You write the full GraphQL query using Shopify Admin API schema",
                    "   - Uses Relay-style connections: edges { node { ... } } with pageInfo { hasNextPage endCursor }",
                    "   - Key types: Product, Order, Customer, Collection, InventoryItem, DiscountCode, DraftOrder, Fulfillment",
                    "   - Money fields use MoneyV2: { amount currencyCode }",
                    "   - Shop info: { shop { name email myshopifyDomain plan { displayName } currencyCode } }",
                    "   - READ-ONLY: mutations are blocked for safety",
                    "",
                    "TOOL SELECTION STRATEGY:",
                    "- Use shopify_analytics for: rankings, sorting, comparisons, date-filtered queries on products/orders/customers",
                    "- Use shopify_graphql for: shop info, inventory, collections, discounts, metafields, complex nested queries, anything else",
                ])

            system_parts.extend([
                "",
                "ANALYSIS GUIDELINES:",
                "- Always include specific numbers: revenue, order count, AOV, units sold",
                "- Explain what the numbers mean in plain language",
                "- Highlight notable trends, outliers, or changes worth attention",
                "- End with 1-2 short actionable recommendations when useful",
                "- Be concise — no filler, every sentence should add value",
                "",
                "RESPONSE FORMATTING (CRITICAL — output is rendered in Telegram):",
                "",
                "Structure rules:",
                "- Start with a bold title line with emoji: **📊 Revenue Summary** or **🏆 Top Products**",
                "- Group data into clear sections separated by a blank line",
                "- Use **bold** for section labels and key terms",
                "- Use `code` for specific values: amounts, IDs, dates, percentages",
                "- Keep paragraphs to 1-2 sentences max",
                "",
                "Lists and rankings:",
                "- Use numbered lists (1. 2. 3.) for rankings and ordered items",
                "- Use bullet points (- item) for unordered info",
                "- Add a sub-detail line indented under each item with key metrics",
                "",
                "Numbers and formatting:",
                "- Currency: $1,234.56 (with commas and 2 decimals)",
                "- Percentages: 15.5% or +12.3% / -8.1% for changes",
                "- Large numbers: 1,234 (always use comma separators)",
                "- Comparisons: show change with arrow: ▲ 12.3% or ▼ 8.1%",
                "",
                "Do NOT use:",
                "- Markdown headings (#, ##, ###) — use **bold** instead",
                "- Horizontal rules (---) — use blank lines to separate",
                "- Markdown tables — convert to numbered or bulleted lists",
                "- Nested markdown (**bold *italic***) — keep it simple",
                "- Long walls of text — break into scannable sections",
                "",
                "Example of ideal response format:",
                "",
                "**📊 Revenue Report — Last 7 Days**",
                "",
                "**Overview**",
                "- Total Revenue: `$12,345.67`",
                "- Orders: `156` (▲ 12.3% vs prior week)",
                "- Avg Order Value: `$79.14`",
                "",
                "**🏆 Top Products**",
                "1. **Premium Widget** — `$3,456.00` (42 units)",
                "2. **Basic Bundle** — `$2,100.50` (87 units)",
                "3. **Gift Set** — `$1,890.00` (31 units)",
                "",
                "**💡 Insight**",
                "Revenue is up 12% week-over-week driven by the Premium Widget. Consider featuring it on your homepage.",
            ])

            return "\n".join(system_parts)

        except Exception as e:
            logger.error("Failed to build system prompt", user_id=user_id, error=str(e))
            return "You are a Shopify Analytics Assistant. Help the merchant analyze their store data."

    def _build_messages(
        self, user_id: int, message: str, session_id: int = None
    ) -> List[Dict[str, Any]]:
        """Build messages array from conversation history and current message."""
        try:
            messages = []

            if session_id:
                try:
                    recent_conversations = self.db_ops.get_session_conversations(session_id)
                except Exception:
                    recent_conversations = self.db_ops.get_recent_conversations(
                        user_id, limit=self.settings.conversation_history_limit,
                    )
            else:
                recent_conversations = self.db_ops.get_recent_conversations(
                    user_id, limit=self.settings.conversation_history_limit,
                )

            # Previous session context
            if session_id:
                try:
                    prev = self.db_ops.get_previous_session(user_id, session_id)
                    if prev and prev.session_summary:
                        messages.append({"role": "user", "content": f"[Previous session context: {prev.session_summary}]"})
                        messages.append({"role": "assistant", "content": "Understood, I have context from our previous conversation."})
                except Exception:
                    pass

            for conv in reversed(recent_conversations):
                messages.append({"role": "user", "content": conv.message})
                messages.append({"role": "assistant", "content": conv.response})

            messages.append({"role": "user", "content": message})

            logger.debug(
                "Messages built",
                user_id=user_id,
                session_id=session_id,
                total_messages=len(messages),
            )
            return messages

        except Exception as e:
            logger.warning("Failed to build messages", user_id=user_id, error=str(e))
            return [{"role": "user", "content": message}]

    @staticmethod
    def _classify_error(error_message: str, tool_name: str) -> str:
        """Classify an error into a category for learning."""
        error_lower = error_message.lower()

        if any(kw in error_lower for kw in ["parse error", "syntax error", "unexpected token"]):
            return "graphql_syntax"
        if any(kw in error_lower for kw in ["field", "does not exist", "unknown field", "undefined field"]):
            return "invalid_field"
        if any(kw in error_lower for kw in ["argument", "required", "missing"]):
            return "missing_argument"
        if any(kw in error_lower for kw in ["type mismatch", "wrong type", "expected type", "enum"]):
            return "type_mismatch"
        if "mutation" in error_lower:
            return "mutation_blocked"
        if any(kw in error_lower for kw in ["timeout", "timed out"]):
            return "timeout"
        if any(kw in error_lower for kw in ["401", "unauthorized", "authentication"]):
            return "auth_error"
        if any(kw in error_lower for kw in ["429", "throttle", "rate limit"]):
            return "rate_limit"
        if any(kw in error_lower for kw in ["500", "internal server"]):
            return "server_error"
        if any(kw in error_lower for kw in ["404", "not found"]):
            return "not_found"

        return "unknown"

    @staticmethod
    def _generate_error_lesson(
        tool_name: str,
        tool_input: Dict[str, Any],
        error_message: str,
        error_type: str,
    ) -> str:
        """Generate a concise lesson from a failed tool call."""
        lessons = {
            "graphql_syntax": (
                f"GraphQL syntax error in {tool_name}. "
                "Double-check query structure, braces, and field names. "
                "Ensure the query starts with '{{' or 'query' keyword."
            ),
            "invalid_field": (
                f"Used an invalid field name in {tool_name}. "
                "Check the Shopify Admin API schema for correct field names. "
                "Common gotchas: 'totalPrice' not 'total_price', "
                "'createdAt' not 'created_at' in GraphQL."
            ),
            "missing_argument": (
                f"Missing required argument in {tool_name}. "
                "Check the tool's input schema for required parameters."
            ),
            "type_mismatch": (
                f"Type mismatch in {tool_name}. "
                "Verify enum values and argument types match the schema. "
                "Common: sort keys must be exact enum values like 'CREATED_AT' not 'created_at'."
            ),
            "mutation_blocked": (
                "Attempted a mutation via shopify_graphql which is read-only. "
                "This bot is analytics-only — write operations are not supported."
            ),
            "timeout": (
                f"Query to {tool_name} timed out. "
                "Try reducing the result limit or simplifying the query."
            ),
            "rate_limit": (
                "Shopify API rate limit hit. Wait a moment before retrying, "
                "or reduce the number of sequential API calls."
            ),
            "auth_error": (
                "Authentication failed. Check that the Shopify access token is valid "
                "and has the required scopes."
            ),
        }

        lesson = lessons.get(error_type, f"Tool '{tool_name}' failed with error type '{error_type}'.")

        if tool_name == "shopify_graphql" and "query" in tool_input:
            query_snippet = str(tool_input["query"])[:150]
            lesson += f" Failed query snippet: {query_snippet}"

        return lesson
