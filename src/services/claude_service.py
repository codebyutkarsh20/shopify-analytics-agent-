"""Claude API integration service for processing Shopify analytics queries."""

import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import anthropic

from src.config.settings import Settings
from src.database.operations import DatabaseOperations
from src.learning.context_builder import ContextBuilder
from src.learning.template_manager import TemplateManager
from src.learning.recovery_manager import RecoveryManager
from src.services.shopify_graphql import ShopifyGraphQLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeService:
    """Service for integrating Claude API with Shopify analytics."""

    def __init__(
        self,
        settings: Settings,
        db_ops: DatabaseOperations,
        context_builder: Optional[ContextBuilder] = None,
        graphql_client: Optional[ShopifyGraphQLClient] = None,
        template_manager: Optional['TemplateManager'] = None,
        recovery_manager: Optional['RecoveryManager'] = None,
    ):
        """
        Initialize the Claude service.

        Args:
            settings: Settings object with anthropic configuration
            db_ops: DatabaseOperations instance for database access
            context_builder: ContextBuilder instance for user context (optional)
            graphql_client: ShopifyGraphQLClient for direct Shopify API queries (optional)
            template_manager: TemplateManager for learning from successful queries (optional)
            recovery_manager: RecoveryManager for learning from errors and recovery (optional)
        """
        self.settings = settings
        self.db_ops = db_ops
        self.context_builder = context_builder
        self.graphql_client = graphql_client
        self.template_manager = template_manager
        self.recovery_manager = recovery_manager

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=settings.anthropic.api_key)
        self.model = settings.anthropic.model
        self.max_tokens = settings.anthropic.max_tokens

        # Track tool calls per message for learning
        self.last_tool_calls = []

    async def process_message(self, user_id: int, message: str, intent=None, session_id: int = None) -> str:
        """
        Process a user message and return Claude's response.

        This method handles the full conversation flow including:
        1. Building conversation history
        2. Creating system prompt with context
        3. Calling Claude API
        4. Handling tool calls via MCP
        5. Managing multi-turn tool use
        6. Saving conversation to database
        7. Returning final response

        Args:
            user_id: User ID for conversation context
            message: User's message/query
            intent: Optional intent classification for template learning
            session_id: Optional session ID for session-aware message loading

        Returns:
            Final response text from Claude

        Raises:
            ValueError: If Claude API returns an error
            RuntimeError: If MCP tool execution fails
        """
        try:
            logger.info(
                "Processing message",
                user_id=user_id,
                message_length=len(message),
                session_id=session_id,
            )

            # Reset tool call tracking for this message
            self.last_tool_calls = []

            # Create session key for recovery tracking
            session_key = f"{user_id}_{session_id or 'nosession'}"

            # Build conversation history
            messages = self._build_messages(user_id, message, session_id)

            # Build system prompt with context
            system_prompt = self._build_system_prompt(user_id)

            # Get available tools (direct GraphQL — no MCP needed)
            tools = []
            if self.graphql_client:
                tools.append(self._get_analytics_tool_definition())
                tools.append(self._get_raw_graphql_tool_definition())

            logger.info("Tools available for Claude", tool_count=len(tools),
                        tool_names=[t["name"] for t in tools])

            # Call Claude API in a loop to handle multi-turn tool use
            response_text = ""
            tool_use_count = 0
            max_tool_iterations = 10

            while tool_use_count < max_tool_iterations:
                logger.debug(
                    "Calling Claude API",
                    messages_count=len(messages),
                    tool_count=len(tools),
                )

                # Call Claude with tools
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )

                logger.debug(
                    "Claude response received",
                    stop_reason=response.stop_reason,
                    content_blocks=len(response.content),
                )

                # Process response content
                tool_calls = []
                text_parts = []
                for content_block in response.content:
                    if content_block.type == "text":
                        text_parts.append(content_block.text)
                    elif content_block.type == "tool_use":
                        tool_calls.append(content_block)

                # Accumulate all text blocks (don't overwrite)
                if text_parts:
                    response_text = "\n".join(text_parts)

                # If Claude didn't use a tool, we're done
                if response.stop_reason != "tool_use" or not tool_calls:
                    logger.info(
                        "Claude response complete",
                        stop_reason=response.stop_reason,
                    )
                    break

                # Add Claude's full response to message history ONCE (before executing tools)
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )

                # Execute all tool calls and collect results
                tool_results = []
                for tool_call in tool_calls:
                    logger.info(
                        "Executing tool call",
                        tool_name=tool_call.name,
                        tool_input=tool_call.input,
                    )

                    tool_start = time.time()

                    # Execute the tool — catch errors so Claude can handle them
                    try:
                        tool_result = await self._execute_tool(
                            tool_call.name,
                            tool_call.input,
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": tool_result,
                            }
                        )

                        # Track tool call for learning
                        tool_call_info = {
                            "tool": tool_call.name,
                            "params": tool_call.input,
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
                                    tool_name=tool_call.name,
                                    tool_params=tool_call.input,
                                    execution_time_ms=tool_call_info["time_ms"],
                                    intent_category=intent.fine if hasattr(intent, 'fine') else str(intent),
                                )
                            except Exception as tmpl_err:
                                logger.warning("Template recording failed", error=str(tmpl_err))

                        # If there was a pending error, this is a recovery
                        if self.recovery_manager:
                            try:
                                self.recovery_manager.record_recovery(
                                    session_key, tool_call.name, tool_call.input,
                                )
                            except Exception as rec_err:
                                logger.warning("Recovery recording failed", error=str(rec_err))

                    except Exception as tool_err:
                        logger.warning(
                            "Tool call failed, sending error to Claude",
                            tool_name=tool_call.name,
                            error=str(tool_err),
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": json.dumps({"error": str(tool_err)}),
                                "is_error": True,
                            }
                        )

                        # Track failed tool call
                        tool_call_info = {
                            "tool": tool_call.name,
                            "params": tool_call.input,
                            "success": False,
                            "error": str(tool_err),
                        }
                        self.last_tool_calls.append(tool_call_info)

                        # Record error for recovery learning
                        if self.recovery_manager:
                            try:
                                error_type = self._classify_error(str(tool_err), tool_call.name)
                                self.recovery_manager.record_error(
                                    session_key, tool_call.name, tool_call.input,
                                    error_type, str(tool_err),
                                )
                            except Exception as rec_err:
                                logger.warning("Error recording for recovery failed", error=str(rec_err))

                        # Record failure for template learning
                        if self.template_manager and intent:
                            try:
                                self.template_manager.record_failed_query(
                                    intent.fine if hasattr(intent, 'fine') else str(intent),
                                    tool_call.name,
                                )
                            except Exception as tmpl_err:
                                logger.warning("Template failure recording failed", error=str(tmpl_err))

                    tool_use_count += 1

                # Add all tool results in a single user message
                messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )

            # Note: conversation is saved by handlers.py with proper query_type detection

            logger.info(
                "Message processed successfully",
                user_id=user_id,
                response_length=len(response_text),
                tool_use_count=tool_use_count,
            )

            return response_text

        except Exception as e:
            logger.error(
                "Failed to process message",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def _build_system_prompt(self, user_id: int) -> str:
        """
        Build the system prompt for Claude.

        Includes role description, context, and tool instructions.

        Args:
            user_id: User ID for building personalized context

        Returns:
            System prompt string
        """
        try:
            # Base system prompt
            system_parts = [
                "You are a Shopify Analytics Assistant. Your role is to help merchants analyze their store data, understand sales trends, and make data-driven decisions.",
                "",
                f"Current date and time: {datetime.now().isoformat()}",
                f"Store domain: {self.settings.shopify.shop_domain}",
            ]

            # Add context from context builder if available
            if self.context_builder:
                try:
                    context = self.context_builder.build_context(user_id)

                    if context.get("preferences"):
                        pref_str = ", ".join(
                            f"{k}: {v}"
                            for k, v in context["preferences"].items()
                        )
                        system_parts.append(f"\nUser preferences: {pref_str}")

                    if context.get("recent_patterns"):
                        patterns_str = ", ".join(
                            f"{p['value']} (frequency: {p['frequency']})"
                            for p in context["recent_patterns"]
                        )
                        system_parts.append(
                            f"\nRecent query patterns: {patterns_str}"
                        )

                    # Include past errors for learning
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
                    logger.warning(
                        "Failed to build context from context_builder",
                        error=str(e),
                    )

            # Add tool guidance
            if self.graphql_client:
                system_parts.extend(
                    [
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
                    ]
                )

            system_parts.extend(
                [
                    "",
                    "Guidelines for analysis:",
                    "- Always provide specific metrics when available (revenue, order count, AOV)",
                    "- Explain trends and patterns in the data",
                    "- Suggest actionable insights based on the data",
                    "- Use appropriate date ranges for comparisons",
                    "- Be concise but comprehensive in your analysis",
                    "",
                    "RESPONSE FORMATTING (VERY IMPORTANT — this is for Telegram):",
                    "- Use **bold** for section titles and key terms",
                    "- Use bullet points with - for lists",
                    "- Use numbered lists (1. 2. 3.) for rankings and ordered items",
                    "- Use `code` for IDs, amounts, or technical values",
                    "- Keep paragraphs short (2-3 sentences max)",
                    "- Separate sections with a blank line",
                    "- Do NOT use markdown headings (#, ##, ###) — use **bold** instead",
                    "- Do NOT use horizontal rules (---)",
                    "- Do NOT use tables — use clean bullet points or numbered lists instead",
                    "- Do NOT use nested markdown (like **bold *italic***) — keep it simple",
                    "- For currency, write values plainly: $1,234.56",
                    "- Use emojis sparingly for visual clarity: 📊 📈 💰 🏆 📦",
                ]
            )

            return "\n".join(system_parts)

        except Exception as e:
            logger.error(
                "Failed to build system prompt",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            # Return a minimal fallback prompt
            return "You are a Shopify Analytics Assistant. Help the merchant analyze their store data."

    @property
    def last_tool_calls_json(self) -> Optional[str]:
        """Get last tool calls as JSON string for storage."""
        if self.last_tool_calls:
            return json.dumps(self.last_tool_calls, default=str)
        return None

    def _get_analytics_tool_definition(self) -> Dict[str, Any]:
        """
        Return the shopify_analytics tool definition for Claude.

        Structured analytics queries with sorting, filtering, and pagination
        against Shopify's GraphQL Admin API.
        """
        return {
            "name": "shopify_analytics",
            "description": (
                "Query Shopify products, orders, or customers with sorting, filtering, and pagination. "
                "Use for: rankings (top products, biggest orders, best customers), "
                "date-filtered queries, sorting by any field (price, revenue, date), "
                "and paginating through large result sets. "
                "Queries Shopify's GraphQL Admin API directly."
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

    async def _execute_analytics_tool(self, tool_input: Dict[str, Any]) -> str:
        """
        Execute the shopify_analytics tool via direct GraphQL.

        Args:
            tool_input: Parameters from Claude's tool call

        Returns:
            JSON string with query results
        """
        resource = tool_input.get("resource", "products")
        sort_key = tool_input.get("sort_key", "CREATED_AT")
        reverse = tool_input.get("reverse", True)
        limit = tool_input.get("limit", 10)
        query = tool_input.get("query")
        after = tool_input.get("after")

        logger.info(
            "Executing analytics query",
            resource=resource,
            sort_key=sort_key,
            reverse=reverse,
            limit=limit,
            query=query,
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

    def _get_raw_graphql_tool_definition(self) -> Dict[str, Any]:
        """
        Return the shopify_graphql tool definition for Claude.

        This gives Claude full power to write any GraphQL query against
        Shopify's Admin API. Mutations are blocked for safety.
        """
        return {
            "name": "shopify_graphql",
            "description": (
                "Execute a custom GraphQL query against Shopify's Admin API. "
                "Use when shopify_analytics doesn't cover your needs — "
                "for example: shop info, inventory levels, discount codes, "
                "draft orders, metafields, collections, fulfillments, refunds, "
                "or any complex/nested query. You write the full GraphQL query. "
                "READ-ONLY: mutations are blocked for safety. "
                "The Shopify Admin API uses Relay-style connections (edges/node pattern)."
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

    async def _execute_raw_graphql(self, tool_input: Dict[str, Any]) -> str:
        """Execute a Claude-written raw GraphQL query."""
        query = tool_input.get("query", "")
        variables = tool_input.get("variables")

        logger.info("Executing raw GraphQL from Claude", query_length=len(query))

        result = await self.graphql_client.execute_raw_query(query, variables)
        return json.dumps(result)

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool call via direct Shopify GraphQL API.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool result as JSON string

        Raises:
            ValueError: If tool execution fails or tool is unknown
        """
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

            # Log tool usage to database (for later analysis)
            try:
                self.db_ops.log_tool_usage(
                    user_id=0,  # Will be set by caller if needed
                    tool_name=tool_name,
                    parameters=json.dumps(tool_input),
                    success=False,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as log_error:
                logger.warning(
                    "Failed to log tool usage",
                    error=str(log_error),
                )

            # Log the query error for learning — so future queries avoid the same mistake
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
                logger.info(
                    "Query error logged for learning",
                    tool_name=tool_name,
                    error_type=error_type,
                    lesson=lesson,
                )
            except Exception as learn_error:
                logger.warning(
                    "Failed to log query error for learning",
                    error=str(learn_error),
                )

            raise ValueError(f"Tool execution failed: {tool_name} - {str(e)}")

    def _build_messages(self, user_id: int, message: str, session_id: int = None) -> List[Dict[str, Any]]:
        """
        Build messages array from conversation history and current message.

        Args:
            user_id: User ID for fetching conversation history
            message: Current user message
            session_id: Optional session ID for session-aware message loading

        Returns:
            List of message dictionaries in Claude's format
        """
        try:
            messages = []

            # Get conversation history (session-aware if provided)
            if session_id:
                try:
                    recent_conversations = self.db_ops.get_session_conversations(session_id)
                except Exception as session_err:
                    logger.warning(
                        "Failed to get session conversations, falling back to recent",
                        session_id=session_id,
                        error=str(session_err),
                    )
                    recent_conversations = self.db_ops.get_recent_conversations(
                        user_id,
                        limit=self.settings.conversation_history_limit,
                    )
            else:
                recent_conversations = self.db_ops.get_recent_conversations(
                    user_id,
                    limit=self.settings.conversation_history_limit,
                )

            # If session_id is provided, try to get previous session summary
            if session_id:
                try:
                    prev = self.db_ops.get_previous_session(user_id, session_id)
                    if prev and prev.session_summary:
                        messages.append({"role": "user", "content": f"[Previous session context: {prev.session_summary}]"})
                        messages.append({"role": "assistant", "content": "Understood, I have context from our previous conversation."})
                except Exception:
                    pass

            # Add historical messages in reverse chronological order (oldest first)
            for conv in reversed(recent_conversations):
                # Add user message
                messages.append(
                    {
                        "role": "user",
                        "content": conv.message,
                    }
                )

                # Add assistant response
                messages.append(
                    {
                        "role": "assistant",
                        "content": conv.response,
                    }
                )

            # Add current message
            messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            logger.debug(
                "Messages built",
                user_id=user_id,
                session_id=session_id,
                total_messages=len(messages),
            )

            return messages

        except Exception as e:
            logger.warning(
                "Failed to build messages from history",
                user_id=user_id,
                session_id=session_id,
                error=str(e),
            )

            # Return just the current message if history retrieval fails
            return [{"role": "user", "content": message}]

    @staticmethod
    def _classify_error(error_message: str, tool_name: str) -> str:
        """
        Classify an error into a category for better learning.

        Args:
            error_message: The error string
            tool_name: Which tool produced the error

        Returns:
            Error type classification string
        """
        error_lower = error_message.lower()

        # GraphQL-specific error patterns
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

        # Network/API errors
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
        """
        Generate a concise, actionable lesson from a failed tool call.

        Args:
            tool_name: Which tool was called
            tool_input: What parameters were sent
            error_message: The error that came back
            error_type: Classified error type

        Returns:
            A short lesson string that helps avoid the same error in the future
        """
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

        # Add the specific query snippet for context
        if tool_name == "shopify_graphql" and "query" in tool_input:
            query_snippet = str(tool_input["query"])[:150]
            lesson += f" Failed query snippet: {query_snippet}"

        return lesson
