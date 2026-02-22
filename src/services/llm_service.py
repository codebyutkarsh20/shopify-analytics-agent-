"""Abstract LLM service base class for multi-provider support.

Provides shared logic for system prompt building, message history,
tool execution, and error learning. Provider-specific API calls,
tool formatting, and response parsing are delegated to subclasses.
"""

import json
import re
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
from src.services.chart_generator import ChartGenerator
from src.services.shopify_graphql import ShopifyGraphQLClient
from src.utils.logger import get_logger
from src.utils.timezone import now_ist, IST
from src.services.tools.shopify_analytics import ShopifyAnalyticsTool
from src.services.tools.shopify_graphql import ShopifyGraphQLTool
from src.services.tools.chart_generator import ChartGeneratorTool

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
        chart_generator: Optional[ChartGenerator] = None,
    ):
        self.settings = settings
        self.db_ops = db_ops
        self.context_builder = context_builder
        self.graphql_client = graphql_client
        self.template_manager = template_manager
        self.recovery_manager = recovery_manager
        self.chart_generator = chart_generator

        # Track tool calls per message for learning
        self.last_tool_calls: List[Dict] = []
        # Track chart image files generated during message processing
        self.last_chart_files: List[str] = []

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

    async def get_classification(self, query: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Get a lightweight classification from the LLM.
        
        Args:
            query: User query to classify
            system_prompt: Instructions for classification
            
        Returns:
            Dict containing classification results or None if failed
        """
        try:
            messages = [{"role": "user", "content": f"Query: {query}"}]
            # We use an empty list for tools to force a defined output or just text
            # Some providers might need specific handling, but _call_api handles the protocol.
            response = self._call_api(system_prompt, messages, tools=[])
            
            text, _ = self._parse_response(response)
            
            # extract JSON from text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None
        except Exception as e:
            logger.warning("Classification failed", error=str(e))
            return None

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

            # Reset tool call and chart tracking
            self.last_tool_calls = []
            self.last_chart_files = []
            session_key = f"{user_id}_{session_id or 'nosession'}"

            # Determine query type for context prioritization
            current_query_type = None
            if intent:
                if hasattr(intent, 'coarse'):
                    current_query_type = intent.coarse
                elif isinstance(intent, str):
                    # Fallback if a string was passed (old behavior or fine-grained only)
                    # We might not know coarse type easily unless we parse it or passed it differently.
                    # For now, let's assume if it's a string, it might be the coarse one or just ignore.
                    current_query_type = intent

            # Build conversation history and system prompt
            messages = self._build_messages(user_id, message, session_id)
            system_prompt = self._build_system_prompt(user_id, current_query_type, current_query_text=message)

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

    async def get_classification(self, query: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Get a lightweight classification from the LLM.
        
        Args:
            query: User query to classify
            system_prompt: Instructions for classification
            
        Returns:
            Dict containing classification results or None if failed
        """
        try:
            messages = [{"role": "user", "content": f"Query: {query}"}]
            # We use an empty list for tools to force a text response
            response = self._call_api(system_prompt, messages, tools=[])
            
            text, _ = self._parse_response(response)
            
            # Extract JSON from the response text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            logger.warning("No JSON found in classification response", response_text=text[:100])
            return None
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return None

    def _get_base_tool_definitions(self) -> List[Dict]:
        """Return provider-neutral tool definitions.

        These use the Anthropic-style schema (name, description, input_schema)
        as the canonical format. Provider subclasses convert as needed.
        """
        tools = []
        if self.graphql_client:
            # We instantiate with no args for now as they are stateles schemas
            tools.append(ShopifyAnalyticsTool().to_dict())
            tools.append(ShopifyGraphQLTool().to_dict())
        if self.chart_generator:
            tools.append(ChartGeneratorTool().to_dict())
        return tools

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool call via direct Shopify GraphQL API or chart generation."""
        start_time = time.time()

        try:
            # Chart generation doesn't require GraphQL client
            if tool_name == "generate_chart":
                if not self.chart_generator:
                    raise ValueError("Chart generator not initialized")
                result_json = await self._execute_chart_tool(tool_input)
            elif not self.graphql_client:
                raise ValueError("GraphQL client not initialized — check Shopify credentials")
            else:
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

            # Log successful tool usage
            try:
                self.db_ops.log_tool_usage(
                    user_id=0,
                    tool_name=tool_name,
                    parameters=json.dumps(tool_input, default=str)[:2000],
                    success=True,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as log_error:
                logger.warning("Failed to log tool usage", error=str(log_error))

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

    @staticmethod
    def _normalize_date_filters(query_str: str) -> str:
        """Normalize date filters in Shopify query strings from IST to UTC.

        Safety net: if the LLM generates date-only filters like
        'created_at:>2026-02-09' (which Shopify treats as UTC midnight),
        this converts them to the correct UTC boundary for IST midnight.

        IST = UTC + 5:30, so IST midnight = previous day 18:30 UTC.
        """
        if not query_str:
            return query_str

        import pytz
        from datetime import datetime as _dt

        # Pattern: created_at or updated_at with a bare date (no time component)
        # Matches: created_at:>2026-02-09, created_at:>=2026-02-09, created_at:<2026-02-09
        bare_date_re = re.compile(
            r'((?:created_at|updated_at)\s*:\s*(?:>=?|<=?))\s*(\d{4}-\d{2}-\d{2})(?!T)'
        )

        def _convert_match(m):
            operator = m.group(1)
            date_str = m.group(2)
            try:
                # Parse the bare date as IST midnight
                naive_dt = _dt.strptime(date_str, "%Y-%m-%d")
                ist_midnight = IST.localize(naive_dt)
                utc_equivalent = ist_midnight.astimezone(pytz.UTC)
                utc_str = utc_equivalent.strftime("%Y-%m-%dT%H:%M:%SZ")
                logger.info(
                    "Date filter normalized IST→UTC",
                    original=f"{operator}{date_str}",
                    normalized=f"{operator}{utc_str}",
                )
                return f"{operator}{utc_str}"
            except (ValueError, Exception) as e:
                logger.warning("Failed to normalize date filter", error=str(e))
                return m.group(0)

        normalized = bare_date_re.sub(_convert_match, query_str)
        return normalized

    async def _execute_analytics_tool(self, tool_input: Dict[str, Any]) -> str:
        """Execute the shopify_analytics tool."""
        resource = tool_input.get("resource", "products")
        sort_key = tool_input.get("sort_key", "CREATED_AT")
        reverse = tool_input.get("reverse", True)
        limit = tool_input.get("limit", 10)
        query = self._normalize_date_filters(tool_input.get("query"))
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

    async def _execute_chart_tool(self, tool_input: Dict[str, Any]) -> str:
        """Execute the generate_chart tool — create a chart image.

        The generated chart file path is stored in self.last_chart_files
        so the message handler can send it as a Telegram photo.

        Returns:
            JSON string with chart metadata for the LLM
        """
        chart_type = tool_input.get("chart_type", "bar")
        title = tool_input.get("title", "Chart")
        labels = tool_input.get("labels", [])
        values = tool_input.get("values", [])
        y_axis_label = tool_input.get("y_axis_label")

        if not labels or not values:
            raise ValueError("Both 'labels' and 'values' are required and must not be empty")
        if len(labels) != len(values):
            raise ValueError(
                f"labels ({len(labels)}) and values ({len(values)}) must have the same length"
            )

        logger.info(
            "Generating chart",
            chart_type=chart_type,
            title=title,
            data_points=len(labels),
        )

        # Route to appropriate chart generator method
        if chart_type == "line":
            filepath = self.chart_generator.generate_line_chart(
                labels, values, title, y_axis_label
            )
        elif chart_type == "bar":
            filepath = self.chart_generator.generate_bar_chart(
                labels, values, title, y_axis_label
            )
        elif chart_type == "pie":
            filepath = self.chart_generator.generate_pie_chart(
                labels, values, title
            )
        elif chart_type == "horizontal_bar":
            filepath = self.chart_generator.generate_horizontal_bar_chart(
                labels, values, title, y_axis_label
            )
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        if not filepath:
            raise ValueError("Chart generation failed — check logs for details")

        # Store file path; the index is used by the LLM for inline placement
        chart_index = len(self.last_chart_files)
        self.last_chart_files.append(filepath)

        logger.info(
            "Chart generated successfully",
            filepath=filepath, chart_type=chart_type, chart_index=chart_index,
        )

        return json.dumps({
            "status": "success",
            "chart_type": chart_type,
            "title": title,
            "chart_index": chart_index,
            "IMPORTANT": (
                f"You MUST write [CHART:{chart_index}] on its own line in your "
                f"final response, exactly where you want this chart image to appear. "
                f"Write intro text ABOVE the marker and analysis/insight BELOW it. "
                f"Do NOT say 'charts sent above' or 'see charts below' — the image "
                f"appears inline wherever you place [CHART:{chart_index}]."
            ),
        })

    def _build_system_prompt(self, user_id: int, query_type: Optional[str] = None, current_query_text: Optional[str] = None) -> str:
        """Build the system prompt with context, tool guidance, and formatting rules."""
        try:
            # Pre-compute UTC date boundaries so the LLM doesn't need timezone math
            from datetime import timedelta
            import pytz

            now_ist_dt = now_ist()
            now_ist_aware = IST.localize(now_ist_dt) if now_ist_dt.tzinfo is None else now_ist_dt
            now_utc = now_ist_aware.astimezone(pytz.UTC)

            # IST day boundaries in UTC (for Shopify API filtering)
            today_start_ist = now_ist_aware.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start_ist.astimezone(pytz.UTC)

            yesterday_start_ist = today_start_ist - timedelta(days=1)
            yesterday_start_utc = yesterday_start_ist.astimezone(pytz.UTC)
            yesterday_end_utc = today_start_utc  # yesterday ends where today starts

            week_ago_ist = today_start_ist - timedelta(days=7)
            week_ago_utc = week_ago_ist.astimezone(pytz.UTC)

            month_ago_ist = today_start_ist - timedelta(days=30)
            month_ago_utc = month_ago_ist.astimezone(pytz.UTC)

            # Format UTC timestamps for Shopify query filters
            utc_fmt = "%Y-%m-%dT%H:%M:%SZ"
            
            # Chain of Thought & Intelligence Enhancements
            reasoning_instruction = (
                "\n\nIMPORTANT: BEFORE calling any tools, you must perform a 'Chain of Thought' reasoning step.\n"
                "1. Analyze the user's intent: Is it a simple lookup, a comparison, or a complex analysis?\n"
                "2. Identify the time range: Convert relative terms (today, last week) to specific UTC ISO-8601 timestamps.\n"
                "   - Current IST time: {now_ist}\n"
                "   - 'Today' starts at: {today_utc} (UTC)\n"
                "   - 'Yesterday' is: {yesterday_start_utc} to {yesterday_end_utc} (UTC)\n"
                "   - 'Last 7 Days' includes data since: {week_ago_utc} (UTC)\n"
                "   - 'Last 30 Days' includes data since: {month_ago_utc} (UTC)\n"
                "3. Check for specific filters: Status, Tags, Fulfillment, etc.\n"
                "4. Choose the best tool: Use 'shopify_analytics' for standard queries, 'shopify_graphql' ONLY for complex nested data.\n"
                "5. Formulate the query parameters carefully.\n"
            ).format(
                now_ist=now_ist_aware.strftime("%Y-%m-%d %H:%M:%S %Z"),
                today_utc=today_start_utc.strftime(utc_fmt),
                yesterday_start_utc=yesterday_start_utc.strftime(utc_fmt),
                yesterday_end_utc=yesterday_end_utc.strftime(utc_fmt),
                week_ago_utc=week_ago_utc.strftime(utc_fmt),
                month_ago_utc=month_ago_utc.strftime(utc_fmt),
            )

            prompt = (
                f"You are the Shopify Analytics Agent, an intelligent assistant for store owners.\n"
                f"Current Time (IST): {now_ist_aware.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{reasoning_instruction}\n"
                f"You have access to the following tools to fetch data directly from Shopify:\n"
            )
            today_start_utc_str = today_start_utc.strftime(utc_fmt)
            yesterday_start_utc_str = yesterday_start_utc.strftime(utc_fmt)
            yesterday_end_utc_str = yesterday_end_utc.strftime(utc_fmt)
            week_ago_utc_str = week_ago_utc.strftime(utc_fmt)
            month_ago_utc_str = month_ago_utc.strftime(utc_fmt)
            now_utc_str = now_utc.strftime(utc_fmt)

            system_parts = [
                "You are a Shopify Analytics Assistant. Your role is to help merchants analyze their store data, understand sales trends, and make data-driven decisions.",
                "",
                f"Current date and time (IST, Indian Standard Time, UTC+5:30): {now_ist_aware.isoformat()}",
                f"Current date and time (UTC, for Shopify API queries): {now_utc_str}",
                f"Store domain: {self.settings.shopify.shop_domain}",
                "",
                "IMPORTANT — TIMEZONE HANDLING:",
                "- All timestamps in tool results are ALREADY converted to IST (Indian Standard Time, UTC+5:30).",
                "- Do NOT convert or adjust any timestamps — display them exactly as received.",
                "- When showing a timestamp like '2026-02-09T00:15:00+05:30', display it as 'Feb 9, 2026 at 12:15 AM IST'.",
                "- Always append 'IST' when showing times to the user.",
                "",
                "CRITICAL — DATE FILTERING IN SHOPIFY QUERIES (READ CAREFULLY):",
                "Shopify's API uses UTC internally. IST is UTC+5:30, so date boundaries differ!",
                "You MUST use the pre-computed UTC boundaries below when filtering by date.",
                "NEVER use IST dates directly in query filters — always use these UTC values:",
                "",
                f"  'Today' (IST)       → created_at:>={today_start_utc_str}",
                f"  'Yesterday' (IST)   → created_at:>={yesterday_start_utc_str} created_at:<{yesterday_end_utc_str}",
                f"  'Last 7 days' (IST) → created_at:>={week_ago_utc_str}",
                f"  'Last 30 days'(IST) → created_at:>={month_ago_utc_str}",
                "",
                "Example: If a user asks 'what are today's orders?', use:",
                f"  query: 'created_at:>={today_start_utc_str}'",
                "Do NOT use 'created_at:>2026-02-09' — this would use UTC midnight and miss",
                "orders placed between 12:00 AM and 5:30 AM IST.",
            ]

            # Context from context builder
            if self.context_builder:
                try:
                    context = self.context_builder.build_context(
                        user_id, current_query_text=current_query_text
                    )

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
                    "   - Filter with Shopify queries: 'status:active', 'financial_status:paid'",
                    "   - Date filters MUST use the pre-computed UTC boundaries above (e.g., created_at:>=YYYY-MM-DDTHH:MM:SSZ)",
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

            if self.chart_generator:
                system_parts.extend([
                    "",
                    "3. generate_chart — Create chart images that appear INLINE in your response",
                    "   - Query data first, then call generate_chart with extracted labels and values",
                    "   - Chart types: 'bar' (comparisons), 'line' (trends), 'pie' (distribution), 'horizontal_bar' (ranked lists)",
                    "",
                    "WHEN TO GENERATE CHARTS:",
                    "- GENERATE a chart when: user explicitly asks ('show chart', 'graph', 'visualize'), or data has 3+ comparison points and visual adds value",
                    "- DO NOT generate a chart when: user says 'no chart'/'text only'/'brief'/'just the number', or it's a simple single-number answer",
                    "- When in doubt: skip the chart for simple queries, generate for complex multi-point data",
                    "",
                    "CHART INLINE PLACEMENT — MANDATORY RULES (only when you DO generate a chart):",
                    "",
                    "After calling generate_chart, the result gives you a chart_index (0, 1, 2, ...).",
                    "You MUST write [CHART:N] on its own line in your final response to place the image.",
                    "The system splits your text at those markers and sends text + images interleaved.",
                    "",
                    "RULES (follow strictly):",
                    "1. Every generated chart MUST have a [CHART:N] marker in your response — no exceptions.",
                    "2. Place each [CHART:N] on its OWN line, between text paragraphs.",
                    "3. Write context/intro text ABOVE each [CHART:N], then insight/analysis BELOW it.",
                    "4. NEVER say 'charts sent above', 'see the charts below', or 'charts attached'.",
                    "   The image appears right where you put [CHART:N] — refer to it as 'the chart above' AFTER the marker.",
                    "5. For multi-chart responses, INTERLEAVE charts with text — don't group all charts together.",
                    "6. Keep each text section between charts SHORT (2-5 lines max). Let the chart speak.",
                    "",
                    "GOOD example (text-chart-text-chart-text):",
                    "",
                    "  **📊 Store Revenue Analysis**",
                    "",
                    "  Here are your top 5 products by revenue:",
                    "",
                    "  [CHART:0]",
                    "",
                    "  **ABC** dominates with ₹5.1L — but note the ₹5.9L refund. **Combat Trimmer** at ₹1.02L is your strongest net performer.",
                    "",
                    "  Now let's look at the monthly revenue trend:",
                    "",
                    "  [CHART:1]",
                    "",
                    "  November 2025 was your peak month at ₹1.26L from 20 orders. February 2026 is off to a strong start.",
                    "",
                    "  Here's how your payment statuses break down:",
                    "",
                    "  [CHART:2]",
                    "",
                    "  ⚠️ 60% orders have pending payments — follow up to convert ₹2L+ in revenue.",
                    "",
                    "BAD example (NEVER do this):",
                    "  [all text in one block, then] 'Charts sent above visualize all this data!'",
                    "  [or] grouping all [CHART:0] [CHART:1] [CHART:2] together in one spot",
                    "",
                    "CHART TYPE SELECTION:",
                    "- Rankings, top-N, comparisons → 'bar' or 'horizontal_bar'",
                    "- Revenue/orders over time → 'line'",
                    "- Status distribution, market share → 'pie'",
                    "- Keep titles descriptive: 'Top 5 Products by Revenue' not just 'Products'",
                ])

            system_parts.extend([
                "",
                "DATA INTEGRITY — MANDATORY RULES (MOST IMPORTANT SECTION):",
                "",
                "1. ALWAYS USE TOOLS FOR DATA QUESTIONS — never answer from memory or conversation history.",
                "   Every time the user asks about orders, revenue, products, customers, or any store data,",
                "   you MUST call shopify_analytics or shopify_graphql to get fresh data. NEVER say",
                "   'based on our earlier conversation' or reuse numbers from previous messages.",
                "",
                "2. NEVER FABRICATE OR GUESS DATA — if a tool call returns no results, say exactly that.",
                "   Do not invent order numbers, amounts, dates, or product names. Only report what the",
                "   tool actually returned.",
                "",
                "3. IF THE USER SAYS YOU ARE WRONG — re-query with DIFFERENT or BROADER parameters.",
                "   Do NOT just apologize and make up different numbers. Call the tool again with a wider",
                "   date range or fewer filters. Let the real data speak.",
                "",
                "4. NEVER CONTRADICT YOUR OWN TOOL RESULTS — if a tool returned 11 orders, report 11 orders.",
                "   Do not later say 'actually there are 0 orders' without a new tool call proving it.",
                "",
                "5. SHOW YOUR WORK — when reporting data, mention what query/filter you used so the user",
                "   can understand what was searched. Example: 'I queried orders with created_at>=2026-02-08T18:30:00Z'.",
                "",
            ])

            system_parts.extend([
                "",
                "USER INSTRUCTION FOLLOWING — ADAPT TO EACH MESSAGE:",
                "",
                "You MUST respect per-message formatting instructions from the user.",
                "The user's explicit instructions ALWAYS override your default behavior.",
                "Scan each message for these cues and adapt accordingly:",
                "",
                "FORMAT LENGTH CUES:",
                "- 'brief', 'short', 'quick', 'just tell me', 'one line', 'summary' → Keep answer to 1-3 sentences. No lists, no charts, just the key number(s).",
                "- 'detailed', 'elaborate', 'explain', 'break down', 'in depth', 'full analysis' → Give comprehensive analysis with breakdowns, comparisons, and insights.",
                "- 'just the number', 'just numbers', 'how much', 'how many' (without elaboration request) → Reply with ONLY the number/stat. Example: '₹67,364.78' or '11 orders'.",
                "",
                "CHART / VISUAL CUES:",
                "- 'chart', 'graph', 'visualize', 'plot', 'show me a chart' → Generate a chart using the generate_chart tool.",
                "- 'no chart', 'without chart', 'text only', 'no graph', 'skip the chart' → Do NOT generate any chart. Respond with text only.",
                "- If the user does not mention charts at all → Use your judgment: generate a chart only when data has 3+ comparison points AND visual representation clearly adds value. For simple single-number answers, NEVER generate a chart.",
                "",
                "COMPARISON / FORMAT CUES:",
                "- 'compare', 'versus', 'vs', 'side by side' → Show data in a structured comparison format.",
                "- 'list', 'rank', 'top N' → Use numbered list format.",
                "- 'table' → Format as a clean text table (use fixed-width formatting).",
                "",
                "EXAMPLES OF CORRECT ADAPTATION:",
                "  User: 'how many orders today?' → '11 orders today totaling ₹67,364.78'  (brief by default for simple questions)",
                "  User: 'how many orders today? give me details' → Full breakdown with each order, amounts, statuses, trends",
                "  User: 'show revenue with a chart' → Query data + generate chart + brief text around it",
                "  User: 'show revenue, no chart' → Query data + text-only response with numbers",
                "  User: 'just the total revenue this month' → '₹2,45,678.90'  (just the number)",
                "",
                "IMPORTANT: When in doubt about length/format, match the complexity of the user's question.",
                "Simple questions get simple answers. Complex questions get detailed answers.",
                "NEVER give a 10-line response to 'how many orders today?' unless the user asks for details.",
                "",
            ])

            system_parts.extend([
                "",
                "ANALYSIS GUIDELINES (apply when user asks for analysis or detailed responses):",
                "- Always include specific numbers: revenue, order count, AOV, units sold",
                "- Explain what the numbers mean in plain language",
                "- Highlight notable trends, outliers, or changes worth attention",
                "- End with 1-2 short actionable recommendations when useful",
                "- Be concise — no filler, every sentence should add value",
                "- When charts are generated, keep text sections SHORT (2-5 lines) between charts",
                "- Let charts do the heavy lifting — don't repeat all the data the chart already shows",
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
                "",
                "Numbers and formatting:",
                "- Currency: ₹1,234.56 (with commas and 2 decimals)",
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
                "- NEVER write 'charts sent above/below' or 'see attached charts' — charts are inline",
            ])

            return "\n".join(system_parts)

        except Exception as e:
            logger.error("Failed to build system prompt", user_id=user_id, error=str(e))
            return "You are a Shopify Analytics Assistant. Help the merchant analyze their store data."

    @staticmethod
    def _detect_format_hints(message: str) -> Optional[str]:
        """Detect per-message format/style instructions from the user's query.

        Scans the message for explicit formatting cues (brief, chart, detailed, etc.)
        and returns a compact directive string that gets injected alongside the message
        so the LLM respects the user's real-time instructions.

        Args:
            message: The raw user message

        Returns:
            A directive string if format cues were found, otherwise None
        """
        msg_lower = message.lower()
        directives = []

        # Brevity / length cues
        brevity_cues = ["brief", "short answer", "quick", "just tell me", "one line",
                        "just the number", "just numbers", "just the total",
                        "just the count", "only the number", "only the total",
                        "just show number", "just show the number", "numbers only",
                        "how much", "how many",
                        "in short", "tldr", "tl;dr", "summary only",
                        "no details", "no explanation", "simple answer",
                        "without details", "without explanation"]
        detail_cues = ["detailed", "elaborate", "explain", "break down", "in depth",
                       "full analysis", "comprehensive", "deep dive", "thorough"]

        if any(cue in msg_lower for cue in brevity_cues):
            directives.append("BRIEF — keep response to 1-3 sentences, just key numbers")
        elif any(cue in msg_lower for cue in detail_cues):
            directives.append("DETAILED — give comprehensive analysis with breakdowns")

        # Chart cues
        chart_positive = ["chart", "graph", "visualize", "plot", "show.*chart",
                          "with chart", "with graph"]
        chart_negative = ["no chart", "without chart", "text only", "no graph",
                          "skip chart", "skip the chart", "don't.*chart",
                          "no visual"]

        if any(re.search(cue, msg_lower) for cue in chart_negative):
            directives.append("NO CHART — respond with text only, do not generate any chart")
        elif any(re.search(cue, msg_lower) for cue in chart_positive):
            directives.append("CHART REQUESTED — generate a chart for this data")

        # Comparison cues
        if any(cue in msg_lower for cue in ["compare", "versus", "vs ", "side by side", "comparison"]):
            directives.append("COMPARISON FORMAT — show data side-by-side")

        # List/ranking cues
        if any(re.search(cue, msg_lower) for cue in [r"top \d+", "rank", "ranking", "list them"]):
            directives.append("LIST/RANKING FORMAT — use numbered list")

        if not directives:
            return None

        return " | ".join(directives)

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

            # Detect format hints from the current message and inject as a directive
            format_hint = self._detect_format_hints(message)
            if format_hint:
                # Prepend format directive so LLM sees it right before the actual question
                message_with_hint = f"[FORMAT: {format_hint}]\n\n{message}"
                messages.append({"role": "user", "content": message_with_hint})
                logger.debug("Format hints detected", hints=format_hint)
            else:
                messages.append({"role": "user", "content": message})

            # Inject a data-freshness reminder if conversation history is present
            # This prevents the LLM from reusing stale/wrong data from earlier turns
            if len(messages) > 1:
                messages.insert(
                    -1,  # Right before the current user message
                    {
                        "role": "user",
                        "content": (
                            "[SYSTEM REMINDER: Any numbers, order counts, or revenue figures "
                            "in the conversation above may be outdated or incorrect. "
                            "You MUST call a tool to get fresh data for every data question. "
                            "Never reuse data from previous messages.]"
                        ),
                    },
                )
                messages.insert(
                    -1,  # Matching assistant turn
                    {
                        "role": "assistant",
                        "content": "Understood. I will always query fresh data from Shopify for every question.",
                    },
                )

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
