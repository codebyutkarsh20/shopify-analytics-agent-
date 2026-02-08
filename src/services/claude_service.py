"""Claude API integration service for processing Shopify analytics queries."""

import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import anthropic

from src.config.settings import Settings
from src.database.operations import DatabaseOperations
from src.learning.context_builder import ContextBuilder
from src.services.mcp_service import MCPService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeService:
    """Service for integrating Claude API with Shopify analytics."""

    def __init__(
        self,
        settings: Settings,
        mcp_service: MCPService,
        db_ops: DatabaseOperations,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize the Claude service.

        Args:
            settings: Settings object with anthropic configuration
            mcp_service: MCPService instance for executing tools
            db_ops: DatabaseOperations instance for database access
            context_builder: ContextBuilder instance for user context (optional)
        """
        self.settings = settings
        self.mcp_service = mcp_service
        self.db_ops = db_ops
        self.context_builder = context_builder

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=settings.anthropic.api_key)
        self.model = settings.anthropic.model
        self.max_tokens = settings.anthropic.max_tokens

    async def process_message(self, user_id: int, message: str) -> str:
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
            )

            # Build conversation history
            messages = self._build_messages(user_id, message)

            # Build system prompt with context
            system_prompt = self._build_system_prompt(user_id)

            # Get available tools dynamically from MCP server
            tools = self.mcp_service.get_tools_for_claude()
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
                except Exception as e:
                    logger.warning(
                        "Failed to build context from context_builder",
                        error=str(e),
                    )

            # Dynamically list all available tools from MCP
            available_tools = self.mcp_service.get_tools_for_claude()
            if available_tools:
                system_parts.append("")
                system_parts.append("You have access to the following tools:")
                for tool in available_tools:
                    system_parts.append(f"- {tool['name']}: {tool['description']}")

            system_parts.extend(
                [
                    "",
                    "Guidelines for analysis:",
                    "- Always provide specific metrics when available (revenue, order count, AOV)",
                    "- Explain trends and patterns in the data",
                    "- Suggest actionable insights based on the data",
                    "- Use appropriate date ranges for comparisons",
                    "- Be concise but comprehensive in your analysis",
                    "- Format responses cleanly for Telegram (avoid excessive markdown)",
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

    # NOTE: Tools are no longer hardcoded here.
    # They are discovered dynamically from the MCP server at startup via
    # mcp_service.get_tools_for_claude() — see process_message() above.

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool call via MCP service.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool result as JSON string

        Raises:
            ValueError: If tool execution fails
            RuntimeError: If MCP service is not running
        """
        start_time = time.time()
        success = False

        try:
            logger.info(
                "Executing tool via MCP",
                tool_name=tool_name,
                tool_input=tool_input,
            )

            # Call the tool via MCP service
            result = await self.mcp_service.call_tool(tool_name, tool_input)

            execution_time_ms = int((time.time() - start_time) * 1000)
            success = True

            logger.info(
                "Tool execution successful",
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )

            # Convert result to JSON string
            return json.dumps(result)

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

            raise ValueError(f"Tool execution failed: {tool_name} - {str(e)}")

    def _build_messages(self, user_id: int, message: str) -> List[Dict[str, Any]]:
        """
        Build messages array from conversation history and current message.

        Args:
            user_id: User ID for fetching conversation history
            message: Current user message

        Returns:
            List of message dictionaries in Claude's format
        """
        try:
            messages = []

            # Get recent conversation history
            recent_conversations = self.db_ops.get_recent_conversations(
                user_id,
                limit=self.settings.conversation_history_limit,
            )

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
                total_messages=len(messages),
            )

            return messages

        except Exception as e:
            logger.warning(
                "Failed to build messages from history",
                user_id=user_id,
                error=str(e),
            )

            # Return just the current message if history retrieval fails
            return [{"role": "user", "content": message}]
