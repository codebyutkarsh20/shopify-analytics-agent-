"""OpenAI LLM service implementation."""

import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from src.services.llm_service import LLMService, ToolCall
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIService(LLMService):
    """OpenAI Chat Completions API implementation.

    Uses the OpenAI chat completions API with function/tool calling support.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def _create_api_client(self) -> Any:
        return OpenAI(api_key=self.settings.openai.api_key)

    def _get_model_name(self) -> str:
        return self.settings.openai.model

    def _get_max_tokens(self) -> int:
        return self.settings.openai.max_tokens

    def _format_tools_for_provider(self, base_tools: List[Dict]) -> List[Dict]:
        """Convert base tool schemas to OpenAI function calling format.

        OpenAI expects: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Base format uses: {"name": ..., "description": ..., "input_schema": ...}
        """
        openai_tools = []
        for tool in base_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            })
        return openai_tools

    def _call_api(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict],
    ) -> Any:
        """Call OpenAI Chat Completions API.

        OpenAI takes system prompt as a message, not a separate parameter.
        """
        logger.debug(
            "Calling OpenAI API",
            model=self.model,
            messages_count=len(messages),
            tool_count=len(tools),
        )

        # Prepend system message
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        logger.debug(
            "OpenAI response received",
            finish_reason=response.choices[0].finish_reason if response.choices else "none",
        )
        return response

    def _parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        """Parse OpenAI response into text and tool calls."""
        choice = response.choices[0]
        message = choice.message

        # Extract text
        response_text = message.content or ""

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=arguments,
                ))

        return response_text, tool_calls

    def _is_tool_use_response(self, response: Any) -> bool:
        """Check if OpenAI response requests tool use."""
        choice = response.choices[0]
        return choice.finish_reason == "tool_calls"

    def _build_assistant_message(self, response: Any) -> Dict:
        """Build assistant message for OpenAI conversation history.

        OpenAI requires the full assistant message including tool_calls
        to be echoed back so tool results can reference the correct IDs.
        """
        choice = response.choices[0]
        message = choice.message

        msg = {
            "role": "assistant",
            "content": message.content or "",
        }

        # Include tool_calls if present (required for OpenAI tool result matching)
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return msg

    def _build_tool_results_message(self, tool_results: List[Dict]) -> List[Dict]:
        """Build OpenAI-format tool results (separate message per tool result).

        OpenAI uses individual 'tool' role messages, one per tool call result,
        unlike Anthropic which bundles them in a single 'user' message.
        """
        messages = []
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            })
        return messages
