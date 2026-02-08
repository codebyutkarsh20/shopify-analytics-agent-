"""Anthropic (Claude) LLM service implementation."""

from typing import Any, Dict, List, Tuple

import anthropic

from src.services.llm_service import LLMService, ToolCall
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicService(LLMService):
    """Anthropic Claude API implementation.

    Uses the Anthropic Messages API with native tool use support.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _create_api_client(self) -> Any:
        return anthropic.Anthropic(api_key=self.settings.anthropic.api_key)

    def _get_model_name(self) -> str:
        return self.settings.anthropic.model

    def _get_max_tokens(self) -> int:
        return self.settings.anthropic.max_tokens

    def _format_tools_for_provider(self, base_tools: List[Dict]) -> List[Dict]:
        """Anthropic uses the base format directly (name, description, input_schema)."""
        return base_tools

    def _call_api(
        self,
        system_prompt: str,
        messages: List[Dict],
        tools: List[Dict],
    ) -> Any:
        """Call Anthropic Messages API."""
        logger.debug(
            "Calling Anthropic API",
            model=self.model,
            messages_count=len(messages),
            tool_count=len(tools),
        )

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        logger.debug(
            "Anthropic response received",
            stop_reason=response.stop_reason,
            content_blocks=len(response.content),
        )
        return response

    def _parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        """Parse Anthropic response into text and tool calls."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))

        response_text = "\n".join(text_parts) if text_parts else ""
        return response_text, tool_calls

    def _is_tool_use_response(self, response: Any) -> bool:
        """Check if Anthropic response requests tool use."""
        return response.stop_reason == "tool_use"

    def _build_assistant_message(self, response: Any) -> Dict:
        """Build assistant message with full Anthropic content blocks."""
        return {
            "role": "assistant",
            "content": response.content,
        }

    def _build_tool_results_message(self, tool_results: List[Dict]) -> Dict:
        """Build Anthropic-format tool results (single user message with tool_result blocks)."""
        results = []
        for tr in tool_results:
            result_block = {
                "type": "tool_result",
                "tool_use_id": tr["tool_call_id"],
                "content": tr["content"],
            }
            if tr.get("is_error"):
                result_block["is_error"] = True
            results.append(result_block)

        return {
            "role": "user",
            "content": results,
        }
