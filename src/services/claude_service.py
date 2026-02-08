"""Backward-compatible alias for AnthropicService.

This module exists so that existing imports like:
    from src.services.claude_service import ClaudeService
continue to work after the multi-LLM refactoring.
"""

from src.services.anthropic_service import AnthropicService as ClaudeService

__all__ = ["ClaudeService"]
