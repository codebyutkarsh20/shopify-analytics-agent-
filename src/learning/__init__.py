"""Learning system modules for Shopify Analytics Agent.

This package contains modules for learning user patterns, building context,
and managing preferences:

- pattern_learner: Learns query patterns from user interactions
- context_builder: Builds rich context from user history
- preference_manager: Manages user preferences and learning
"""

from src.learning.context_builder import ContextBuilder
from src.learning.pattern_learner import PatternLearner
from src.learning.preference_manager import PreferenceManager

__all__ = [
    "PatternLearner",
    "ContextBuilder",
    "PreferenceManager",
]
