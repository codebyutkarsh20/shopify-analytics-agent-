"""Learning system modules for Shopify Analytics Agent.

This package contains modules for learning user patterns, building context,
and managing preferences:

- pattern_learner: Learns query patterns from user interactions
- context_builder: Builds rich context from user history
- preference_manager: Manages user preferences and learning
- session_manager: Manages conversation sessions for single-thread platforms
- template_manager: Learns and manages successful query templates
- recovery_manager: Learns error recovery patterns
- feedback_analyzer: Analyzes implicit response quality feedback
- insight_aggregator: Aggregates cross-user intelligence
- template_seeds: Provides seed templates for common queries
"""

from src.learning.context_builder import ContextBuilder
from src.learning.feedback_analyzer import FeedbackAnalyzer
from src.learning.insight_aggregator import InsightAggregator
from src.learning.pattern_learner import PatternLearner
from src.learning.preference_manager import PreferenceManager
from src.learning.recovery_manager import RecoveryManager
from src.learning.session_manager import SessionManager
from src.learning.template_manager import TemplateManager
from src.learning.template_seeds import seed_templates

__all__ = [
    "ContextBuilder",
    "FeedbackAnalyzer",
    "InsightAggregator",
    "PatternLearner",
    "PreferenceManager",
    "RecoveryManager",
    "SessionManager",
    "TemplateManager",
    "seed_templates",
]
