"""Implicit response quality feedback analysis.

Infers response quality from user behavior patterns without explicit ratings.
Detects corrections, clarifications, frustration, refinements, and satisfaction.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackAnalyzer:
    """Analyzes implicit feedback from user follow-ups.

    Examines user messages after receiving responses to infer satisfaction:
    - Positive feedback: thanks, excellent, exactly
    - Negative feedback: wrong, doesn't work, useless
    - Corrections: trying alternate approach
    - Refinements: requesting more detail
    - Topic continuity: same or new topic
    """

    POSITIVE_KEYWORDS = [
        "thanks",
        "thank you",
        "great",
        "perfect",
        "awesome",
        "helpful",
        "exactly",
        "nice",
        "good",
        "excellent",
        "precisely",
        "spot on",
        "just what",
    ]

    NEGATIVE_KEYWORDS = [
        "wrong",
        "incorrect",
        "not what i",
        "doesn't work",
        "that's not",
        "useless",
        "terrible",
        "bad",
        "not helpful",
        "not right",
        "wrong answer",
    ]

    CORRECTION_KEYWORDS = [
        "no,",
        "not that",
        "i meant",
        "i said",
        "actually",
        "instead",
        "try again",
        "different",
        "let me",
        "i want",
    ]

    REFINEMENT_KEYWORDS = [
        "more detail",
        "tell me more",
        "can you also",
        "what about",
        "break it down",
        "drill down",
        "specifically",
        "how about",
        "and also",
        "expand on",
    ]

    def __init__(self):
        """Initialize FeedbackAnalyzer."""
        logger.info("FeedbackAnalyzer initialized")

    def analyze_follow_up(
        self,
        previous_query_type: str,
        current_message: str,
        current_query_type: str,
        time_gap_seconds: float = 0,
    ) -> dict:
        """Analyze a follow-up message for implicit feedback.

        Detects feedback signals in order of priority:
        1. Explicit positive ("thanks", "great")
        2. Correction ("no, I meant...")
        3. Explicit negative ("wrong", "useless")
        4. Refinement ("more details")
        5. Same topic continuation
        6. New topic (default)

        Args:
            previous_query_type: Type of previous query
            current_message: Current user message
            current_query_type: Current query type
            time_gap_seconds: Seconds since last message

        Returns:
            Dict with keys:
            - "feedback_type": str (explicit_positive, follow_up_correction, etc.)
            - "quality_score": float (-0.8 to 0.8)
            - "signal_text": str (extracted signal text or None)
        """
        msg_lower = current_message.lower()

        # Check 1: Explicit positive
        if self._contains_keywords(msg_lower, self.POSITIVE_KEYWORDS):
            logger.debug(
                "Detected explicit positive feedback",
                message_preview=current_message[:80],
            )
            return {
                "feedback_type": "explicit_positive",
                "quality_score": 0.8,
                "signal_text": current_message[:100],
            }

        # Check 2: Correction
        if self._contains_keywords(msg_lower, self.CORRECTION_KEYWORDS):
            logger.debug(
                "Detected correction follow-up",
                message_preview=current_message[:80],
            )
            return {
                "feedback_type": "follow_up_correction",
                "quality_score": -0.5,
                "signal_text": current_message[:100],
            }

        # Check 3: Explicit negative
        if self._contains_keywords(msg_lower, self.NEGATIVE_KEYWORDS):
            logger.debug(
                "Detected explicit negative feedback",
                message_preview=current_message[:80],
            )
            return {
                "feedback_type": "explicit_negative",
                "quality_score": -0.8,
                "signal_text": current_message[:100],
            }

        # Check 4: Refinement
        if self._contains_keywords(msg_lower, self.REFINEMENT_KEYWORDS):
            logger.debug(
                "Detected refinement follow-up",
                message_preview=current_message[:80],
            )
            return {
                "feedback_type": "follow_up_refinement",
                "quality_score": 0.5,
                "signal_text": current_message[:100],
            }

        # Check 5: Same topic (continuation)
        if (
            current_query_type
            and previous_query_type
            and current_query_type == previous_query_type
        ):
            logger.debug(
                "Detected same-topic follow-up",
                query_type=current_query_type,
            )
            return {
                "feedback_type": "follow_up_same_topic",
                "quality_score": 0.2,
                "signal_text": None,
            }

        # Check 6: Different topic
        logger.debug(
            "Detected topic change",
            previous_type=previous_query_type,
            current_type=current_query_type,
        )
        return {
            "feedback_type": "follow_up_new_topic",
            "quality_score": 0.0,
            "signal_text": None,
        }

    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """Check if text contains any of the keywords.

        Args:
            text: Text to search (should be lowercase)
            keywords: List of keywords to find

        Returns:
            True if any keyword found in text
        """
        return any(keyword in text for keyword in keywords)
