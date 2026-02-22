"""Multi-factor response quality scoring.

Replaces the old keyword-only approach with a weighted composite score
that combines three independent factors:

    composite = completeness × 0.30 + sentiment × 0.40 + tool_perf × 0.30

Each factor is scored independently, then merged.  Explicit 👍/👎 button
feedback bypasses the formula entirely and sets ±1.0 directly (handled
in handlers.py, not here).

Score range: -1.0 (terrible) … 0.0 (neutral) … +1.0 (excellent)
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Weight constants ──────────────────────────────────────────────────
W_COMPLETENESS = 0.30
W_SENTIMENT = 0.40
W_TOOL_PERF = 0.30


class FeedbackAnalyzer:
    """Multi-factor quality scorer for bot responses.

    Three independent scoring axes are computed and merged:

    1. **Completeness** – did the response contain substance & data?
    2. **Sentiment** – what does the user's follow-up message signal?
    3. **Tool performance** – did the tool calls succeed quickly?
    """

    # ── Keyword lists (used by sentiment factor) ──────────────────────

    POSITIVE_KEYWORDS: list[str] = [
        "thanks", "thank you", "great", "perfect", "awesome",
        "helpful", "exactly", "nice", "good", "excellent",
        "precisely", "spot on", "just what",
    ]

    NEGATIVE_KEYWORDS: list[str] = [
        "wrong", "incorrect", "not what i", "doesn't work",
        "that's not", "useless", "terrible", "bad",
        "not helpful", "not right", "wrong answer",
        "numbers seem off", "seem off", "seems off",
        "no thanks", "nah",
    ]

    CORRECTION_KEYWORDS: list[str] = [
        "no,", "not that", "i meant", "i said", "actually",
        "instead", "try again", "different", "let me rephrase",
    ]

    REFINEMENT_KEYWORDS: list[str] = [
        "more detail", "tell me more", "can you also",
        "what about", "break it down", "drill down",
        "specifically", "how about", "and also", "expand on",
    ]

    # Words that flip positive sentiment to negative when found between
    # or after a positive keyword.
    CONTRADICTION_MARKERS: list[str] = [
        "but", "however", "yet", "although", "though",
        "except", "unfortunately",
    ]

    def __init__(self) -> None:
        """Initialize FeedbackAnalyzer."""
        # Pre-compile word-boundary regexes for positive keywords.
        self._pos_re = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in self.POSITIVE_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )
        self._neg_re = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in self.NEGATIVE_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )
        self._corr_re = re.compile(
            r"(?:" + "|".join(re.escape(k) for k in self.CORRECTION_KEYWORDS) + r")",
            re.IGNORECASE,
        )
        self._refine_re = re.compile(
            r"(?:" + "|".join(re.escape(k) for k in self.REFINEMENT_KEYWORDS) + r")",
            re.IGNORECASE,
        )
        self._contradiction_re = re.compile(
            r"\b(?:" + "|".join(re.escape(m) for m in self.CONTRADICTION_MARKERS) + r")\b",
            re.IGNORECASE,
        )
        # Numbers / data presence in response
        self._data_re = re.compile(r"\d[\d,]*\.?\d*")
        logger.info("FeedbackAnalyzer initialized (multi-factor scoring)")

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def analyze_follow_up(
        self,
        previous_query_type: str,
        current_message: str,
        current_query_type: str,
        time_gap_seconds: float = 0,
        *,
        response: Optional[str] = None,
        tool_calls_json: Optional[str] = None,
    ) -> dict:
        """Score the previous response using multiple quality factors.

        Args:
            previous_query_type: Query type of the previous (scored) response.
            current_message: The user's follow-up message.
            current_query_type: Query type of the current message.
            time_gap_seconds: Seconds since previous response (unused for now).
            response: The bot's previous response text (for completeness scoring).
            tool_calls_json: JSON string of tool calls from the previous response.

        Returns:
            Dict with keys:
              feedback_type  – human-readable label
              quality_score  – composite float in [-1.0, 1.0]
              signal_text    – the user text that triggered sentiment
              sub_scores     – {completeness, sentiment, tool_performance,
                               composite, reasoning}
        """
        # ── Factor 1: Completeness ────────────────────────────────────
        completeness, comp_reason = self._compute_completeness(
            response, tool_calls_json
        )

        # ── Factor 2: Sentiment ───────────────────────────────────────
        sentiment, feedback_type, signal_text, sent_reason = (
            self._compute_sentiment(
                current_message, previous_query_type, current_query_type
            )
        )

        # ── Factor 3: Tool performance ────────────────────────────────
        tool_perf, tool_reason = self._compute_tool_performance(tool_calls_json)

        # ── Composite ─────────────────────────────────────────────────
        composite = (
            completeness * W_COMPLETENESS
            + sentiment * W_SENTIMENT
            + tool_perf * W_TOOL_PERF
        )
        # Clamp to [-1.0, 1.0]
        composite = max(-1.0, min(1.0, composite))

        sub_scores = {
            "completeness": round(completeness, 3),
            "sentiment": round(sentiment, 3),
            "tool_performance": round(tool_perf, 3),
            "composite": round(composite, 3),
            "reasoning": {
                "completeness": comp_reason,
                "sentiment": sent_reason,
                "tool_performance": tool_reason,
            },
        }

        logger.debug(
            "Quality scored",
            feedback_type=feedback_type,
            composite=round(composite, 3),
            completeness=round(completeness, 2),
            sentiment=round(sentiment, 2),
            tool_perf=round(tool_perf, 2),
        )

        return {
            "feedback_type": feedback_type,
            "quality_score": round(composite, 3),
            "signal_text": signal_text,
            "sub_scores": sub_scores,
        }

    # ══════════════════════════════════════════════════════════════════
    #  FACTOR 1 – RESPONSE COMPLETENESS
    # ══════════════════════════════════════════════════════════════════

    def _compute_completeness(
        self,
        response: Optional[str],
        tool_calls_json: Optional[str],
    ) -> tuple[float, str]:
        """Score how substantive the bot's response was.

        Sub-signals (summed, capped at 1.0):
          - Response length:          min(len / 500, 1.0) × 0.3
          - Contains numbers/data:    0 or 0.3
          - Tool call success rate:   (successes / total) × 0.4
        """
        if not response:
            return 0.0, "No response text available"

        reasons: list[str] = []
        score = 0.0

        # -- Length substance --
        length_score = min(len(response) / 500, 1.0) * 0.3
        score += length_score
        reasons.append(f"length={len(response)} chars ({length_score:.2f})")

        # -- Data presence --
        if self._data_re.search(response):
            score += 0.3
            reasons.append("contains data (+0.3)")
        else:
            reasons.append("no data found")

        # -- Tool success rate --
        tools = self._parse_tool_calls(tool_calls_json)
        if tools:
            successes = sum(1 for t in tools if t.get("success"))
            rate = successes / len(tools)
            tool_score = rate * 0.4
            score += tool_score
            reasons.append(f"tools {successes}/{len(tools)} ok ({tool_score:.2f})")
        else:
            reasons.append("no tool calls")

        return min(score, 1.0), "; ".join(reasons)

    # ══════════════════════════════════════════════════════════════════
    #  FACTOR 2 – USER SENTIMENT
    # ══════════════════════════════════════════════════════════════════

    def _compute_sentiment(
        self,
        current_message: str,
        previous_query_type: str,
        current_query_type: str,
    ) -> tuple[float, str, Optional[str], str]:
        """Detect user satisfaction from their follow-up message.

        Returns (score, feedback_type, signal_text, reason).

        Key improvement over the old system:
          - Checks for **negation / contradiction** so "thanks but wrong"
            scores negative, not positive.
          - Uses word-boundary regex instead of substring matching.
          - Checks negative keywords BEFORE positive (reversed priority).
        """
        msg = current_message.strip()
        msg_lower = msg.lower()

        # ── Check 1: Explicit negative (highest priority) ─────────────
        if self._neg_re.search(msg_lower):
            return -0.8, "explicit_negative", msg[:100], "Negative keyword detected"

        # ── Check 2: Correction ───────────────────────────────────────
        if self._corr_re.search(msg_lower):
            return -0.5, "follow_up_correction", msg[:100], "Correction keyword detected"

        # ── Check 3: Positive — but check for contradiction ──────────
        pos_match = self._pos_re.search(msg_lower)
        if pos_match:
            if self._has_contradiction(msg_lower):
                return (
                    -0.4,
                    "contradicted_positive",
                    msg[:100],
                    "Positive keyword with contradiction (e.g. 'thanks but wrong')",
                )
            return 0.8, "explicit_positive", msg[:100], "Positive keyword, no contradiction"

        # ── Check 4: Refinement ───────────────────────────────────────
        if self._refine_re.search(msg_lower):
            return 0.5, "follow_up_refinement", msg[:100], "Refinement keyword detected"

        # ── Check 5: Same topic continuation ──────────────────────────
        if (
            current_query_type
            and previous_query_type
            and current_query_type == previous_query_type
        ):
            return 0.2, "follow_up_same_topic", None, "Same query type continuation"

        # ── Check 6: Topic change (neutral) ───────────────────────────
        return 0.0, "follow_up_new_topic", None, "Topic change – neutral"

    def _has_contradiction(self, text: str) -> bool:
        """Detect if a positive sentiment is contradicted.

        Looks for patterns like:
          "thanks but that's wrong"
          "good however it doesn't work"
          "nice, yet incorrect"

        Returns True if a contradiction marker appears AND is followed
        by a negative / correction keyword.
        """
        contra_match = self._contradiction_re.search(text)
        if not contra_match:
            return False

        # Text after the contradiction marker
        after = text[contra_match.end():]
        # Check if anything negative follows
        if self._neg_re.search(after) or self._corr_re.search(after):
            return True

        return False

    # ══════════════════════════════════════════════════════════════════
    #  FACTOR 3 – TOOL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════

    def _compute_tool_performance(
        self,
        tool_calls_json: Optional[str],
    ) -> tuple[float, str]:
        """Score tool call success rate and speed.

        Scoring:
          - Base: successes / total_calls  (0.0 – 1.0)
          - Speed bonus: +0.15 if all calls < 2000 ms
          - No tools used → 0.5 baseline (neutral, shouldn't drag score)
        """
        tools = self._parse_tool_calls(tool_calls_json)
        if not tools:
            return 0.5, "No tools used (neutral baseline)"

        successes = sum(1 for t in tools if t.get("success"))
        total = len(tools)
        base = successes / total

        reasons = [f"{successes}/{total} succeeded"]

        # Speed bonus
        times = [t.get("time_ms", 0) for t in tools if t.get("time_ms")]
        if times and all(t < 2000 for t in times):
            base += 0.15
            avg_ms = sum(times) / len(times)
            reasons.append(f"speed bonus (avg {avg_ms:.0f}ms)")

        score = min(base, 1.0)
        return score, "; ".join(reasons)

    # ══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_tool_calls(tool_calls_json: Optional[str]) -> list[dict]:
        """Safely parse tool_calls_json, returning [] on any error."""
        if not tool_calls_json:
            return []
        try:
            data = json.loads(tool_calls_json)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    # Keep old method for backward compatibility if anything calls it
    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """Check if text contains any of the keywords (legacy helper)."""
        return any(keyword in text for keyword in keywords)
