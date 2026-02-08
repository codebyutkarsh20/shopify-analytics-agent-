"""Query template learning and management.

Learns successful query patterns and stores them as reusable templates.
When similar queries are encountered, templates provide known-good parameters
instead of constructing from scratch.
"""

import json
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemplateManager:
    """Manages query templates for successful patterns.

    Tracks which tool+parameter combinations work well for which intents.
    Uses success/failure counts and execution time to rank templates.
    """

    def __init__(self, db_ops):
        """Initialize TemplateManager.

        Args:
            db_ops: DatabaseOperations instance
        """
        self.db_ops = db_ops
        logger.info("TemplateManager initialized")

    def record_successful_query(
        self,
        user_id: int,
        user_message: str,
        tool_name: str,
        tool_params: dict,
        execution_time_ms: int,
        intent_category: str,
    ) -> None:
        """Record a successful query for learning.

        Finds or creates a template for this intent+tool combination,
        increments success count, and adds the user message as an example.

        Args:
            user_id: User who executed the query
            user_message: Original user message (for examples)
            tool_name: Which tool was used (e.g., "shopify_analytics")
            tool_params: Tool parameters dict
            execution_time_ms: How long execution took in milliseconds
            intent_category: Intent category (e.g., "products_ranking")
        """
        # Step 1: Find existing template
        existing = self.db_ops.find_template(
            intent_category=intent_category,
            tool_name=tool_name,
        )

        if existing:
            # Step 2a: Increment success
            logger.debug(
                "Incrementing template success",
                template_id=existing.id,
                intent_category=intent_category,
                tool_name=tool_name,
            )
            self.db_ops.increment_template_success(
                template_id=existing.id,
                execution_time_ms=execution_time_ms,
            )

            # Step 2b: Add example
            self.db_ops.add_template_example(
                template_id=existing.id,
                example_query=user_message[:200],
            )
        else:
            # Step 3: Create new template
            description = self._generate_description(user_message, tool_name)
            logger.info(
                "Creating new template",
                intent_category=intent_category,
                tool_name=tool_name,
                description=description,
            )
            self.db_ops.create_template(
                intent_category=intent_category,
                intent_description=description,
                tool_name=tool_name,
                tool_parameters=json.dumps(tool_params),
                created_by_user_id=user_id,
                example_queries=json.dumps([user_message[:200]]),
                confidence=0.8,
            )

    def record_failed_query(
        self,
        intent_category: str,
        tool_name: str,
    ) -> None:
        """Record a failed query for learning.

        Finds template and increments failure count, lowering confidence.

        Args:
            intent_category: Intent category
            tool_name: Tool that failed
        """
        existing = self.db_ops.find_template(
            intent_category=intent_category,
            tool_name=tool_name,
        )

        if existing:
            logger.debug(
                "Incrementing template failure",
                template_id=existing.id,
                intent_category=intent_category,
                tool_name=tool_name,
            )
            self.db_ops.increment_template_failure(
                template_id=existing.id,
            )

    def get_template_for_intent(
        self,
        intent_category: str,
    ) -> Optional[dict]:
        """Get the best template for an intent.

        Returns highest-confidence template if confidence >= 0.7.

        Args:
            intent_category: The intent to find a template for

        Returns:
            Dict with keys: tool_name, parameters, confidence, description
            or None if no suitable template found
        """
        template = self.db_ops.get_best_template(
            intent_category=intent_category,
            min_confidence=0.7,
        )

        if template:
            logger.debug(
                "Found template for intent",
                intent_category=intent_category,
                template_id=template.id,
                confidence=template.confidence,
            )
            try:
                parameters = json.loads(template.tool_parameters) if template.tool_parameters else {}
            except (json.JSONDecodeError, TypeError):
                parameters = {}
            return {
                "template_id": template.id,
                "tool_name": template.tool_name,
                "parameters": parameters,
                "confidence": template.confidence,
                "description": template.intent_description,
            }

        logger.debug(
            "No suitable template found",
            intent_category=intent_category,
        )
        return None

    def update_template_quality(
        self,
        template_id: int,
        quality_score: float,
    ) -> None:
        """Update template confidence based on quality feedback.

        Positive scores boost confidence (up to 1.0).
        Negative scores penalize confidence (down to 0.1).

        Args:
            template_id: Template to update
            quality_score: Quality score (-1.0 to 1.0)
        """
        from sqlalchemy import select
        from src.database.models import QueryTemplate

        session = self.db_ops.get_session()
        try:
            # Get current template
            stmt = select(QueryTemplate).where(QueryTemplate.id == template_id)
            template = session.execute(stmt).scalar_one_or_none()

            if not template:
                logger.warning(
                    "Template not found for quality update",
                    template_id=template_id,
                )
                return

            old_confidence = template.confidence or 0.5

            if quality_score > 0:
                # Slight boost: confidence += quality_score * 0.02, capped at 1.0
                delta = quality_score * 0.02
                new_confidence = min(old_confidence + delta, 1.0)
            else:
                # Slight penalty: confidence += quality_score * 0.05, floored at 0.1
                delta = quality_score * 0.05
                new_confidence = max(old_confidence + delta, 0.1)

            template.confidence = new_confidence
            session.commit()

            logger.debug(
                "Updated template confidence",
                template_id=template_id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                quality_score=quality_score,
            )
        except Exception as e:
            logger.error(
                "Failed to update template confidence",
                template_id=template_id,
                error=str(e),
            )
            session.rollback()
        finally:
            session.close()

    def _generate_description(
        self,
        user_message: str,
        tool_name: str,
    ) -> str:
        """Generate a human-readable template description.

        Args:
            user_message: Example user message
            tool_name: Tool being used

        Returns:
            Description string
        """
        return f"Query using {tool_name}: {user_message[:100]}"
