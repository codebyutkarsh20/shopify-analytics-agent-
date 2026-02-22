"""Error recovery pattern learning.

Learns how to recover from errors. When tool A with parameters X fails
and tool B with parameters Y succeeds shortly after, the pattern is stored
so the system can automatically suggest the recovery next time.
"""

import hashlib
import json
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecoveryManager:
    """Learns error recovery patterns.

    Tracks error->recovery mappings to automatically suggest fixes
    when similar errors occur again.
    """

    def __init__(self, db_ops):
        """Initialize RecoveryManager.

        Args:
            db_ops: DatabaseOperations instance
        """
        self.db_ops = db_ops
        self._pending_errors = {}  # {session_key: error_info}
        logger.info("RecoveryManager initialized")

    def record_error(
        self,
        session_key: str,
        tool_name: str,
        tool_params: dict,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record an error for potential recovery learning.

        Stores error details temporarily. If a recovery succeeds shortly
        after, this error and recovery are linked.

        Args:
            session_key: Unique session identifier
            tool_name: Tool that errored
            tool_params: Parameters used with the tool
            error_type: Type of error (e.g., "invalid_parameter")
            error_message: Error message text
        """
        fingerprint = self._compute_fingerprint(error_type, tool_name, tool_params)

        self._pending_errors[session_key] = {
            "tool_name": tool_name,
            "tool_params": tool_params,
            "error_type": error_type,
            "error_message": error_message,
            "fingerprint": fingerprint,
        }

        logger.debug(
            "Error recorded for potential recovery learning",
            session_key=session_key,
            tool_name=tool_name,
            error_type=error_type,
            fingerprint=fingerprint,
        )

    def record_recovery(
        self,
        session_key: str,
        recovery_tool_name: str,
        recovery_params: dict,
    ) -> None:
        """Record successful recovery from error.

        Links recovery to preceding error. Creates or updates recovery pattern
        in database.

        Args:
            session_key: Session that had the error then recovery
            recovery_tool_name: Tool used for recovery
            recovery_params: Parameters for recovery tool
        """
        pending = self._pending_errors.pop(session_key, None)
        if not pending:
            logger.debug(
                "No pending error for recovery",
                session_key=session_key,
            )
            return

        description = self._describe_recovery(
            pending["tool_name"],
            pending["tool_params"],
            recovery_tool_name,
            recovery_params,
        )

        logger.info(
            "Recording error recovery pattern",
            error_type=pending["error_type"],
            fingerprint=pending["fingerprint"],
            description=description,
        )

        existing = self.db_ops.find_recovery_pattern(pending["fingerprint"])
        if existing:
            self.db_ops.increment_recovery_success(existing.id)
        else:
            self.db_ops.create_recovery_pattern(
                error_type=pending["error_type"],
                failed_tool_name=pending["tool_name"],
                failed_parameters=json.dumps(pending["tool_params"]),
                error_fingerprint=pending["fingerprint"],
                recovery_tool_name=recovery_tool_name,
                recovery_parameters=json.dumps(recovery_params),
                recovery_description=description,
            )

    def suggest_recovery(
        self,
        error_type: str,
        tool_name: str,
        tool_params: dict,
    ) -> Optional[dict]:
        """Suggest recovery for an error.

        Looks up recovery patterns by fingerprint or error type.
        Returns recovery only if confidence >= 0.7.

        Args:
            error_type: Type of error
            tool_name: Tool that failed
            tool_params: Parameters used

        Returns:
            Dict with keys: recovery_tool, recovery_params, description, confidence
            or None if no recovery available
        """
        fingerprint = self._compute_fingerprint(error_type, tool_name, tool_params)

        # Try exact fingerprint match first
        pattern = self.db_ops.find_recovery_pattern(fingerprint)

        # Fall back to error type match if exact not found
        if not pattern:
            pattern = self.db_ops.find_recovery_by_type(error_type, tool_name)

        if pattern and pattern.confidence >= 0.7:
            logger.info(
                "Recovery pattern found",
                error_type=error_type,
                fingerprint=fingerprint,
                confidence=pattern.confidence,
            )
            try:
                recovery_params = json.loads(pattern.recovery_parameters) if pattern.recovery_parameters else {}
            except (json.JSONDecodeError, TypeError):
                recovery_params = {}
            return {
                "recovery_tool": pattern.recovery_tool_name,
                "recovery_params": recovery_params,
                "description": pattern.recovery_description,
                "confidence": pattern.confidence,
            }

        logger.debug(
            "No suitable recovery pattern found",
            error_type=error_type,
            fingerprint=fingerprint,
        )
        return None

    def clear_pending(self, session_key: str) -> None:
        """Clear pending error for session.

        Call this if session ends without recovery.

        Args:
            session_key: Session to clear
        """
        if session_key in self._pending_errors:
            self._pending_errors.pop(session_key)
            logger.debug(
                "Cleared pending error",
                session_key=session_key,
            )

    def _compute_fingerprint(
        self,
        error_type: str,
        tool_name: str,
        tool_params: dict,
    ) -> str:
        """Compute a fingerprint for error+tool+params.

        Normalizes parameters for consistent matching while keeping
        fingerprints reasonably specific.

        Args:
            error_type: Type of error
            tool_name: Tool name
            tool_params: Tool parameters

        Returns:
            MD5 hex digest of normalized error info
        """
        normalized = {
            "error_type": error_type,
            "tool": tool_name,
        }

        if tool_name == "shopify_analytics":
            # For analytics, key params are resource and sort_key
            normalized["resource"] = tool_params.get("resource")
            normalized["sort_key"] = tool_params.get("sort_key")
        elif tool_name == "shopify_graphql":
            # For GraphQL, capture query start
            query = tool_params.get("query", "")
            
            # Make the query fuzzy to match queries despite different generated IDs or variable string inputs
            import re
            query_clean = re.sub(r'\s+', ' ', query).strip()
            query_clean = re.sub(r'gid://shopify/\w+/\d+', 'ID', query_clean)
            query_clean = re.sub(r'["\'][^"\']*["\']', 'STRING', query_clean)
            
            normalized["query_start"] = query_clean[:50]

        fingerprint_str = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def _describe_recovery(
        self,
        failed_tool: str,
        failed_params: dict,
        recovery_tool: str,
        recovery_params: dict,
    ) -> str:
        """Generate description of recovery changes.

        Args:
            failed_tool: Tool that failed
            failed_params: Failed tool's parameters
            recovery_tool: Tool used for recovery
            recovery_params: Recovery tool's parameters

        Returns:
            Human-readable description of changes
        """
        changes = []

        # Tool change
        if failed_tool != recovery_tool:
            changes.append(f"Switched from {failed_tool} to {recovery_tool}")

        # Parameter changes (for analytics)
        if failed_tool == recovery_tool == "shopify_analytics":
            for key in ["sort_key", "resource", "query"]:
                if failed_params.get(key) != recovery_params.get(key):
                    changes.append(
                        f"Changed {key} from "
                        f"'{failed_params.get(key)}' to "
                        f"'{recovery_params.get(key)}'"
                    )

        return "; ".join(changes) if changes else "Modified query parameters"
