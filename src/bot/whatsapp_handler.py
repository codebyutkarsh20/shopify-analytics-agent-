"""WhatsApp webhook handler and message processing.

Provides an aiohttp-based HTTP server that receives inbound WhatsApp
messages via webhook (Meta Cloud API or Twilio), processes them through
the same agent pipeline used by Telegram, and replies via the WhatsApp
Business API.

Architecture:
    Webhook POST → parse payload → resolve/create user → run agent pipeline
        → send reply (text + optional chart images) back via WhatsApp API.
"""

import asyncio
import hmac
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import web

from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.learning.pattern_learner import PatternLearner
from src.learning.preference_manager import PreferenceManager
from src.learning.session_manager import SessionManager
from src.learning.template_manager import TemplateManager
from src.learning.feedback_analyzer import FeedbackAnalyzer
from src.learning.insight_aggregator import InsightAggregator
from src.bot.whatsapp_adapter import WhatsAppAdapter
from src.utils.logger import get_logger, new_correlation_id
from src.utils.rate_limiter import RateLimiter
from src.utils.formatters import format_error_message

logger = get_logger(__name__)

# WhatsApp message length limit
WHATSAPP_MESSAGE_LIMIT = 4096


# ---------------------------------------------------------------------------
# WhatsApp API clients (Meta Cloud API + Twilio)
# ---------------------------------------------------------------------------

class MetaWhatsAppClient:
    """Send messages via Meta Cloud API (v18.0+)."""

    BASE_URL = "https://graph.facebook.com/v18.0"

    def __init__(self, access_token: str, phone_number_id: str):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def send_text(self, to: str, text: str) -> Dict[str, Any]:
        """Send a text message to a WhatsApp number."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error("Meta API send_text failed", status=resp.status, body=data)
            return data

    async def send_image(self, to: str, image_path: str, caption: str = "") -> Dict[str, Any]:
        """Send an image message to a WhatsApp number (upload as media)."""
        session = await self._get_session()

        # Step 1: Upload media
        upload_url = f"{self.BASE_URL}/{self.phone_number_id}/media"
        form = aiohttp.FormData()
        form.add_field("messaging_product", "whatsapp")
        form.add_field("type", "image/png")
        form.add_field(
            "file",
            open(image_path, "rb"),
            filename=os.path.basename(image_path),
            content_type="image/png",
        )
        # Need a fresh session without JSON content-type for multipart
        upload_session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        try:
            async with upload_session.post(upload_url, data=form) as resp:
                upload_data = await resp.json()
                if resp.status != 200:
                    logger.error("Media upload failed", status=resp.status, body=upload_data)
                    return upload_data
                media_id = upload_data.get("id")
        finally:
            await upload_session.close()

        # Step 2: Send image message using media_id
        msg_url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "image",
            "image": {"id": media_id},
        }
        if caption:
            payload["image"]["caption"] = caption[:1024]  # WhatsApp caption limit

        async with session.post(msg_url, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error("Meta API send_image failed", status=resp.status, body=data)
            return data

    async def send_interactive_buttons(
        self, to: str, body_text: str, buttons: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Send an interactive button message via Meta Cloud API.

        WhatsApp supports up to 3 quick-reply buttons per message.

        Args:
            to: Recipient phone number.
            body_text: Message body text shown above the buttons.
            buttons: List of dicts with ``id`` and ``title`` keys.
                     Example: [{"id": "fb_up_42", "title": "👍"}, ...]

        Returns:
            API response dict.
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body_text},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {"id": btn["id"], "title": btn["title"]},
                        }
                        for btn in buttons[:3]  # WhatsApp max 3 buttons
                    ]
                },
            },
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error("Meta API send_interactive_buttons failed", status=resp.status, body=data)
            return data

    async def send_typing_indicator(self, to: str) -> None:
        """Send a typing indicator ('typing...' bubble) to a WhatsApp user.

        Uses the Meta Cloud API messages endpoint with status=typing.
        Silently ignores errors so it never blocks message processing.
        """
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "status": "typing",
            }
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    # Non-critical – log and move on
                    data = await resp.json()
                    logger.debug(
                        "Typing indicator failed (non-critical)",
                        status=resp.status,
                        body=data,
                    )
        except Exception as exc:
            logger.debug("Typing indicator error (ignored)", error=str(exc))

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class TwilioWhatsAppClient:
    """Send messages via Twilio WhatsApp API."""

    BASE_URL = "https://api.twilio.com/2010-04-01"

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number  # e.g., "whatsapp:+14155238886"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            self._session = aiohttp.ClientSession(auth=auth)
        return self._session

    async def send_text(self, to: str, text: str) -> Dict[str, Any]:
        """Send a text message via Twilio."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/Accounts/{self.account_sid}/Messages.json"
        to_number = f"whatsapp:+{to}" if not to.startswith("whatsapp:") else to
        payload = {
            "From": self.from_number,
            "To": to_number,
            "Body": text,
        }
        async with session.post(url, data=payload) as resp:
            data = await resp.json()
            if resp.status not in (200, 201):
                logger.error("Twilio send_text failed", status=resp.status, body=data)
            return data

    async def send_image(self, to: str, image_path: str, caption: str = "") -> Dict[str, Any]:
        """Send an image via Twilio (requires publicly accessible URL or media upload).

        Note: Twilio requires a publicly accessible URL for MediaUrl.
        For local files, you would need to host them temporarily.
        This implementation sends the caption as text + a note about the chart.
        """
        # For a production setup, you'd upload the image to a public URL first.
        # For now, send caption as text with a note.
        logger.warning("Twilio image sending requires public URL hosting — sending text fallback")
        fallback_text = caption or "📊 Chart generated (image not available in Twilio mode)"
        return await self.send_text(to, fallback_text)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Webhook payload parsers
# ---------------------------------------------------------------------------

def parse_meta_webhook(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a Meta Cloud API webhook payload into a normalized event.

    Returns None if the payload doesn't contain a text message.
    """
    try:
        entry = data.get("entry", [])
        if not entry:
            return None

        changes = entry[0].get("changes", [])
        if not changes:
            return None

        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None

        msg = messages[0]
        msg_type = msg.get("type")

        # Handle interactive button replies (e.g. feedback 👍/👎)
        if msg_type == "interactive":
            interactive = msg.get("interactive", {})
            if interactive.get("type") == "button_reply":
                button_reply = interactive.get("button_reply", {})
                from_number = msg.get("from", "")
                contacts = value.get("contacts", [])
                profile_name = ""
                if contacts:
                    profile = contacts[0].get("profile", {})
                    profile_name = profile.get("name", "")
                return {
                    "from_number": from_number,
                    "message_text": button_reply.get("title", ""),
                    "profile_name": profile_name,
                    "message_id": msg.get("id", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "button_reply_id": button_reply.get("id", ""),
                }
            return None

        if msg_type != "text":
            logger.debug("Ignoring non-text WhatsApp message", msg_type=msg_type)
            return None

        from_number = msg.get("from", "")
        text = msg.get("text", {}).get("body", "")

        # Extract profile name from contacts
        contacts = value.get("contacts", [])
        profile_name = ""
        if contacts:
            profile = contacts[0].get("profile", {})
            profile_name = profile.get("name", "")

        return {
            "from_number": from_number,
            "message_text": text,
            "profile_name": profile_name,
            "message_id": msg.get("id", ""),
            "timestamp": msg.get("timestamp", ""),
        }

    except (KeyError, IndexError) as e:
        logger.warning("Failed to parse Meta webhook", error=str(e))
        return None


def parse_twilio_webhook(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a Twilio webhook (form data) into a normalized event."""
    try:
        from_number = data.get("From", "")
        # Twilio sends as "whatsapp:+1234567890" — strip prefix
        from_number = from_number.replace("whatsapp:", "").lstrip("+")

        text = data.get("Body", "")
        if not text:
            return None

        profile_name = data.get("ProfileName", "")

        return {
            "from_number": from_number,
            "message_text": text,
            "profile_name": profile_name,
            "message_id": data.get("MessageSid", ""),
            "timestamp": "",
        }

    except (KeyError, IndexError) as e:
        logger.warning("Failed to parse Twilio webhook", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Chart marker handling (shared with Telegram handler)
# ---------------------------------------------------------------------------

_CHART_MARKER_RE = re.compile(r"\[CHART:(\d+)\]")


def build_interleaved_segments(
    message: str,
    chart_files: List[str],
) -> List[Union[str, Tuple[str, str]]]:
    """Split response into interleaved text and chart segments.

    Same logic as MessageHandler._build_interleaved_segments but standalone.
    """
    if not chart_files:
        return [message] if message.strip() else []

    segments: List[Union[str, Tuple[str, str]]] = []
    used_indices: set = set()
    parts = _CHART_MARKER_RE.split(message)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                segments.append(text)
        else:
            chart_idx = int(part)
            if chart_idx < len(chart_files):
                segments.append(("chart", chart_files[chart_idx]))
                used_indices.add(chart_idx)

    # Append unreferenced charts
    for idx, filepath in enumerate(chart_files):
        if idx not in used_indices:
            segments.append(("chart", filepath))

    return segments


# ---------------------------------------------------------------------------
# WhatsApp message handler (webhook server)
# ---------------------------------------------------------------------------

class WhatsAppWebhookHandler:
    """Handles inbound WhatsApp webhooks and processes messages through the agent.

    This class:
    1. Runs an aiohttp web server to receive webhooks
    2. Parses Meta/Twilio payloads
    3. Resolves users via ChannelSession
    4. Processes messages through the LLM pipeline
    5. Sends responses back via WhatsApp API
    """

    def __init__(
        self,
        llm_service,
        pattern_learner: PatternLearner,
        preference_manager: PreferenceManager,
        db_ops: DatabaseOperations,
        session_manager: SessionManager = None,
        feedback_analyzer: FeedbackAnalyzer = None,
        template_manager: TemplateManager = None,
        insight_aggregator: InsightAggregator = None,
        channel_linker=None,
    ):
        self.llm_service = llm_service
        self.pattern_learner = pattern_learner
        self.preference_manager = preference_manager
        self.db_ops = db_ops
        self.session_manager = session_manager
        self.feedback_analyzer = feedback_analyzer
        self.template_manager = template_manager
        self.insight_aggregator = insight_aggregator
        self.channel_linker = channel_linker

        self.adapter = WhatsAppAdapter()
        self._rate_limiter = RateLimiter(
            max_requests=settings.security.rate_limit_per_minute,
            window_seconds=60,
        )
        self._bot_access_code = settings.security.bot_access_code

        # Interaction counter for periodic aggregation
        self._interaction_count = 0
        self._aggregation_interval = getattr(settings, "aggregation_interval", 50)

        # Initialize the API client based on provider
        self._client = self._create_client()

        self.app = web.Application()
        self.app.router.add_get("/webhook", self._handle_verification)
        self.app.router.add_post("/webhook", self._handle_incoming)
        self.app.router.add_get("/health", self._handle_health)

        logger.info(
            "WhatsApp webhook handler initialized",
            provider=settings.whatsapp.provider,
            port=settings.whatsapp.webhook_port,
        )

    def _create_client(self):
        """Create the appropriate WhatsApp API client."""
        if settings.whatsapp.provider == "twilio":
            return TwilioWhatsAppClient(
                account_sid=settings.whatsapp.twilio_account_sid,
                auth_token=settings.whatsapp.twilio_auth_token,
                from_number=settings.whatsapp.twilio_whatsapp_number,
            )
        else:
            return MetaWhatsAppClient(
                access_token=settings.whatsapp.access_token,
                phone_number_id=settings.whatsapp.phone_number_id,
            )

    # ── HTTP endpoint handlers ──────────────────────────────────────

    async def _handle_verification(self, request: web.Request) -> web.Response:
        """Handle Meta webhook verification (GET request).

        Meta sends: GET /webhook?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...
        We must return the challenge if the verify_token matches.
        """
        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge")

        if mode == "subscribe" and token == settings.whatsapp.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return web.Response(text=challenge, content_type="text/plain")

        logger.warning("WhatsApp webhook verification failed", mode=mode)
        return web.Response(status=403, text="Verification failed")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "channel": "whatsapp"})

    def _verify_meta_signature(self, request: web.Request, body: bytes) -> bool:
        """Verify the X-Hub-Signature-256 header from Meta webhooks.

        Meta signs every webhook POST with HMAC-SHA256 using the App Secret.
        If the app_secret is not configured, signature verification is skipped
        (development mode).

        Args:
            request: The incoming aiohttp request.
            body: The raw request body bytes.

        Returns:
            True if the signature is valid or verification is skipped.
        """
        app_secret = os.getenv("META_APP_SECRET", "")
        if not app_secret:
            # No app secret configured — skip verification (dev mode)
            return True

        signature_header = request.headers.get("X-Hub-Signature-256", "")
        if not signature_header.startswith("sha256="):
            logger.warning("Missing or malformed X-Hub-Signature-256 header")
            return False

        expected_sig = signature_header[7:]  # strip "sha256=" prefix
        computed_sig = hmac.new(
            app_secret.encode(), body, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_sig, computed_sig):
            logger.warning("WhatsApp webhook signature verification failed")
            return False

        return True

    async def _handle_incoming(self, request: web.Request) -> web.Response:
        """Handle inbound webhook POST with a WhatsApp message.

        Returns 200 immediately (WhatsApp expects fast acknowledgement)
        and processes the message asynchronously.
        """
        try:
            # Read raw body for signature verification
            body = await request.read()

            # Verify Meta HMAC signature (if app secret is configured)
            if settings.whatsapp.provider != "twilio":
                if not self._verify_meta_signature(request, body):
                    logger.warning("Rejected webhook: invalid signature")
                    return web.Response(status=403, text="Invalid signature")

            # Parse based on provider
            if settings.whatsapp.provider == "twilio":
                # Twilio sends form-encoded data
                data = dict(await request.post())
                event = parse_twilio_webhook(data)
            else:
                # Meta sends JSON
                import json as _json
                data = _json.loads(body)
                event = parse_meta_webhook(data)

            if event is None:
                # Not a text message or status update — acknowledge silently
                return web.Response(status=200, text="OK")

            logger.info(
                "WhatsApp message received",
                from_number=event["from_number"],
                message_length=len(event["message_text"]),
            )

            # Process asynchronously so we return 200 immediately
            asyncio.create_task(self._process_message(event))

            return web.Response(status=200, text="OK")

        except Exception as e:
            logger.error("Error handling WhatsApp webhook", error=str(e), exc_info=True)
            # Still return 200 to prevent webhook retries
            return web.Response(status=200, text="OK")

    # ── Message processing pipeline ────────────────────────────────

    async def _process_message(self, event: Dict[str, Any]) -> None:
        """Process an inbound WhatsApp message through the agent pipeline.

        Mirrors the 10-step flow in handlers.py MessageHandler.handle_message
        but adapted for WhatsApp's stateless webhook model.
        """
        # Generate a correlation ID for this request so all log lines can be traced
        cid = new_correlation_id()

        from_number = event["from_number"]
        message_text = event["message_text"]

        try:
            # Step 0: Rate limiting
            allowed, retry_after = self._rate_limiter.is_allowed(from_number)
            if not allowed:
                await self._send_text(
                    from_number,
                    f"⏳ Too many requests. Please wait {retry_after}s before trying again.",
                )
                return

            # Step 1: Resolve or create user via ChannelSession
            user = self._resolve_whatsapp_user(event)
            logger.debug("WhatsApp user resolved", user_id=user.id, phone=from_number)

            # Check if this is a feedback button reply (👍/👎)
            feedback = self._handle_feedback_button(event)
            if feedback:
                await self._process_feedback(user, from_number, feedback)
                return

            # Update user activity
            self.db_ops.update_user_activity(user.id)
            try:
                self.db_ops.increment_user_interaction(user.id)
            except Exception:
                pass

            # Step 1.5: Access-code verification
            if self._bot_access_code and not user.is_verified:
                code_attempt = message_text.strip()
                if hmac.compare_digest(code_attempt, self._bot_access_code):
                    self.db_ops.verify_user(user.id)
                    logger.info("WhatsApp user verified via access code", user_id=user.id)
                    await self._send_text(
                        from_number,
                        "✅ *Access Granted!*\n\nYou're now verified. Welcome aboard!\nJust ask me a question to get started.",
                    )
                    return
                else:
                    await self._send_text(
                        from_number,
                        "🔒 *Verification Required*\n\nPlease enter the access code to use this bot.\nIf you don't have a code, contact the administrator.",
                    )
                    return

            # Step 2: Check store connection (auto-connect from env)
            self._auto_connect_store(user.id)
            store = self.db_ops.get_store_by_user(user.id)
            if not store:
                await self._send_text(
                    from_number,
                    "⚠️ *No Shopify Store Connected*\n\nI need your Shopify store credentials to help with analytics.\n\n"
                    "Send your credentials in this format:\n"
                    "`connect domain:token`\n\n"
                    "Example: `connect myshop.myshopify.com:shpat_abc123xyz`",
                )
                return

            # Handle inline /connect command
            if message_text.strip().lower().startswith("connect "):
                await self._handle_connect(user, from_number, message_text)
                return

            # Handle cross-channel linking commands
            msg_lower = message_text.strip().lower()
            if msg_lower == "link" or msg_lower.startswith("link "):
                await self._handle_link(user, from_number, message_text)
                return

            # Step 3: Classify intent
            intent = self.pattern_learner.classify_intent(message_text)

            if intent.coarse == "general" or self.pattern_learner.assess_query_complexity(message_text) == "complex":
                try:
                    refined = await self.pattern_learner.refine_intent_with_llm(
                        message_text, self.llm_service
                    )
                    if refined:
                        intent = refined
                except Exception:
                    pass

            query_type = intent.coarse

            # Step 4: Session management
            session = None
            session_id = None
            if self.session_manager:
                session = self.session_manager.get_or_create_session(
                    user_id=user.id,
                    channel_type="whatsapp",
                    current_intent=intent.coarse,
                )
                session_id = session.id

            # Step 5: Feedback analysis
            if self.feedback_analyzer:
                self._analyze_feedback(user.id, message_text, query_type)

            # Step 5.5: Show typing indicator while processing
            if hasattr(self._client, "send_typing_indicator"):
                await self._client.send_typing_indicator(from_number)

            # Step 6: Process through LLM
            response = await self.llm_service.process_message(
                user_id=user.id,
                message=message_text,
                intent=intent,
                session_id=session_id,
            )

            if not response:
                response = "Sorry, I couldn't process that query. Please try again."

            # Step 7: Learn from query
            self.pattern_learner.learn_from_query(
                user_id=user.id,
                query=message_text,
                query_type=query_type,
            )
            self.preference_manager.update_preferences_from_patterns(user.id)

            # Step 8: Format and send response (with charts)
            chart_files = getattr(self.llm_service, "last_chart_files", [])
            await self._send_response(from_number, response, chart_files)

            # Step 9: Save conversation
            tool_calls_json = self.llm_service.last_tool_calls_json
            conversation = self.db_ops.save_conversation(
                user_id=user.id,
                message=message_text,
                response=response,
                query_type=query_type,
                session_id=session_id,
                channel_type="whatsapp",
                tool_calls_json=tool_calls_json,
            )

            # Step 9.5: Send feedback buttons (👍/👎) — only for analytical responses
            # Skip for general/utility responses (greetings, errors, no-store prompts)
            has_tool_calls = any(
                tc.get("success") for tc in getattr(self.llm_service, "last_tool_calls", [])
            )
            has_charts = bool(chart_files)
            if has_tool_calls or has_charts:
                await self._send_feedback_buttons(from_number, conversation.id)

            logger.info(
                "WhatsApp message processed",
                user_id=user.id,
                query_type=query_type,
                session_id=session_id,
            )

            # Step 10: Periodic aggregation
            self._check_aggregation()

        except Exception as e:
            logger.error(
                "Error processing WhatsApp message",
                from_number=from_number,
                error=str(e),
                exc_info=True,
            )
            try:
                await self._send_text(
                    from_number,
                    "⚠️ Something went wrong while processing your request. Please try again.",
                )
            except Exception:
                pass

    # ── User resolution ────────────────────────────────────────────

    def _resolve_whatsapp_user(self, event: Dict[str, Any]):
        """Resolve a WhatsApp user to an internal User via ChannelSession.

        If the phone number has been seen before, returns the linked user.
        Otherwise, creates a new User + ChannelSession.
        """
        from_number = event["from_number"]
        profile_name = event.get("profile_name", "")

        # Try to find existing channel session
        channel_session = self.db_ops.get_channel_session(
            channel_type="whatsapp",
            channel_user_id=from_number,
        )

        if channel_session:
            # Update last_active
            self.db_ops.update_channel_session_activity(channel_session.id)
            user = self.db_ops.get_user_by_id(channel_session.user_id)
            if user:
                return user

        # Create new user for WhatsApp
        user = self.db_ops.create_whatsapp_user(
            whatsapp_phone=from_number,
            display_name=profile_name or from_number,
            first_name=profile_name.split()[0] if profile_name else "",
        )

        # Create channel session linking
        self.db_ops.create_channel_session(
            user_id=user.id,
            channel_type="whatsapp",
            channel_user_id=from_number,
            channel_username=profile_name,
        )

        logger.info(
            "New WhatsApp user created",
            user_id=user.id,
            phone=from_number,
            name=profile_name,
        )

        return user

    def _auto_connect_store(self, user_id: int) -> bool:
        """Auto-connect store from .env if user has no store."""
        store = self.db_ops.get_store_by_user(user_id)
        if store:
            return False

        domain = settings.shopify.shop_domain
        token = settings.shopify.access_token
        if domain and token:
            self.db_ops.add_store(
                user_id=user_id,
                shop_domain=domain,
                access_token=token,
            )
            logger.info("Auto-connected store for WhatsApp user", user_id=user_id)
            return True
        return False

    async def _handle_connect(self, user, from_number: str, message_text: str) -> None:
        """Handle inline 'connect domain:token' command from WhatsApp."""
        try:
            # Extract credentials: "connect domain:token"
            creds = message_text.strip()[8:].strip()  # Remove "connect "
            if ":" not in creds:
                await self._send_text(
                    from_number,
                    "❌ Invalid format. Use: `connect domain:token`\n"
                    "Example: `connect myshop.myshopify.com:shpat_abc123`",
                )
                return

            domain, token = creds.split(":", 1)
            domain = domain.strip()
            token = token.strip()

            if not domain or not token:
                await self._send_text(from_number, "❌ Domain and token cannot be empty.")
                return

            if not re.match(r"^[\w\-]+\.myshopify\.com$", domain):
                await self._send_text(
                    from_number,
                    "❌ Invalid domain format. Expected: `myshop.myshopify.com`",
                )
                return

            self.db_ops.add_store(
                user_id=user.id,
                shop_domain=domain,
                access_token=token,
            )

            await self._send_text(
                from_number,
                f"✅ *Store Connected!*\n\nConnected to *{domain}*.\n\n"
                "You can now ask me questions about your analytics:\n"
                '• "Show me sales for last 7 days"\n'
                '• "Top 5 products by revenue"\n'
                '• "Compare this week to last week"',
            )

        except Exception as e:
            logger.error("Error in WhatsApp connect flow", error=str(e))
            await self._send_text(from_number, "❌ Failed to connect store. Please try again.")

    async def _handle_link(self, user, from_number: str, message_text: str) -> None:
        """Handle cross-channel linking commands from WhatsApp.

        - "link"         → Generate a code to use on Telegram
        - "link <code>"  → Redeem a code generated on Telegram
        """
        if not self.channel_linker:
            await self._send_text(
                from_number,
                "ℹ️ Cross-channel linking is not available.",
            )
            return

        # Check if already linked
        already_msg = self.channel_linker.check_already_linked(user.id)
        if already_msg:
            await self._send_text(from_number, f"✅ {already_msg}")
            return

        parts = message_text.strip().split(maxsplit=1)

        if len(parts) > 1:
            # Redeeming a code: "link 123456"
            code = parts[1].strip()
            success, message = self.channel_linker.redeem_link_code(
                code=code,
                target_channel="whatsapp",
                target_user_id=user.id,
                target_channel_id=from_number,
            )
            emoji = "✅" if success else "❌"
            await self._send_text(from_number, f"{emoji} {message}")
        else:
            # Generating a code: "link"
            code = self.channel_linker.generate_link_code(
                user_id=user.id,
                source_channel="whatsapp",
                source_channel_id=from_number,
            )
            await self._send_text(
                from_number,
                f"🔗 *Account Linking Code*\n\n"
                f"Your linking code is: `{code}`\n\n"
                f"Send this on Telegram within 10 minutes:\n"
                f"`/link {code}`\n\n"
                f"This will merge your WhatsApp and Telegram accounts "
                f"so you share the same history, preferences, and store connections.",
            )

    # ── Explicit feedback (👍 / 👎 buttons) ─────────────────────

    async def _send_feedback_buttons(self, to: str, conversation_id: int) -> None:
        """Send 👍/👎 interactive buttons after a bot response.

        Uses WhatsApp interactive button messages if the client supports it,
        otherwise falls back to a simple text prompt.
        """
        try:
            if hasattr(self._client, "send_interactive_buttons"):
                await self._client.send_interactive_buttons(
                    to=to,
                    body_text="Was this helpful?",
                    buttons=[
                        {"id": f"fb_up_{conversation_id}", "title": "👍 Yes"},
                        {"id": f"fb_down_{conversation_id}", "title": "👎 No"},
                    ],
                )
            else:
                # Twilio fallback — simple text
                await self._send_text(
                    to,
                    f"_Was this helpful? Reply *yes* or *no*_",
                )
        except Exception as e:
            logger.debug("Failed to send feedback buttons", error=str(e))

    def _handle_feedback_button(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if an incoming message is a feedback button reply.

        WhatsApp interactive button replies arrive with type=interactive
        and contain the button ID we set (e.g., fb_up_42).

        Returns:
            Dict with ``direction`` ("up"/"down") and ``conversation_id``,
            or None if not a feedback reply.
        """
        button_id = event.get("button_reply_id", "")
        if not button_id.startswith("fb_"):
            return None

        try:
            parts = button_id.split("_", 2)  # ["fb", "up"/"down", "conv_id"]
            return {
                "direction": parts[1],
                "conversation_id": int(parts[2]),
            }
        except (IndexError, ValueError):
            return None

    async def _process_feedback(self, user, from_number: str, feedback: Dict[str, Any]) -> None:
        """Process an explicit 👍/👎 feedback reply.

        Updates the same tables as the implicit FeedbackAnalyzer so the
        learning pipeline benefits from both implicit and explicit signals.
        """
        is_positive = feedback["direction"] == "up"
        conversation_id = feedback["conversation_id"]
        quality_score = 1.0 if is_positive else -1.0
        feedback_type = "explicit_thumbs_up" if is_positive else "explicit_thumbs_down"

        try:
            # 1. Save feedback record
            self.db_ops.save_response_feedback(
                conversation_id=conversation_id,
                user_id=user.id,
                feedback_type=feedback_type,
                quality_score=quality_score,
                signal_text="👍" if is_positive else "👎",
            )

            # 2. Update conversation quality
            self.db_ops.update_conversation_quality(
                conversation_id=conversation_id,
                quality_score=quality_score,
            )

            # 3. Adjust template confidence if a template was used
            if self.template_manager:
                convs = self.db_ops.get_latest_conversation(user.id, limit=5)
                for conv in convs:
                    if conv.id == conversation_id and conv.template_id_used:
                        self.template_manager.update_template_quality(
                            template_id=conv.template_id_used,
                            quality_score=quality_score,
                        )
                        break

            # 4. Acknowledge
            emoji = "👍" if is_positive else "👎"
            await self._send_text(from_number, f"Thanks for the feedback! {emoji}")

            logger.info(
                "WhatsApp explicit feedback received",
                user_id=user.id,
                conversation_id=conversation_id,
                feedback=feedback["direction"],
            )
        except Exception as e:
            logger.error("Error processing WhatsApp feedback", error=str(e))

    # ── Response sending ───────────────────────────────────────────

    async def _send_text(self, to: str, text: str) -> None:
        """Send a text message, splitting if over the limit."""
        formatted = self.adapter.format_response(text)
        limit = self.adapter.get_message_limit()

        # Split into chunks if needed
        chunks = self._split_message(formatted, limit)
        for chunk in chunks:
            try:
                await self._client.send_text(to, chunk)
            except Exception as e:
                logger.error("Failed to send WhatsApp text", to=to, error=str(e))

    def _split_message(self, text: str, limit: int) -> List[str]:
        """Split a long message into chunks respecting the limit."""
        if len(text) <= limit:
            return [text]

        chunks = []
        current = ""
        for paragraph in text.split("\n\n"):
            if len(current) + len(paragraph) + 2 > limit:
                if current:
                    chunks.append(current)
                # If single paragraph exceeds limit, hard-split
                if len(paragraph) > limit:
                    while paragraph:
                        chunks.append(paragraph[:limit])
                        paragraph = paragraph[limit:]
                    current = ""
                else:
                    current = paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current:
            chunks.append(current)

        return chunks

    async def _send_chart(self, to: str, filepath: str) -> None:
        """Send a chart image and clean up the temp file."""
        try:
            if os.path.exists(filepath):
                await self._client.send_image(to, filepath, caption="📊 Analytics Chart")
                logger.info("WhatsApp chart sent", to=to, chart_file=filepath)
            else:
                logger.warning("Chart file not found", chart_file=filepath)
        except Exception as e:
            logger.error("Failed to send WhatsApp chart", error=str(e))
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

    async def _send_response(
        self, to: str, message: str, chart_files: List[str] = None
    ) -> None:
        """Send a response with interleaved text and charts."""
        if not message:
            return

        segments = build_interleaved_segments(message, chart_files or [])

        for segment in segments:
            if isinstance(segment, tuple) and segment[0] == "chart":
                await self._send_chart(to, segment[1])
            elif isinstance(segment, str):
                await self._send_text(to, segment)

    # ── Feedback analysis ──────────────────────────────────────────

    def _analyze_feedback(self, user_id: int, current_message: str, current_query_type: str) -> None:
        """Analyze implicit feedback from this message (mirrors handlers.py logic)."""
        try:
            latest_convs = self.db_ops.get_latest_conversation(user_id)
            if not latest_convs:
                return

            latest_conv = latest_convs[0]
            previous_query_type = latest_conv.query_type or ""

            feedback = self.feedback_analyzer.analyze_follow_up(
                previous_query_type=previous_query_type,
                current_message=current_message,
                current_query_type=current_query_type,
            )

            if feedback and feedback.get("quality_score", 0) != 0:
                self.db_ops.save_response_feedback(
                    conversation_id=latest_conv.id,
                    user_id=user_id,
                    feedback_type=feedback["feedback_type"],
                    quality_score=feedback["quality_score"],
                    signal_text=feedback.get("signal_text"),
                )
                self.db_ops.update_conversation_quality(
                    conversation_id=latest_conv.id,
                    quality_score=feedback["quality_score"],
                )
                if self.template_manager and latest_conv.template_id_used:
                    self.template_manager.update_template_quality(
                        template_id=latest_conv.template_id_used,
                        quality_score=feedback["quality_score"],
                    )

        except Exception as e:
            logger.warning("WhatsApp feedback analysis failed", error=str(e))

    def _check_aggregation(self) -> None:
        """Run periodic insight aggregation (same as handlers.py)."""
        if not self.insight_aggregator:
            return
        self._interaction_count += 1
        if self._interaction_count >= self._aggregation_interval:
            try:
                self.insight_aggregator.run_aggregation()
                self._interaction_count = 0
            except Exception as e:
                logger.warning("Insight aggregation failed", error=str(e))

    # ── Server lifecycle ───────────────────────────────────────────

    async def start(self) -> web.AppRunner:
        """Start the webhook HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", settings.whatsapp.webhook_port)
        await site.start()
        logger.info(
            "WhatsApp webhook server started",
            port=settings.whatsapp.webhook_port,
        )
        return runner

    async def shutdown(self):
        """Clean up resources."""
        if self._client:
            await self._client.close()
        logger.info("WhatsApp handler shut down")
