# Comprehensive Memory & Learning System Design

## Shopify Analytics Agent — v2 Architecture

---

## 1. Executive Summary

This document designs a **multi-layer memory and learning system** that enables the Shopify Analytics Agent to improve over time as more users interact with it. The system learns what works, what doesn't, and gradually becomes smarter at query construction, tool selection, and response personalization.

**Core principles:**
- **Learn from every interaction** — successes AND failures
- **Cross-user intelligence** — what one user teaches benefits all users
- **Channel-agnostic** — works identically across Telegram, WhatsApp, or any future channel
- **Graceful degradation** — the system works fine with zero history and gets better over time
- **Privacy-first** — per-user data can be fully wiped; global knowledge is anonymized

---

## 2. Architecture Overview

The new system is organized into **5 memory layers**, each serving a distinct purpose:

```
┌─────────────────────────────────────────────────────────┐
│                    MEMORY LAYERS                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: SESSION MEMORY (within conversation)           │
│  ├── Current conversation context                        │
│  ├── Working state (what tools were tried, what worked)  │
│  └── In-flight corrections ("no, I meant last month")   │
│                                                          │
│  Layer 2: USER MEMORY (per user, long-term)              │
│  ├── Query preferences & patterns                        │
│  ├── Favorite metrics, time ranges, query styles         │
│  ├── Business context & seasonal events                  │
│  └── Communication preferences                           │
│                                                          │
│  Layer 3: QUERY KNOWLEDGE (global, shared)               │
│  ├── Successful query templates                          │
│  ├── Intent → query mappings                             │
│  ├── Error recovery patterns (X fails → try Y)          │
│  └── GraphQL field/sort-key validations                  │
│                                                          │
│  Layer 4: TOOL INTELLIGENCE (global, shared)             │
│  ├── Tool success rates per query type                   │
│  ├── Execution time benchmarks                           │
│  ├── Tool selection rules                                │
│  └── Parameter validation rules                          │
│                                                          │
│  Layer 5: RESPONSE QUALITY (feedback-driven)             │
│  ├── User satisfaction signals                           │
│  ├── Response effectiveness scoring                      │
│  └── Format preference learning                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema — New & Modified Tables

### 3.1 New Tables

#### `query_templates` — Successful Query Library

Stores proven GraphQL queries that worked, indexed by the natural language intent that triggered them. When Claude sees a similar question in the future, it can reuse a known-good query instead of constructing one from scratch.

```python
class QueryTemplate(Base):
    """Stores successful query patterns for reuse."""
    __tablename__ = "query_templates"

    id: Mapped[int] = mapped_column(primary_key=True)

    # What the user asked (natural language intent)
    intent_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # e.g., "top_products_by_revenue", "orders_last_7_days", "customer_spending"

    intent_description: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g., "User wants to see top selling products sorted by revenue"

    # What tool was used
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # "shopify_analytics" or "shopify_graphql"

    # Exact parameters / query that worked
    tool_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON: {"resource": "products", "sort_key": "BEST_SELLING", ...}
    # or for raw graphql: {"query": "{ orders(first:10 ...) { ... } }"}

    # Quality metrics
    success_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_execution_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Confidence: success_count / (success_count + failure_count)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Metadata
    created_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # NULL = system-generated, otherwise the user who first triggered it
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Example natural language queries that match this template
    example_queries: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON array: ["top products this week", "best sellers last month", ...]
```

**Key indexes:** `(intent_category)`, `(tool_name, intent_category)`, `(confidence DESC)`

---

#### `error_recovery_patterns` — When X Fails, Try Y

Stores pairs of failed-query → successful-recovery, so when a similar error happens again, the system can suggest the fix immediately.

```python
class ErrorRecoveryPattern(Base):
    """Maps failed queries to their successful recoveries."""
    __tablename__ = "error_recovery_patterns"

    id: Mapped[int] = mapped_column(primary_key=True)

    # The error pattern (what went wrong)
    error_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    failed_tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    failed_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON of the parameters that failed
    error_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # Hash of (error_type + tool + key_params) for deduplication

    # The recovery (what worked instead)
    recovery_tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    recovery_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    recovery_description: Mapped[str] = mapped_column(Text, nullable=False)
    # e.g., "Changed sort_key from 'REVENUE' to 'BEST_SELLING'"

    # Quality metrics
    times_applied: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    times_succeeded: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

**Key indexes:** `(error_fingerprint)`, `(error_type, failed_tool_name)`

---

#### `global_insights` — Cross-User Aggregate Intelligence

Stores anonymized, aggregated patterns across ALL users. This is the "collective wisdom" table.

```python
class GlobalInsight(Base):
    """Anonymized cross-user intelligence."""
    __tablename__ = "global_insights"

    id: Mapped[int] = mapped_column(primary_key=True)

    insight_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # Types: "popular_query", "tool_preference", "time_pattern",
    #        "common_mistake", "best_practice", "field_mapping"

    insight_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # e.g., "revenue_query_tool_preference", "sort_key_products_best"

    insight_value: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON: {"preferred_tool": "shopify_analytics", "confidence": 0.85, ...}

    sample_size: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # How many data points contributed to this insight

    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("insight_type", "insight_key", name="unique_insight"),
    )
```

---

#### `response_feedback` — User Satisfaction Tracking

Tracks implicit and explicit signals about whether a response was helpful.

```python
class ResponseFeedback(Base):
    """Tracks response quality signals."""
    __tablename__ = "response_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )

    # Feedback type
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # "follow_up_correction" — user corrected the agent
    # "follow_up_refinement" — user asked for more detail (positive)
    # "follow_up_new_topic" — user moved on (neutral)
    # "explicit_positive" — user said thanks/great/etc.
    # "explicit_negative" — user expressed frustration
    # "no_follow_up" — user didn't respond (neutral-positive)

    # Derived quality score: -1.0 (terrible) to 1.0 (excellent)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # What was the query type and tool used
    query_type: Mapped[Optional[str]] = mapped_column(String(100))
    tool_used: Mapped[Optional[str]] = mapped_column(String(100))

    # Optional text signal
    signal_text: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

---

#### `channel_sessions` — Channel-Agnostic Session Tracking

Decouples the session/channel concept from the core User model, enabling multi-channel support.

```python
class ChannelSession(Base):
    """Maps channel-specific identifiers to internal users."""
    __tablename__ = "channel_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )

    # Channel identification
    channel_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # "telegram", "whatsapp", "web", "api"

    channel_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Telegram: str(telegram_user_id)
    # WhatsApp: phone number
    # Web: session token

    channel_username: Mapped[Optional[str]] = mapped_column(String(255))
    channel_metadata: Mapped[Optional[str]] = mapped_column(Text)
    # JSON: channel-specific data (e.g., WhatsApp profile, Telegram chat_id)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("channel_type", "channel_user_id", name="unique_channel_user"),
    )
```

---

### 3.2 Modified Existing Tables

#### `users` — Make Channel-Agnostic

The User model becomes the **canonical identity**, decoupled from Telegram:

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Keep telegram_user_id for backward compatibility
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(255))
    first_name: Mapped[Optional[str]] = mapped_column(String(255))

    # NEW: General display name (channel-agnostic)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))

    # NEW: User tier for learning personalization
    interaction_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Total interactions across all channels

    created_at: Mapped[datetime]
    last_active: Mapped[datetime]

    # NEW relationship
    channel_sessions: Mapped[list["ChannelSession"]] = relationship(...)
    response_feedback: Mapped[list["ResponseFeedback"]] = relationship(...)
```

#### `conversations` — Add Channel & Feedback Tracking

```python
class Conversation(Base):
    __tablename__ = "conversations"

    # ... existing fields ...

    # NEW fields
    channel_type: Mapped[str] = mapped_column(String(50), default="telegram", nullable=False)
    # Which channel this conversation came from

    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON array of tool calls made during this conversation
    # [{"tool": "shopify_analytics", "params": {...}, "success": true, "time_ms": 230}]

    response_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Computed from ResponseFeedback, -1.0 to 1.0

    template_id_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # If a QueryTemplate was used, reference it here
```

#### `mcp_tool_usage` → Rename to `tool_usage`

The legacy MCP name is confusing. Rename the table (with migration) and add fields:

```python
class ToolUsage(Base):
    __tablename__ = "tool_usage"  # renamed from mcp_tool_usage

    # ... existing fields ...

    # NEW fields
    intent_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # What type of user intent triggered this tool use

    query_template_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # If a template was used, link it

    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # If failed, what kind of error

    channel_type: Mapped[str] = mapped_column(String(50), default="telegram")
```

#### `query_errors` — Add Recovery Tracking

```python
class QueryError(Base):
    __tablename__ = "query_errors"

    # ... existing fields ...

    # NEW fields
    recovery_pattern_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("error_recovery_patterns.id"), nullable=True
    )
    # If this error was recovered from, link to the recovery pattern

    was_auto_recovered: Mapped[bool] = mapped_column(Boolean, default=False)
    # True if the system automatically applied a recovery pattern

    original_intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # What the user was trying to do when this error happened
```

---

## 4. Learning Pipelines

### 4.1 Pipeline 1: Query Template Learning

**When it fires:** After every SUCCESSFUL tool call.

**Flow:**
```
User asks question
    → Claude selects tool + parameters
    → Tool executes SUCCESSFULLY
    → System extracts intent category from user's query
    → System checks: does a matching template exist?
        → YES: increment success_count, update confidence, update last_used_at
        → NO: create new template from this successful interaction
    → Store template reference in conversation record
```

**Intent categorization** uses a two-level system:
1. **Coarse category** (rule-based): revenue, orders, products, customers, comparison, shop_info, inventory, other
2. **Fine category** (from query analysis): top_products_by_revenue, orders_date_filtered, customer_lifetime_value, etc.

**Template matching algorithm:**
```python
def find_matching_template(intent_category, tool_name, key_params):
    """Find a template that matches the current query intent."""
    # 1. Exact match on intent_category + tool_name
    templates = db.query(QueryTemplate).filter(
        intent_category=intent_category,
        tool_name=tool_name,
        confidence >= 0.7,
    ).order_by(confidence.desc(), success_count.desc())

    # 2. Score each template against current params
    for template in templates:
        similarity = compute_param_similarity(
            json.loads(template.tool_parameters),
            key_params,
        )
        if similarity > 0.8:
            return template

    # 3. Fall back to intent_category-only match
    return db.query(QueryTemplate).filter(
        intent_category=intent_category,
        confidence >= 0.9,
    ).first()
```

---

### 4.2 Pipeline 2: Error Recovery Learning

**When it fires:** When a tool call FAILS and then the conversation continues with a SUCCESSFUL call.

**Flow:**
```
User asks question
    → Claude tries Tool A with params X
    → FAILS with error E
    → Error is logged to query_errors
    → Claude retries with Tool B / params Y
    → SUCCEEDS
    → System creates ErrorRecoveryPattern:
        failed: (Tool A, params X, error type E)
        recovery: (Tool B, params Y, description of change)
```

**Error fingerprinting** creates a stable hash for deduplication:
```python
def compute_error_fingerprint(error_type, tool_name, key_params):
    """Create a stable fingerprint for error deduplication."""
    # Normalize key params (remove values, keep structure)
    normalized = {
        "error_type": error_type,
        "tool": tool_name,
    }
    if tool_name == "shopify_analytics":
        normalized["resource"] = key_params.get("resource")
        normalized["sort_key"] = key_params.get("sort_key")
    elif tool_name == "shopify_graphql":
        # Extract the operation type from the query
        query = key_params.get("query", "")
        normalized["query_root"] = extract_query_root_fields(query)

    return hashlib.md5(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
```

**Using recovery patterns in future queries:**
```python
# In _build_system_prompt(), before the error list:
recovery_patterns = db.get_recovery_patterns(limit=10, min_confidence=0.8)
if recovery_patterns:
    system_parts.append("KNOWN ERROR RECOVERY PATTERNS:")
    for rp in recovery_patterns:
        system_parts.append(
            f"  If {rp.error_type} occurs with {rp.failed_tool_name}: "
            f"{rp.recovery_description} "
            f"(worked {rp.times_succeeded}/{rp.times_applied} times)"
        )
```

---

### 4.3 Pipeline 3: Cross-User Intelligence Aggregation

**When it fires:** Periodically (every N interactions or on a schedule) or on-demand.

**What it computes:**

1. **Popular query patterns** — What do most users ask about?
   ```python
   # Aggregate query_type distribution across all users
   # → GlobalInsight(type="popular_query", key="revenue", value={"frequency": 450, "pct": 35.2})
   ```

2. **Tool effectiveness by query type** — Which tool works better for which intent?
   ```python
   # For each intent_category, compute:
   #   tool_success_rate = successful_uses / total_uses per tool
   # → GlobalInsight(type="tool_preference", key="top_products",
   #     value={"shopify_analytics": {"success_rate": 0.95, "avg_time": 180},
   #            "shopify_graphql": {"success_rate": 0.72, "avg_time": 350}})
   ```

3. **Common mistakes** — What errors happen most frequently?
   ```python
   # Aggregate query_errors by error_type + tool_name
   # → GlobalInsight(type="common_mistake", key="invalid_field_shopify_graphql",
   #     value={"count": 23, "common_fields": ["totalPrice", "createdAt"],
   #            "correct_fields": ["totalPriceSet", "createdAt"]})
   ```

4. **Best practices** — Proven parameter combinations
   ```python
   # From query_templates with highest confidence
   # → GlobalInsight(type="best_practice", key="orders_date_filter",
   #     value={"use": "shopify_analytics", "sort_key": "CREATED_AT",
   #            "query_format": "created_at:>{start} created_at:<{end}"})
   ```

5. **GraphQL field mappings** — Valid field names discovered through trial/error
   ```python
   # Extracted from successful queries
   # → GlobalInsight(type="field_mapping", key="products",
   #     value={"valid_sort_keys": ["TITLE","PRICE","BEST_SELLING","CREATED_AT"],
   #            "common_fields": ["title","handle","status","totalInventory"],
   #            "money_fields": ["priceRangeV2.minVariantPrice"]})
   ```

---

### 4.4 Pipeline 4: Response Quality Learning

**Implicit feedback signals** (no user action needed):

| Signal | Detection Method | Quality Score |
|--------|-----------------|---------------|
| User says "thanks", "great", "perfect" | Keyword detection in next message | +0.8 |
| User asks a follow-up on same topic | Same query_type in next message | +0.3 |
| User asks to refine/drill down | Refinement keywords + same topic | +0.5 |
| User corrects the agent | Correction keywords ("no", "wrong", "I meant") | -0.5 |
| User repeats same question | Duplicate detection | -0.7 |
| User expresses frustration | Frustration keywords ("doesn't work", "useless") | -0.8 |
| User switches to unrelated topic | Different query_type | 0.0 (neutral) |
| User doesn't respond (>30 min gap) | Timeout detection | +0.1 (slight positive) |

**How feedback is used:**
- Templates with low quality scores get demoted (lower confidence)
- Templates with high quality scores get promoted
- Persistent low-quality responses trigger system prompt adjustments
- Quality scores feed into the global insights aggregation

---

### 4.5 Pipeline 5: Temporal Decay & Relevance

All pattern frequencies and confidence scores decay over time to keep the system responsive to changing user behavior.

**Decay formula:**
```python
def compute_decayed_score(base_score, last_used_at, half_life_days=30):
    """Apply exponential decay to a score based on recency."""
    days_since = (datetime.utcnow() - last_used_at).days
    decay_factor = 0.5 ** (days_since / half_life_days)
    return base_score * decay_factor
```

**Applied to:**
- `QueryPattern.frequency` — older patterns matter less
- `QueryTemplate.confidence` — old templates should be re-validated
- `UserPreference.confidence_score` — preferences can shift over time
- `ErrorRecoveryPattern.confidence` — API changes may invalidate old recoveries

**Decay schedule:**
- User patterns: 30-day half-life
- Query templates: 60-day half-life (more stable)
- Error recoveries: 14-day half-life (APIs change frequently)
- Global insights: 90-day half-life (slow-changing)

---

## 5. Channel-Agnostic Architecture

### 5.1 The Channel Adapter Pattern

```
                    ┌──────────────────────┐
                    │     Core Agent        │
                    │  (Claude + Learning)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │   Channel Router      │
                    │  (resolves user,      │
                    │   normalizes I/O)     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼────────┐
    │  Telegram       │  │  WhatsApp   │  │  Future       │
    │  Adapter        │  │  Adapter    │  │  Channels     │
    │                 │  │             │  │               │
    │  - Bot API      │  │  - Business │  │  - Web API    │
    │  - HTML format  │  │    API      │  │  - Slack      │
    │  - Inline KB    │  │  - Template │  │  - Discord    │
    │  - Commands     │  │    messages │  │  - etc.       │
    └─────────────────┘  └────────────┘  └───────────────┘
```

### 5.2 Channel Adapter Interface

Every channel adapter must implement:

```python
class ChannelAdapter(ABC):
    """Base class for all channel adapters."""

    @abstractmethod
    def get_channel_type(self) -> str:
        """Return channel identifier: 'telegram', 'whatsapp', etc."""

    @abstractmethod
    async def resolve_user(self, channel_event) -> tuple[int, dict]:
        """
        Map a channel-specific event to an internal user_id.

        Returns:
            (user_id, metadata_dict)
        """

    @abstractmethod
    async def send_response(self, channel_event, response: str) -> None:
        """
        Send the agent's response through this channel.
        Handles channel-specific formatting (HTML for Telegram,
        plain text for WhatsApp, etc.)
        """

    @abstractmethod
    async def send_error(self, channel_event, error: Exception) -> None:
        """Send a user-friendly error message."""

    @abstractmethod
    def format_response(self, markdown_response: str) -> str:
        """
        Convert the agent's internal format (Markdown) to
        channel-specific format.
        - Telegram → HTML
        - WhatsApp → WhatsApp-flavored markdown
        - Web → full Markdown/HTML
        """

    @abstractmethod
    def get_message_limit(self) -> int:
        """Max characters per message for this channel."""
        # Telegram: 4096, WhatsApp: 65536, Web: unlimited
```

### 5.3 Format-Agnostic Response Pipeline

The agent always generates responses in **canonical markdown**. Each channel adapter converts to its native format:

```
Claude generates Markdown
    → Core stores in conversations table (canonical markdown)
    → Channel adapter converts:
        Telegram: markdown_to_telegram_html()
        WhatsApp: markdown_to_whatsapp()  # *bold*, _italic_, ```code```
        Web:      render as-is or to HTML
```

---

## 6. Enhanced Learning Components

### 6.1 Enhanced PatternLearner

**New capabilities beyond the current implementation:**

```python
class PatternLearner:
    """Enhanced pattern learner with multi-signal analysis."""

    def learn_from_query(self, user_id, query, query_type=None):
        # ... existing pattern extraction ...

        # NEW: Intent classification (coarse + fine)
        intent = self.classify_intent(query)
        self.db_ops.update_pattern_frequency(
            user_id, "intent", intent.fine_category
        )

        # NEW: Complexity assessment
        complexity = self.assess_query_complexity(query)
        # "simple" (single metric/period), "moderate" (comparison, multi-metric),
        # "complex" (multi-step, conditional)
        self.db_ops.update_pattern_frequency(
            user_id, "complexity", complexity
        )

    def classify_intent(self, query: str) -> Intent:
        """Two-level intent classification."""
        coarse = self.detect_query_type(query)  # existing

        # Fine-grained classification
        metrics = self.extract_metrics(query)
        time_range = self.extract_time_range(query)
        has_comparison = coarse == "comparison"
        has_ranking = any(w in query.lower() for w in
            ["top", "best", "worst", "highest", "lowest", "most", "least"])

        fine = self._build_fine_intent(
            coarse, metrics, time_range, has_comparison, has_ranking
        )
        return Intent(coarse=coarse, fine=fine)

    def _build_fine_intent(self, coarse, metrics, time_range, comparison, ranking):
        """Build fine-grained intent string."""
        parts = [coarse]
        if ranking:
            parts.append("ranking")
        if metrics:
            parts.append(metrics[0])
        if time_range:
            parts.append(time_range)
        if comparison:
            parts.append("comparison")
        return "_".join(parts)
        # e.g., "products_ranking_revenue_last_30_days"

    def assess_query_complexity(self, query: str) -> str:
        """Assess how complex a query is."""
        signals = 0
        if len(self.extract_metrics(query)) > 1:
            signals += 1
        if self.detect_query_type(query) == "comparison":
            signals += 1
        if any(w in query.lower() for w in ["and", "also", "plus", "as well as"]):
            signals += 1
        if len(query.split()) > 20:
            signals += 1

        if signals == 0:
            return "simple"
        elif signals <= 2:
            return "moderate"
        else:
            return "complex"
```

### 6.2 Enhanced ContextBuilder

**New capabilities:**

```python
class ContextBuilder:

    def build_context(self, user_id: int) -> dict:
        # ... existing context building ...

        # NEW: Add query templates relevant to user's patterns
        context["recommended_templates"] = self._get_relevant_templates(user_id)

        # NEW: Add recovery patterns for common errors
        context["recovery_patterns"] = self._get_recovery_patterns()

        # NEW: Add global insights
        context["global_insights"] = self._get_relevant_insights(user_id)

        # NEW: Add user's response quality history
        context["quality_trend"] = self._get_quality_trend(user_id)

        return context

    def _get_relevant_templates(self, user_id: int, limit: int = 5) -> list:
        """Get query templates relevant to this user's common intents."""
        # Get user's top intent categories
        top_intents = self.db_ops.get_top_patterns(
            user_id, pattern_type="intent", limit=3
        )

        templates = []
        for intent_pattern in top_intents:
            matching = self.db_ops.get_templates_by_intent(
                intent_pattern.pattern_value,
                min_confidence=0.7,
                limit=2,
            )
            templates.extend(matching)

        return templates[:limit]

    def _get_recovery_patterns(self, limit: int = 5) -> list:
        """Get most useful error recovery patterns."""
        return self.db_ops.get_top_recovery_patterns(
            min_confidence=0.8,
            min_times_applied=2,
            limit=limit,
        )

    def _get_relevant_insights(self, user_id: int) -> list:
        """Get global insights relevant to this user."""
        insights = []

        # Tool preferences for user's common query types
        top_types = self.db_ops.get_top_patterns(
            user_id, pattern_type="query_type", limit=3
        )
        for tp in top_types:
            tool_pref = self.db_ops.get_global_insight(
                "tool_preference", tp.pattern_value
            )
            if tool_pref:
                insights.append(tool_pref)

        # Common mistakes to avoid
        common_mistakes = self.db_ops.get_global_insights_by_type(
            "common_mistake", limit=3
        )
        insights.extend(common_mistakes)

        return insights

    def format_context_for_prompt(self, context: dict) -> str:
        """Enhanced formatting with new context sections."""
        lines = []

        # ... existing sections ...

        # NEW: Recommended Query Templates section
        if context.get("recommended_templates"):
            lines.append("\nPROVEN QUERY TEMPLATES (use these when applicable):")
            for t in context["recommended_templates"]:
                lines.append(f"  Intent: {t.intent_description}")
                lines.append(f"  Tool: {t.tool_name}")
                lines.append(f"  Parameters: {t.tool_parameters[:200]}")
                lines.append(f"  Confidence: {t.confidence:.0%} "
                           f"({t.success_count} successes)")
                lines.append("")

        # NEW: Error Recovery Patterns section
        if context.get("recovery_patterns"):
            lines.append("\nERROR RECOVERY PATTERNS (apply these if errors occur):")
            for rp in context["recovery_patterns"]:
                lines.append(f"  If {rp.error_type} with {rp.failed_tool_name}:")
                lines.append(f"  → {rp.recovery_description}")
                lines.append(f"  Success rate: {rp.confidence:.0%}")
                lines.append("")

        # NEW: Global Insights section
        if context.get("global_insights"):
            lines.append("\nGLOBAL INSIGHTS:")
            for gi in context["global_insights"]:
                if gi.insight_type == "tool_preference":
                    val = json.loads(gi.insight_value)
                    lines.append(f"  For {gi.insight_key} queries: "
                               f"prefer {val.get('preferred_tool')}")
                elif gi.insight_type == "common_mistake":
                    val = json.loads(gi.insight_value)
                    lines.append(f"  Common mistake: {val.get('description', gi.insight_key)}")

        return "\n".join(lines)
```

### 6.3 New Component: TemplateManager

```python
class TemplateManager:
    """Manages query template lifecycle: creation, matching, updating."""

    def __init__(self, db_ops):
        self.db_ops = db_ops

    def record_successful_query(
        self,
        user_id: int,
        user_message: str,
        tool_name: str,
        tool_params: dict,
        execution_time_ms: int,
        intent_category: str,
    ):
        """Record a successful tool call as a query template."""
        # Check for existing template
        existing = self.db_ops.find_template(
            intent_category=intent_category,
            tool_name=tool_name,
        )

        if existing:
            # Update existing template
            self.db_ops.increment_template_success(existing.id, execution_time_ms)
            # Add this query as an example
            self._add_example_query(existing.id, user_message)
        else:
            # Create new template
            self.db_ops.create_template(
                intent_category=intent_category,
                intent_description=self._generate_description(
                    user_message, tool_name, tool_params
                ),
                tool_name=tool_name,
                tool_parameters=json.dumps(tool_params),
                created_by_user_id=user_id,
                example_queries=json.dumps([user_message[:200]]),
                avg_execution_time_ms=execution_time_ms,
            )

    def record_failed_query(self, intent_category: str, tool_name: str):
        """Record a failed tool call against a template."""
        existing = self.db_ops.find_template(
            intent_category=intent_category,
            tool_name=tool_name,
        )
        if existing:
            self.db_ops.increment_template_failure(existing.id)

    def get_template_for_intent(self, intent_category: str) -> Optional[dict]:
        """Get the best template for a given intent."""
        template = self.db_ops.get_best_template(
            intent_category=intent_category,
            min_confidence=0.7,
        )
        if template:
            return {
                "tool_name": template.tool_name,
                "parameters": json.loads(template.tool_parameters),
                "confidence": template.confidence,
                "description": template.intent_description,
            }
        return None
```

### 6.4 New Component: RecoveryManager

```python
class RecoveryManager:
    """Manages error recovery pattern lifecycle."""

    def __init__(self, db_ops):
        self.db_ops = db_ops
        self._pending_errors = {}  # session-level tracking

    def record_error(
        self,
        session_id: str,
        tool_name: str,
        tool_params: dict,
        error_type: str,
        error_message: str,
    ):
        """Record an error for potential recovery tracking."""
        fingerprint = self._compute_fingerprint(error_type, tool_name, tool_params)
        self._pending_errors[session_id] = {
            "tool_name": tool_name,
            "tool_params": tool_params,
            "error_type": error_type,
            "error_message": error_message,
            "fingerprint": fingerprint,
        }

    def record_recovery(
        self,
        session_id: str,
        recovery_tool_name: str,
        recovery_params: dict,
    ):
        """Record a successful recovery after a previous error."""
        pending = self._pending_errors.pop(session_id, None)
        if not pending:
            return

        # Generate recovery description
        description = self._describe_recovery(
            pending["tool_name"], pending["tool_params"],
            recovery_tool_name, recovery_params,
        )

        # Check for existing recovery pattern
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

    def suggest_recovery(self, error_type: str, tool_name: str, tool_params: dict) -> Optional[dict]:
        """Suggest a recovery based on past patterns."""
        fingerprint = self._compute_fingerprint(error_type, tool_name, tool_params)

        # Try exact fingerprint match first
        pattern = self.db_ops.find_recovery_pattern(fingerprint)

        # Fall back to error_type + tool_name match
        if not pattern:
            pattern = self.db_ops.find_recovery_by_type(error_type, tool_name)

        if pattern and pattern.confidence >= 0.7:
            return {
                "recovery_tool": pattern.recovery_tool_name,
                "recovery_params": json.loads(pattern.recovery_parameters),
                "description": pattern.recovery_description,
                "confidence": pattern.confidence,
            }
        return None

    def _describe_recovery(self, failed_tool, failed_params, recovery_tool, recovery_params):
        """Generate a human-readable description of what changed."""
        changes = []

        if failed_tool != recovery_tool:
            changes.append(f"Switched from {failed_tool} to {recovery_tool}")

        if failed_tool == recovery_tool == "shopify_analytics":
            if failed_params.get("sort_key") != recovery_params.get("sort_key"):
                changes.append(
                    f"Changed sort_key from '{failed_params.get('sort_key')}' "
                    f"to '{recovery_params.get('sort_key')}'"
                )
            if failed_params.get("resource") != recovery_params.get("resource"):
                changes.append(
                    f"Changed resource from '{failed_params.get('resource')}' "
                    f"to '{recovery_params.get('resource')}'"
                )

        return "; ".join(changes) if changes else "Modified query parameters"
```

### 6.5 New Component: FeedbackAnalyzer

```python
class FeedbackAnalyzer:
    """Analyzes user behavior to infer response quality."""

    # Positive signals
    POSITIVE_KEYWORDS = [
        "thanks", "thank you", "great", "perfect", "awesome", "helpful",
        "exactly", "nice", "good", "excellent", "wonderful", "brilliant",
    ]

    # Negative signals
    NEGATIVE_KEYWORDS = [
        "wrong", "incorrect", "not what i", "doesn't work", "that's not",
        "no i meant", "no, i", "useless", "terrible", "broken",
    ]

    # Correction signals
    CORRECTION_KEYWORDS = [
        "no,", "not that", "i meant", "i said", "actually", "instead",
        "wrong", "try again", "redo", "different",
    ]

    # Refinement signals (positive — user wants more)
    REFINEMENT_KEYWORDS = [
        "more detail", "tell me more", "can you also", "what about",
        "break it down", "drill down", "specifically", "in particular",
    ]

    def analyze_follow_up(
        self,
        previous_response: str,
        previous_query_type: str,
        current_message: str,
        current_query_type: str,
        time_gap_seconds: int,
    ) -> ResponseFeedback:
        """Analyze the user's follow-up message to infer quality of previous response."""

        msg_lower = current_message.lower()

        # Check explicit positive
        if any(kw in msg_lower for kw in self.POSITIVE_KEYWORDS):
            return ResponseFeedback(
                feedback_type="explicit_positive",
                quality_score=0.8,
                signal_text=current_message[:100],
            )

        # Check corrections
        if any(kw in msg_lower for kw in self.CORRECTION_KEYWORDS):
            return ResponseFeedback(
                feedback_type="follow_up_correction",
                quality_score=-0.5,
                signal_text=current_message[:100],
            )

        # Check frustration
        if any(kw in msg_lower for kw in self.NEGATIVE_KEYWORDS):
            return ResponseFeedback(
                feedback_type="explicit_negative",
                quality_score=-0.8,
                signal_text=current_message[:100],
            )

        # Check refinement (positive)
        if any(kw in msg_lower for kw in self.REFINEMENT_KEYWORDS):
            return ResponseFeedback(
                feedback_type="follow_up_refinement",
                quality_score=0.5,
                signal_text=current_message[:100],
            )

        # Same topic continuation (mildly positive)
        if current_query_type == previous_query_type:
            return ResponseFeedback(
                feedback_type="follow_up_same_topic",
                quality_score=0.2,
            )

        # Different topic (neutral)
        return ResponseFeedback(
            feedback_type="follow_up_new_topic",
            quality_score=0.0,
        )
```

### 6.6 New Component: InsightAggregator

```python
class InsightAggregator:
    """Periodically aggregates cross-user data into global insights."""

    def __init__(self, db_ops):
        self.db_ops = db_ops

    def run_aggregation(self):
        """Run all aggregation pipelines."""
        self._aggregate_popular_queries()
        self._aggregate_tool_preferences()
        self._aggregate_common_mistakes()
        self._aggregate_best_practices()
        self._aggregate_field_mappings()
        self._prune_stale_insights()

    def _aggregate_tool_preferences(self):
        """Compute which tool works best for each query type."""
        query_types = ["revenue", "orders", "products", "customers",
                       "comparison", "shop_info", "inventory"]

        for qt in query_types:
            # Get success rates per tool for this query type
            stats = self.db_ops.get_tool_stats_by_intent(qt)
            # stats = [{"tool": "shopify_analytics", "successes": 45, "failures": 3, "avg_time": 200}, ...]

            if not stats:
                continue

            best_tool = max(stats, key=lambda s: s["successes"] / max(s["successes"] + s["failures"], 1))

            self.db_ops.upsert_global_insight(
                insight_type="tool_preference",
                insight_key=qt,
                insight_value=json.dumps({
                    "preferred_tool": best_tool["tool"],
                    "success_rate": best_tool["successes"] / max(best_tool["successes"] + best_tool["failures"], 1),
                    "avg_time_ms": best_tool["avg_time"],
                    "all_tools": stats,
                }),
                sample_size=sum(s["successes"] + s["failures"] for s in stats),
                confidence=best_tool["successes"] / max(best_tool["successes"] + best_tool["failures"], 1),
            )

    def _aggregate_common_mistakes(self):
        """Identify the most common error patterns."""
        error_groups = self.db_ops.get_error_groups(days=30)
        # groups = [{"error_type": "invalid_field", "tool": "shopify_graphql", "count": 15}, ...]

        for group in error_groups:
            if group["count"] < 3:
                continue  # Not common enough

            self.db_ops.upsert_global_insight(
                insight_type="common_mistake",
                insight_key=f"{group['error_type']}_{group['tool']}",
                insight_value=json.dumps({
                    "description": f"{group['error_type']} errors with {group['tool']}",
                    "count": group["count"],
                    "common_lessons": group.get("lessons", []),
                }),
                sample_size=group["count"],
                confidence=min(group["count"] / 20, 1.0),
            )

    def _prune_stale_insights(self, max_age_days=90):
        """Remove insights that haven't been updated recently."""
        self.db_ops.delete_stale_insights(max_age_days)
```

---

## 7. Integration Flow — Updated Message Processing

Here's the complete updated flow showing how all layers work together:

```python
async def handle_message(update, context):
    # ── Step 0: Resolve channel & user ──
    channel_adapter = get_adapter(update)  # Telegram, WhatsApp, etc.
    user_id, metadata = await channel_adapter.resolve_user(update)
    user = db_ops.get_or_create_user(user_id, metadata)

    # ── Step 1: Show typing ──
    await channel_adapter.show_typing(update)

    # ── Step 2: Analyze the incoming message (pre-processing) ──
    intent = pattern_learner.classify_intent(message_text)
    complexity = pattern_learner.assess_query_complexity(message_text)

    # ── Step 3: Check for quality feedback on PREVIOUS response ──
    previous_conv = db_ops.get_latest_conversation(user.id)
    if previous_conv:
        feedback = feedback_analyzer.analyze_follow_up(
            previous_conv.response,
            previous_conv.query_type,
            message_text,
            intent.coarse,
            time_gap_seconds,
        )
        if feedback.quality_score != 0.0:
            db_ops.save_response_feedback(previous_conv.id, user.id, feedback)

            # Update template confidence based on feedback
            if previous_conv.template_id_used:
                template_manager.update_template_quality(
                    previous_conv.template_id_used,
                    feedback.quality_score,
                )

    # ── Step 4: Process through Claude ──
    # (Claude's system prompt now includes templates, recovery patterns, global insights)
    response = await claude_service.process_message(
        user_id=user.id,
        message=message_text,
        intent=intent,
    )

    # ── Step 5: Learn from this interaction ──
    # 5a. Pattern learning (existing)
    pattern_learner.learn_from_query(user.id, message_text)
    preference_manager.update_preferences_from_patterns(user.id)

    # 5b. Template learning (NEW)
    # (happens inside claude_service when tools succeed/fail)

    # 5c. Periodic global aggregation check
    if should_run_aggregation():
        insight_aggregator.run_aggregation()

    # ── Step 6: Format & send response ──
    formatted = channel_adapter.format_response(response)
    await channel_adapter.send_response(update, formatted)

    # ── Step 7: Save conversation ──
    db_ops.save_conversation(
        user_id=user.id,
        message=message_text,
        response=response,
        query_type=intent.coarse,
        channel_type=channel_adapter.get_channel_type(),
        tool_calls_json=claude_service.last_tool_calls_json,
    )
```

---

## 8. Edge Cases & How We Handle Them

### 8.1 Cold Start — Brand New User, Zero History

**Problem:** No patterns, preferences, or templates to draw from.

**Solution:**
- Global insights provide a baseline ("most users ask about revenue first")
- Query templates with high confidence work as defaults
- System prompt includes general guidance without personalization
- After just 3-5 interactions, basic patterns emerge

### 8.2 Cold Start — Brand New Deployment, No Global Data

**Problem:** Fresh installation with no global insights at all.

**Solution:**
- Ship with a set of **seed templates** covering common queries:
  - "Top products" → shopify_analytics with resource=products, sort_key=BEST_SELLING
  - "Recent orders" → shopify_analytics with resource=orders, sort_key=CREATED_AT
  - "Shop info" → shopify_graphql with { shop { name ... } }
- Seed templates have confidence=0.5 (moderate) and get validated/promoted through use
- Error classification and lesson generation work immediately (no history needed)

### 8.3 Conflicting Templates

**Problem:** Two templates for the same intent have different tools/params.

**Solution:**
- Always prefer higher confidence
- When confidence is tied, prefer more recently used
- When tied further, prefer shorter execution time
- If the chosen template fails, automatically try the next best

### 8.4 Stale Templates (API Changes)

**Problem:** Shopify updates their API, breaking previously-working queries.

**Solution:**
- Templates that start failing get their failure_count incremented
- Confidence drops: `confidence = success_count / (success_count + failure_count)`
- Below 0.5 confidence → template is excluded from recommendations
- Below 0.2 confidence → template is auto-archived
- Error recovery patterns capture the API change and the fix

### 8.5 User Sends Non-Analytics Messages

**Problem:** "Hello", "How are you?", "What can you do?"

**Solution:**
- Pattern learner classifies as "general"
- No template matching for general/greeting intents
- No tool calls → no template/error learning
- Conversation is still saved for context
- Quality feedback still applies to the response

### 8.6 Rapid Fire Duplicate Queries

**Problem:** User sends the same query 5 times quickly (network issues, impatience).

**Solution:**
- Deduplicate within a 30-second window (same user, same message)
- Only the first query triggers learning
- Response is cached and reused for duplicates
- Pattern frequency only incremented once

### 8.7 User Corrects the Agent

**Problem:** "No, I meant last month, not last week"

**Solution:**
- FeedbackAnalyzer detects correction keywords
- Previous response gets negative quality score
- If a template was used, its confidence is reduced
- The correction itself becomes a learning signal:
  - "last week" in context of "last month" → time range disambiguation
  - Stored as a correction pattern for future reference

### 8.8 Multi-Channel Same User

**Problem:** User starts on Telegram, continues on WhatsApp.

**Solution:**
- ChannelSession maps both channel identities to same user_id
- Initial linking: user must verify (e.g., same phone number, or explicit /link command)
- Once linked, all learning data is shared across channels
- Channel-specific preferences (e.g., format preference) stored per channel_session

### 8.9 System Prompt Token Budget

**Problem:** Too much learning context inflates the system prompt.

**Solution:**
- **Hard budget:** Max 2000 tokens for learning context
- **Priority ordering:**
  1. Error recovery patterns (highest value, fewest tokens)
  2. Top 3 query templates for user's common intents
  3. User preferences & patterns
  4. Past errors (truncated)
  5. Global insights (only if budget remains)
- **Truncation rules:**
  - Template params: max 200 chars each
  - Error messages: max 100 chars each
  - Max 5 templates, 5 recovery patterns, 5 past errors
  - Global insights: max 3

### 8.10 Privacy & Data Deletion

**Problem:** User requests to delete all their data (/forget).

**Solution:**
- Delete all per-user data: conversations, patterns, preferences, feedback, channel_sessions
- Query templates created by this user: remove `created_by_user_id` (anonymize), keep the template
- Error logs from this user: remove user_id, keep for system learning
- Global insights: not affected (already anonymized)

### 8.11 Concurrent Users Hitting Same Template

**Problem:** Two users trigger the same template simultaneously; one succeeds, one fails.

**Solution:**
- Template updates are atomic (increment counters)
- SQLite handles this with write-ahead logging (WAL mode)
- If scale requires it, switch to PostgreSQL for proper concurrency

### 8.12 WhatsApp-Specific Challenges

**Problem:** WhatsApp has different constraints than Telegram.

**Handled by:**
- 24-hour messaging window rules → channel adapter manages
- No inline keyboards → text-based menus instead
- Template messages for re-engagement → channel adapter generates
- Media handling differences → format adapter converts
- Phone number as identifier → channel_sessions table maps to user_id

---

## 9. System Prompt Enhancement

The enhanced system prompt structure:

```
[ROLE]
You are a Shopify Analytics Assistant...

[CURRENT CONTEXT]
Date: ...
Store: ...

[USER PROFILE]
- Favorite metrics: revenue, orders
- Preferred time range: last_7_days
- Common query types: revenue (40%), products (30%), orders (20%)
- Interaction count: 47

[PROVEN QUERY TEMPLATES]
(Use these when the user's intent matches)
1. "Top products by revenue" → shopify_analytics(resource=products, sort_key=BEST_SELLING)
   Confidence: 95% (42 successes)
2. "Orders in date range" → shopify_analytics(resource=orders, sort_key=CREATED_AT, query="created_at:>...")
   Confidence: 91% (31 successes)

[ERROR RECOVERY PATTERNS]
(Apply these if you encounter errors)
- If invalid_field with shopify_graphql → Check camelCase field names, use totalPriceSet not total_price
- If type_mismatch with sort_key → Use SCREAMING_SNAKE_CASE enum values

[PAST ERRORS TO AVOID]
1. [shopify_graphql] invalid_field: ...
   Lesson: ...

[GLOBAL INSIGHTS]
- For revenue queries: prefer shopify_analytics (95% success rate)
- Common mistake: Using 'REVENUE' as sort_key (not valid; use BEST_SELLING for products)

[TOOL DEFINITIONS]
...

[FORMATTING RULES]
...
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Core Schema + Channel Abstraction)
- Add new database tables (query_templates, error_recovery_patterns, global_insights, response_feedback, channel_sessions)
- Modify existing tables (add new columns to conversations, users, query_errors)
- Rename mcp_tool_usage → tool_usage
- Create database migration script
- Implement ChannelAdapter base class
- Refactor Telegram-specific code into TelegramAdapter
- Estimated effort: Medium

### Phase 2: Template & Recovery Learning
- Implement TemplateManager
- Implement RecoveryManager
- Integrate into ClaudeService (record successes/failures)
- Add template context to system prompt
- Add recovery pattern context to system prompt
- Seed initial templates
- Estimated effort: Medium

### Phase 3: Feedback & Quality System
- Implement FeedbackAnalyzer
- Integrate into message handler pipeline
- Wire feedback to template confidence updates
- Estimated effort: Small

### Phase 4: Cross-User Intelligence
- Implement InsightAggregator
- Add aggregation trigger (every N interactions or periodic)
- Add global insights to system prompt
- Implement temporal decay across all scores
- Estimated effort: Medium

### Phase 5: Enhanced Pattern Learning
- Upgrade PatternLearner with two-level intent classification
- Add complexity assessment
- Implement token budget management for system prompt
- Estimated effort: Small

### Phase 6: WhatsApp Channel
- Implement WhatsAppAdapter
- Add WhatsApp-specific formatting
- Handle phone-based user resolution
- Implement channel session linking
- Estimated effort: Large (depends on WhatsApp Business API)

---

## 11. Metrics & Observability

### Key Metrics to Track

| Metric | Source | Purpose |
|--------|--------|---------|
| Template hit rate | query_templates.success_count | Are templates being reused? |
| Template confidence trend | query_templates.confidence over time | Are templates getting better? |
| Error recovery rate | error_recovery_patterns.times_succeeded / times_applied | Do recoveries work? |
| First-try success rate | tool_usage WHERE success=True / total | Is the agent getting smarter? |
| Average response quality | response_feedback.quality_score average | Are users happy? |
| Tool selection accuracy | global_insights tool_preference | Is the right tool being chosen? |
| Cold start time | Queries until first template hit | How fast do new users benefit? |
| Correction rate | response_feedback WHERE type=correction / total | How often is the agent wrong? |

### Logging Strategy

```python
# Add structured logging events for learning system
logger.info("template_hit", template_id=t.id, intent=intent, confidence=t.confidence)
logger.info("template_miss", intent=intent, reason="no_match")
logger.info("recovery_applied", pattern_id=rp.id, error_type=et)
logger.info("feedback_recorded", type=fb.feedback_type, score=fb.quality_score)
logger.info("insight_updated", type=gi.insight_type, key=gi.insight_key)
```

---

## 12. Summary

This design transforms the current per-user pattern tracking into a **5-layer adaptive learning system** that:

1. **Remembers** what works (query templates) and what doesn't (error patterns)
2. **Recovers** from errors automatically using proven recovery patterns
3. **Learns** across all users through anonymized global insights
4. **Adapts** to changing APIs through temporal decay and confidence scoring
5. **Scales** to any messaging channel through the adapter pattern
6. **Respects privacy** with clear data separation and deletion capabilities

The system gets smarter with every single interaction — whether it succeeds or fails — and shares that intelligence across all users while keeping individual data private.
