import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from src.bot.handlers import MessageHandler
from src.learning.pattern_learner import Intent
from src.database.models import User, Conversation, Session
from src.config.settings import settings

# --- E2E Simulation Test ---

@pytest.mark.asyncio
async def test_full_e2e_flow(db_ops):
    """
    Simulates a full user journey:
    1. User starts conversation (Unitialized -> Initialized).
    2. User asks ambiguous question -> Intelligent Refinement -> Answer.
    3. User asks specific question -> Contextual Answer.
    """
    print("\n🚀 Starting E2E Simulation...")
    
    # Get session from db_ops
    db_session = db_ops.get_session()

    # --- 1. Setup Mocks & Dependencies ---
    
    # Mock LLM Service
    mock_claude = AsyncMock()
    # Behavior: slightly different response based on input
    async def mock_process_message(user_id, message, intent=None, session_id=None):
        if "sales" in message.lower():
            return "Based on your sales data, revenue is up 10% [CHART:0]"
        if "product" in message.lower():
            return "Top product is 'Widget A'"
        return "Hello! I am your analytics assistant."
    mock_claude.process_message.side_effect = mock_process_message
    mock_claude.last_tool_calls_json = '{"tool": "shopify_analytics"}'

    # Mock Pattern Learner (Real logic is complex, mocking for stability in E2E)
    mock_learner = MagicMock()
    # Scenario: "sales" -> Ambiguous -> Refined
    # Scenario: "products" -> Specific
    def mock_classify(text):
        if "sales" in text.lower():
            return Intent(coarse="general", fine="unknown")
        if "product" in text.lower():
            return Intent(coarse="products", fine="top_products")
        return Intent(coarse="general", fine="greeting")
    mock_learner.classify_intent.side_effect = mock_classify
    
    mock_learner.assess_query_complexity.return_value = "simple" 
    
    # Refinement Mock
    async def mock_refine(text, service):
        if "sales" in text.lower():
            return Intent(coarse="analytics", fine="revenue_summary")
        return None
    mock_learner.refine_intent_with_llm.side_effect = mock_refine

    # Real DB Operations (using the test db_session from conftest)
    # We need to instantiate DatabaseOperations with a session factory that returns our test session
    from src.database.operations import DatabaseOperations
    
    # Patch get_session to return our test fixture session
    # Note: DatabaseOperations usually creates a new session per call. 
    # For testing, we want it to use the SAME sqlite in-memory session or a transaction-bound one.
    # The simplest way is to mock the internal session creation.
    mock_db_ops = DatabaseOperations("sqlite:///:memory:")
    mock_db_ops.get_session = MagicMock(return_value=db_session)

    # Mock Session Manager
    mock_session = MagicMock()
    mock_session.id = 1
    mock_session_manager = MagicMock()
    mock_session_manager.get_or_create_session.return_value = mock_session

    # Initialize Handler
    handler = MessageHandler(
        claude_service=mock_claude,
        pattern_learner=mock_learner,
        preference_manager=MagicMock(),
        db_ops=mock_db_ops,
        session_manager=mock_session_manager,
        feedback_analyzer=None,
        template_manager=None,
        insight_aggregator=None
    )
    
    # Mock allowed user check - clear list to allow verify-all (dev mode behavior)
    handler._allowed_users = []
    print(f"DEBUG: handler._allowed_users set to: {handler._allowed_users}")

    # --- 2. Simulation Step A: First Interaction (Greeting) ---
    print("\n🔹 Step A: User says 'Hi'")
    
    # Create User in DB first (since we mocked get_session, we can use db_session directly to check)
    user = User(telegram_user_id=12345, telegram_username="test_user", first_name="Test", is_verified=True)
    db_session.add(user)
    db_session.commit()
    
    # Add Store (required for processing)
    if not user.id: # db_session.add(user) might not populate id if not refreshed?
         db_session.refresh(user)
         
    db_ops.add_store(
        user_id=user.id,
        shop_domain="test-shop.myshopify.com",
        access_token="test_token_12345"
    )
    
    # Mock Telegram Update
    update_a = MagicMock()
    update_a.effective_user.id = 12345
    update_a.message.text = "Hi"
    update_a.message.reply_text = AsyncMock()
    context_a = MagicMock()
    context_a.bot.send_chat_action = AsyncMock()

    await handler.handle_message(update_a, context_a)

    # Assertions A
    assert update_a.message.reply_text.called
    # Check if conversation was saved
    convs = db_session.query(Conversation).filter_by(user_id=user.id).all()
    assert len(convs) == 1
    assert convs[0].message == "Hi"
    print("✅ Step A Passed: Greeting processed and saved.")

    # --- 3. Simulation Step B: Ambiguous Query (Intelligence Check) ---
    print("\n🔹 Step B: User says 'How are sales?' (Ambiguous Intent)")

    update_b = MagicMock()
    update_b.effective_user.id = 12345
    update_b.message.text = "How are sales?"
    update_b.message.reply_text = AsyncMock()
    
    await handler.handle_message(update_b, context_a)

    # Assertions B:
    # 1. verify refine_intent_with_llm was called
    mock_learner.refine_intent_with_llm.assert_called()
    # 2. verify process_message called with refined intent
    call_args = mock_claude.process_message.call_args
    # Check kwargs
    passed_intent = call_args.kwargs.get('intent')
    assert passed_intent.coarse == "analytics"  # Was refined from "general"
    
    # 3. Check DB for 2nd conversation
    convs = db_session.query(Conversation).filter_by(user_id=user.id).all()
    assert len(convs) == 2
    print("✅ Step B Passed: Ambiguous query refined and processed.")

    # --- 4. Simulation Step C: Specific Query (Context Check) ---
    print("\n🔹 Step C: User says 'Top products' (Specific Intent)")
    
    # Reset mocks
    mock_learner.refine_intent_with_llm.reset_mock()
    
    update_c = MagicMock()
    update_c.effective_user.id = 12345
    update_c.message.text = "Show top products"
    update_c.message.reply_text = AsyncMock()

    await handler.handle_message(update_c, context_a)

    # Assertions C:
    # 1. refine_intent_with_llm should NOT be called (intent is clear)
    mock_learner.refine_intent_with_llm.assert_not_called()
    # 2. Response should be about products
    args, _ = update_c.message.reply_text.call_args
    assert "Top product" in args[0] or "Widget A" in args[0] # Note: MessageHandler often sends chunks, might need to check call_args_list
    
    print("✅ Step C Passed: Specific query handled directly.")

if __name__ == "__main__":
    # Allow running directly for quick check
    pass
