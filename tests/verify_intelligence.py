import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot.handlers import MessageHandler
from src.learning.pattern_learner import Intent

async def verify_smart_intent_flow():
    print("🚀 Starting Intelligence Verification...")

    # Mock dependencies
    mock_claude_service = AsyncMock()
    mock_claude_service.process_message = AsyncMock(return_value="Response")
    
    mock_pattern_learner = MagicMock()
    # Simulate ambiguous intent
    mock_pattern_learner.classify_intent.return_value = Intent(coarse="general", fine="unknown")
    mock_pattern_learner.assess_query_complexity.return_value = "simple" # Let's test the "general" intent path
    
    # Mock refinement
    refined_intent = Intent(coarse="analytics", fine="sales_over_time")
    mock_pattern_learner.refine_intent_with_llm = AsyncMock(return_value=refined_intent)

    mock_db_ops = MagicMock()
    mock_db_ops.get_or_create_user.return_value = MagicMock(id=123)
    mock_db_ops.get_store_by_user.return_value = MagicMock(shop_domain="test.myshopify.com")

    # Initialize Handler
    handler = MessageHandler(
        claude_service=mock_claude_service,
        pattern_learner=mock_pattern_learner,
        preference_manager=MagicMock(),
        db_ops=mock_db_ops,
        session_manager=MagicMock(), # Mock session manager to avoid real DB calls
    )
    
    # Mock Telegram Update/Context
    mock_update = MagicMock()
    mock_update.effective_user.id = 123
    mock_update.message.text = "How are my sales doing?"
    mock_update.message.reply_text = AsyncMock() # Make reply_text awaitable
    
    mock_context = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock() # Make send_chat_action awaitable

    # Run Handler
    print("\n1️⃣  Testing Ambiguous Intent Handling...")
    await handler.handle_message(mock_update, mock_context)

    # Verification 1: Did it call refine_intent_with_llm?
    if mock_pattern_learner.refine_intent_with_llm.called:
        print("✅ SUCCESS: Ambiguous intent triggered LLM refinement.")
    else:
        print("❌ FAILURE: Ambiguous intent did NOT trigger LLM refinement.")

    # Verification 2: Did it pass the REFINED intent to Claude Service?
    # process_message call args: (user_id, message, intent, session_id)
    call_args = mock_claude_service.process_message.call_args
    if call_args:
        # Check kwargs since MessageHandler uses keyword arguments
        passed_intent = call_args.kwargs.get('intent')
        
        # We check if passed_intent has the refined attributes
        if passed_intent == refined_intent:
             print(f"✅ SUCCESS: Refined intent '{passed_intent.coarse}/{passed_intent.fine}' passed to LLMService.")
        else:
             print(f"❌ FAILURE: Wrong intent passed to LLMService. Got: {passed_intent}")
    else:
        print("❌ FAILURE: LLMService.process_message was not called.")

if __name__ == "__main__":
    asyncio.run(verify_smart_intent_flow())
