#!/usr/bin/env python3
"""
Test UI compatibility after model capability updates.
Verifies that the Agent can be instantiated as the FastAPI backend does.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from medster.agent import Agent
from medster.model_capabilities import get_model_capability

def test_agent_instantiation():
    """Test that Agent can be created with just model_name (as API does)."""
    print("Testing Agent instantiation for UI compatibility...\n")

    # Test all three models that the UI supports
    models = ["gpt-oss:20b", "qwen3-vl:8b", "ministral-3:8b"]

    for model in models:
        print(f"Testing {model}:")
        try:
            # This is how the API creates agents
            agent = Agent(model_name=model)

            # Verify attributes exist
            assert hasattr(agent, 'model_name')
            assert hasattr(agent, 'model_capability')
            assert hasattr(agent, 'logger')

            # Verify capability is loaded
            capability = get_model_capability(model)
            assert agent.model_capability == capability

            print(f"  ✓ Agent created successfully")
            print(f"  ✓ Native tools: {capability.native_tools}")
            print(f"  ✓ Vision: {capability.vision}")
            print(f"  ✓ Tool strategy: {capability.tool_strategy.value}")
            print()

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    print("✓ All models compatible with UI!")
    return True

def test_websocket_callback_compatibility():
    """Test that StreamingCallback pattern still works."""
    print("\nTesting WebSocket callback compatibility...")

    try:
        # Simulate what api.py does
        agent = Agent(model_name="gpt-oss:20b")

        # Verify logger methods exist (used in api.py)
        assert hasattr(agent.logger, '_log')
        assert hasattr(agent.logger, 'log_task_start')
        assert hasattr(agent.logger, 'log_task_done')
        assert hasattr(agent.logger, 'log_tool_run')

        print("  ✓ Logger methods compatible")
        print("  ✓ WebSocket callbacks will work")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

if __name__ == '__main__':
    success = True
    success = test_agent_instantiation() and success
    success = test_websocket_callback_compatibility() and success

    print("\n" + "="*60)
    if success:
        print("✓ UI COMPATIBILITY: PASSED")
        print("  The FastAPI backend will work with all changes!")
    else:
        print("✗ UI COMPATIBILITY: FAILED")
        print("  Review errors above")
        sys.exit(1)
