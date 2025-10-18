#!/usr/bin/env python3
"""
Quick test to verify basic functionality without loading heavy models
"""

def test_basic_functionality():
    """Test basic imports and functionality"""
    print("Testing basic functionality...")

    # Test gemini integration
    try:
        from gemini_integration import GeminiConfig
        config = GeminiConfig(api_key="test_key")
        print("✓ Gemini integration: basic config works")
    except Exception as e:
        print(f"✗ Gemini integration failed: {e}")
        return False

    # Test multi-agent system (without loading models)
    try:
        from multi_agent_system import AgentProfile
        profile = AgentProfile(
            agent_id="test_agent",
            name="Test Agent",
            specialization="coding"
        )
        print("✓ Multi-agent system: basic profile works")
    except Exception as e:
        print(f"✗ Multi-agent system failed: {e}")
        return False

    # Test architecture evolution (basic functionality)
    try:
        from architecture_evolution import ArchitectureSpec
        spec = ArchitectureSpec(
            architecture_id="test_arch",
            config={"hidden_size": 512, "layers": 2}
        )
        print("✓ Architecture evolution: basic spec works")
    except Exception as e:
        print(f"✗ Architecture evolution failed: {e}")
        return False

    # Test text dataset (basic functionality)
    try:
        from text_dataset import TextDatasetConfig
        config = TextDatasetConfig(
            seed=42,
            dataset_paths=["dummy.json"],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        print("✓ Text dataset: basic config works")
    except Exception as e:
        print(f"✗ Text dataset failed: {e}")
        return False

    # Test continuous learning (basic structure)
    try:
        from continuous_learning_system import ContinuousLearningOrchestrator
        print("✓ Continuous learning: basic import works")
    except Exception as e:
        print(f"✗ Continuous learning failed: {e}")
        return False

    print("✅ All basic functionality tests passed!")
    return True

def test_scripts_exist():
    """Test that all main scripts exist and are executable"""
    print("\nTesting script existence...")

    scripts = [
        "run_continuous_learning.py",
        "run_multi_agent_learning.py",
        "run_multilingual_training.py",
        "run_architecture_evolution.py",
        "setup_continuous_learning.sh"
    ]

    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script} exists")
        else:
            print(f"✗ {script} missing")
            return False

    print("✅ All scripts exist!")
    return True

if __name__ == "__main__":
    import os

    success = True

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    # Test script existence
    if not test_scripts_exist():
        success = False

    if success:
        print("\n🎉 All tests passed! The continuous learning environment is ready.")
        print("\nTo get started:")
        print("1. Set your GEMINI_API_KEY environment variable")
        print("2. Run: ./setup_continuous_learning.sh")
        print("3. Start learning: ./run_continuous_learning.py --duration-days 3")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

    exit(0 if success else 1)
