#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_import(module_name, description):
    """Test importing a module"""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {description}: Failed to import {module_name} - {e}")
        return False
    except Exception as e:
        print(f"✗ {description}: Error importing {module_name} - {e}")
        return False

def main():
    print("Testing imports for Continuous Learning Environment")
    print("=" * 60)

    # Core dependencies
    core_deps = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('json', 'JSON'),
        ('os', 'OS'),
        ('sys', 'System'),
        ('time', 'Time'),
        ('threading', 'Threading'),
        ('asyncio', 'AsyncIO'),
        ('typing', 'Typing'),
        ('dataclasses', 'DataClasses'),
        ('pathlib', 'PathLib'),
    ]

    # Optional dependencies
    optional_deps = [
        ('transformers', 'HuggingFace Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('requests', 'Requests'),
        ('tqdm', 'TQDM'),
        ('wandb', 'Weights & Biases'),
        ('pydantic', 'Pydantic'),
        ('omegaconf', 'OmegaConf'),
        ('hydra', 'Hydra'),
    ]

    # Local modules
    local_modules = [
        ('gemini_integration', 'Gemini API Integration'),
        ('multi_agent_system', 'Multi-Agent System'),
        ('cross_modal_transfer', 'Cross-Modal Transfer'),
        ('multi_language_support', 'Multi-Language Support'),
        ('architecture_evolution', 'Architecture Evolution'),
        ('continuous_learning_system', 'Continuous Learning System'),
        ('text_dataset', 'Text Dataset'),
        ('puzzle_dataset', 'Puzzle Dataset'),
    ]

    print("\nCORE DEPENDENCIES:")
    core_results = []
    for module, desc in core_deps:
        core_results.append(test_import(module, desc))

    print("\nOPTIONAL DEPENDENCIES:")
    optional_results = []
    for module, desc in optional_deps:
        optional_results.append(test_import(module, desc))

    print("\nLOCAL MODULES:")
    local_results = []
    for module, desc in local_modules:
        local_results.append(test_import(module, desc))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Core dependencies: {sum(core_results)}/{len(core_results)} working")
    print(f"Optional dependencies: {sum(optional_results)}/{len(optional_results)} working")
    print(f"Local modules: {sum(local_results)}/{len(local_results)} working")

    total_working = sum(core_results) + sum(optional_results) + sum(local_results)
    total_total = len(core_results) + len(optional_results) + len(local_results)

    print(f"Overall: {total_working}/{total_total} modules working")

    if sum(core_results) < len(core_results):
        print("\n❌ CRITICAL: Some core dependencies are missing!")
        return False

    if sum(local_results) < len(local_results):
        print("\n⚠️  WARNING: Some local modules have import issues")
        return False

    print("\n✅ All critical components working!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
