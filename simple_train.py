#!/usr/bin/env python3
"""
Simple TRM training script using standard PyTorch optimizers
"""

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from gemini_integration import GeminiConfig, GeminiAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_batch(batch_size=4, seq_len=256, vocab_size=1000, num_puzzles=100, puzzle_emb_len=16):
    """Create a synthetic training batch for testing"""
    # The model expects inputs to be just the text tokens, puzzle embeddings are added internally
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    puzzle_identifiers = torch.randint(0, num_puzzles, (batch_size,))

    return {
        'inputs': inputs,
        'targets': targets,
        'puzzle_identifiers': puzzle_identifiers
    }

def train_step(model, batch, optimizer, device, carry=None):
    """Single training step"""
    model.train()

    # Move to device
    inputs = batch['inputs'].to(device)
    targets = batch['targets'].to(device)
    puzzle_identifiers = batch['puzzle_identifiers'].to(device)

    # Initialize carry
    carry = model.initial_carry({'inputs': inputs, 'puzzle_identifiers': puzzle_identifiers})

    # Move carry to device (comprehensive recursive move)
    def move_to_device_recursive(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device_recursive(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device_recursive(item, device) for item in obj)
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses and custom objects
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if isinstance(attr_value, torch.Tensor):
                            setattr(obj, attr_name, attr_value.to(device))
                        elif isinstance(attr_value, (dict, list, tuple)) or hasattr(attr_value, '__dict__'):
                            setattr(obj, attr_name, move_to_device_recursive(attr_value, device))
                    except:
                        pass  # Skip attributes that can't be set
            return obj
        else:
            return obj

    # Initialize carry if this is the first step
    if carry is None:
        carry = model.initial_carry({'inputs': inputs, 'puzzle_identifiers': puzzle_identifiers})
        print(f"Initial carry devices - halted: {carry.halted.device}, steps: {carry.steps.device}")
        carry = move_to_device_recursive(carry, device)
        print(f"After move - halted: {carry.halted.device}, steps: {carry.steps.device}")

    # Forward pass
    try:
        new_carry, outputs = model(carry, {'inputs': inputs, 'puzzle_identifiers': puzzle_identifiers})
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print(f"Input shapes: inputs={inputs.shape}, puzzle_ids={puzzle_identifiers.shape}")
        print(f"Carry halted: {carry.halted.shape}, device: {carry.halted.device}")
        print(f"Carry steps: {carry.steps.shape}, device: {carry.steps.device}")
        raise

    # Simple language modeling loss (predict next token)
    logits = outputs['logits']  # [batch_size, seq_len, vocab_size]

    # The logits should be [batch_size, seq_len, vocab_size], but we need to handle the puzzle embedding offset
    # The puzzle embeddings are prepended, so we take from puzzle_emb_len onwards
    puzzle_emb_len = 16  # from config
    if logits.shape[1] > puzzle_emb_len:
        logits = logits[:, puzzle_emb_len:]  # Remove puzzle embedding predictions

    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
        ignore_index=-100
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), new_carry

def main():
    print("🚀 Starting Simple TRM Training")
    print("=" * 50)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Using device: {device}")

    # Model config
    config = {
        'batch_size': 4,
        'seq_len': 256,
        'puzzle_emb_ndim': 512,
        'num_puzzle_identifiers': 100,
        'vocab_size': 1000,
        'H_cycles': 3,
        'L_cycles': 4,
        'H_layers': 0,
        'L_layers': 2,
        'hidden_size': 512,
        'expansion': 4,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 16,
        'halt_exploration_prob': 0.1,
        'mlp_t': False,
        'puzzle_emb_len': 16,
        'no_ACT_continue': True
    }

    # Create model
    print("🏗️  Creating TRM model...")
    model = TinyRecursiveReasoningModel_ACTV1(config)

    # Move model to device and ensure ALL tensors are on the correct device
    model = model.to(device)

    # Explicitly move all parameters and buffers to device
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)

    for name, buffer in model.named_buffers():
        if buffer.device != device:
            # Use setattr to update the buffer
            parent_module = model
            attr_chain = name.split('.')
            for attr in attr_chain[:-1]:
                parent_module = getattr(parent_module, attr)
            setattr(parent_module, attr_chain[-1], buffer.to(device))

    # Verify all tensors are on CUDA
    all_on_cuda = True
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Warning: {name} still on {param.device}")
            all_on_cuda = False
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            print(f"Warning: buffer {name} still on {buffer.device}")
            all_on_cuda = False

    if all_on_cuda:
        print("✅ All model tensors moved to CUDA")
    else:
        print("⚠️  Some tensors may still be on CPU")

    print(f"📊 Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    # Optimizer (using standard Adam instead of AdamATan2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    print("⚡ Using Adam optimizer (standard)")

    # Training loop
    num_steps = 100
    log_interval = 10
    carry = None  # Initialize carry

    print(f"🎯 Training for {num_steps} steps...")
    print("-" * 30)

    for step in range(num_steps):
        # Create synthetic batch
        batch = create_synthetic_batch()

        # Training step
        loss, carry = train_step(model, batch, optimizer, device, carry)

        # Log progress
        if (step + 1) % log_interval == 0:
            print(".4f")

    print("✅ Training completed!")
    print(".4f")

    # Save model
    os.makedirs('checkpoints/simple_training', exist_ok=True)
    checkpoint_path = f'checkpoints/simple_training/trm_step_{num_steps}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': num_steps,
        'config': config,
        'final_loss': loss
    }, checkpoint_path)
    print(f"💾 Model saved to: {checkpoint_path}")

    # Test inference
    print("🧪 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_batch = create_synthetic_batch(batch_size=1)
        test_inputs = test_batch['inputs'].to(device)
        test_puzzle_ids = test_batch['puzzle_identifiers'].to(device)

        carry = model.initial_carry({'inputs': test_inputs, 'puzzle_identifiers': test_puzzle_ids})
        outputs, _ = model(carry, {'inputs': test_inputs, 'puzzle_identifiers': test_puzzle_ids})

        print(f"📤 Inference output shape: {outputs['logits'].shape}")
        print("✅ Inference successful!")

if __name__ == "__main__":
    main()
