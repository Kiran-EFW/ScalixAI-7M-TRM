"""
Text-based Tiny Recursive Model for language tasks
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TextTRM_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TextTRM_ACTV1Carry:
    inner_carry: TextTRM_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]

class TextTRM_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Text-specific additions
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # Language modeling specific
    tie_embeddings: bool = True  # Tie input and output embeddings

class TextTRM_ACTV1Block(nn.Module):
    def __init__(self, config: TextTRM_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=True  # Causal attention for language modeling
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention with causal masking
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)

        # MLP
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states

class TextTRM_ACTV1(nn.Module):
    def __init__(self, config: TextTRM_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len

        # Embeddings
        self.token_emb = CastedEmbedding(config.vocab_size, config.hidden_size)
        self.pos_emb = RotaryEmbedding(config.rope_theta, config.seq_len, config.hidden_size // config.num_heads)

        # Puzzle embeddings (for multimodal compatibility)
        self.puzzle_emb = CastedSparseEmbedding(
            num_embeddings=config.num_puzzle_identifiers,
            embedding_dim=config.puzzle_emb_ndim,
            length=config.puzzle_emb_len
        ) if config.puzzle_emb_ndim > 0 else None

        # Recursive layers
        self.H_layers = nn.ModuleList([
            TextTRM_ACTV1Block(config) for _ in range(config.H_layers)
        ])
        self.L_layers = nn.ModuleList([
            TextTRM_ACTV1Block(config) for _ in range(config.L_layers)
        ])

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if requested
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, CastedLinear):
            trunc_normal_init_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, CastedEmbedding):
            trunc_normal_init_(module.weight, std=0.02)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TextTRM_ACTV1Carry:
        """Initialize the carry state for recursive reasoning"""
        batch_size = batch['input_ids'].shape[0]

        # Get puzzle embeddings if available
        puzzle_emb = None
        if self.puzzle_emb is not None and 'puzzle_identifier' in batch:
            puzzle_emb = self.puzzle_emb(batch['puzzle_identifier'])

        # Initialize latent states
        z_H = torch.zeros(batch_size, self.seq_len, self.config.hidden_size, dtype=getattr(torch, self.config.forward_dtype))
        z_L = torch.zeros(batch_size, self.seq_len, self.config.hidden_size, dtype=getattr(torch, self.config.forward_dtype))

        # Add puzzle embedding to latent state if available
        if puzzle_emb is not None:
            z_L = z_L + puzzle_emb.unsqueeze(1).expand(-1, self.seq_len, -1)

        inner_carry = TextTRM_ACTV1InnerCarry(z_H=z_H, z_L=z_L)

        steps = torch.zeros(batch_size, dtype=torch.int32)
        halted = torch.zeros(batch_size, dtype=torch.bool)

        current_data = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids'])),
            'labels': batch.get('labels', batch['input_ids'].clone())
        }

        return TextTRM_ACTV1Carry(
            inner_carry=inner_carry,
            steps=steps,
            halted=halted,
            current_data=current_data
        )

    def forward(self, carry: TextTRM_ACTV1Carry, batch: Dict[str, torch.Tensor], return_keys: List[str] = None) -> Tuple[TextTRM_ACTV1Carry, torch.Tensor, Dict, Dict, bool]:
        """Forward pass with recursive reasoning"""
        if return_keys is None:
            return_keys = []

        # Get current data
        input_ids = carry.current_data['input_ids']
        attention_mask = carry.current_data['attention_mask']
        labels = carry.current_data['labels']

        # Embed tokens
        hidden_states = self.token_emb(input_ids)

        # Add positional embeddings
        cos_sin = self.pos_emb.get_cos_sin(hidden_states.shape[1], hidden_states.device)
        hidden_states = self.pos_emb.apply_rotary_emb(hidden_states, cos_sin)

        # Apply attention mask for causal modeling
        if attention_mask is not None:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(hidden_states.shape[1], hidden_states.shape[1]), diagonal=1).bool()
            causal_mask = causal_mask.to(hidden_states.device)
            # Apply mask to attention
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        # Recursive reasoning cycles
        z_H = carry.inner_carry.z_H
        z_L = carry.inner_carry.z_L

        # H cycles (High-level reasoning)
        for _ in range(self.config.H_cycles):
            for layer in self.H_layers:
                z_H = layer(cos_sin, z_H)

        # L cycles (Low-level reasoning)
        for _ in range(self.config.L_cycles):
            for layer in self.L_layers:
                z_L = layer(cos_sin, z_L)

        # Combine H and L representations
        combined = z_H + z_L

        # Generate predictions
        logits = self.lm_head(combined)

        # Calculate loss (language modeling loss)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
            reduction='mean'
        )

        # Calculate accuracy metrics
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels).float()
        mask = (shift_labels != IGNORE_LABEL_ID).float()
        accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'count': mask.sum().item()
        }

        # Update carry for next step
        new_carry = copy.deepcopy(carry)
        new_carry.inner_carry.z_H = z_H
        new_carry.inner_carry.z_L = z_L
        new_carry.steps += 1

        # Check halting condition
        all_finished = True  # For language modeling, we typically do one pass

        # Prepare outputs
        preds_dict = {}
        if 'predictions' in return_keys:
            preds_dict['predictions'] = logits.argmax(dim=-1)
        if 'logits' in return_keys:
            preds_dict['logits'] = logits

        return new_carry, loss, metrics, preds_dict, all_finished

def create_text_trm_model(config_dict: Dict) -> TextTRM_ACTV1:
    """Factory function to create TextTRM model"""
    config = TextTRM_ACTV1Config(**config_dict)
    return TextTRM_ACTV1(config)
