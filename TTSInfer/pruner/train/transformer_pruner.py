from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

import math


class SinusoidalPosEmb(nn.Module):
    """"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerPruner(nn.Module):
    """
    Transformer Decoder× → [logit_reuse, logit_compute]

    - step embedding + block embedding  tgt memory
    -  Transformer Decoder  self-attention  cross-attention
    -  head  3  logits
    - forward(step_id, block_names, context=None) -> logits [batch, 1, blocks, 3]
    """

    def __init__(
        self,
        max_steps: int,
        block_names: List[str],
        hidden_dim: int = 512,
        decoder_layers: int = 1,
        block_encoder_type: str = "SA",
        attn_heads: int = 8,
        dim_feedforward: int = 1024,
        obs_dim: int = 0,
        dropout: float = 0.1,
        reuse_block: bool = False,
        tgt_sa: Optional[torch.Tensor] = None,
        head_4: bool = False,
        head_2: bool = False,
        time_emb=None,
        position_embeddings=None,
        n_obs_steps: int = 1,
        encoder=None,
        obs_encoder=None,
        obs_emb=None,
        training: bool = False,
        reuse_dp_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.block_names = list(block_names)
        self.hidden_dim = hidden_dim
        self.decoder_layers = decoder_layers
        self.attn_heads = attn_heads
        self.dim_feedforward = dim_feedforward
        self.obs_dim = obs_dim
        self.dropout = dropout
        self.reuse_block = reuse_block
        self.tgt_sa = tgt_sa
        self.block_to_id: Dict[str, int] = {name: i for i, name in enumerate(self.block_names)}

        # Block Embeddings
        self.block_emb = SinusoidalPosEmb(hidden_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attn_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=decoder_layers
        )
        
        # Memory projection: DP encoder  -> pruner hidden_dim
        cond_dim = obs_dim 
        self.memory_proj = nn.Linear(cond_dim, hidden_dim)

        self.safe_block_names: Dict[str, str] = {name: name.replace(".", "_") for name in self.block_names}
        self.safe_to_original: Dict[str, str] = {safe: orig for orig, safe in self.safe_block_names.items()}

        # Output Head
        if head_2:
            self.head = nn.Linear(hidden_dim, 2)
        elif head_4:
            self.head = nn.Linear(hidden_dim, 4)
        else:
            self.head = nn.Linear(hidden_dim, 3)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """"""
        if hasattr(self, 'head'):
            self._init_mlp(self.head)
        if hasattr(self, 'memory_proj'):
            self._init_mlp(self.memory_proj)
    
    def _init_mlp(self, mlp_module):
        """MLP"""
        for module in mlp_module.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.no_grad()
    def _get_block_ids(self, block_names: List[str]) -> torch.Tensor:
        ids = [self.block_to_id[b] for b in block_names]
        return torch.tensor(ids, dtype=torch.long, device=next(self.parameters()).device)

    def forward(
        self,
        step_id: int,
        block_names: List[str],
        context: Any = None,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transformer Decoder based forward pass.
        
        wrapper  pruner memory_proj, block_emb, 
        transformer_decoder, head forward 

        Args:
            step_id: Single timestep ID (0 to max_steps-1)
            block_names: List of block names
            context: 
            memory: DP  encoder  [batch_size, seq_len, obs_dim]

        Returns:
            logits: [batch_size, 1, num_blocks, 3/4] - gates for single step
        """
        assert step_id < self.max_steps, f"step_id({step_id}) >= max_steps({self.max_steps})"
        assert memory is not None, " memory  DP  encoder "

        batch_size = memory.shape[0]
        
        # 1. Project memory to hidden_dim
        memory = self.memory_proj(memory)  # [batch_size, seq_len, hidden_dim]

        # 2. Prepare target embeddings
        if self.reuse_block:
            assert self.tgt_sa is not None, "reuse_block=True  tgt_sa "
            #  transformer decoder  cross-attention  feedforward
            decoder_output = self.tgt_sa
            for mod in self.transformer_decoder.layers:
                x = decoder_output
                if mod.norm_first:
                    x = x + mod._mha_block(mod.norm2(x), memory, None, None)
                    x = x + mod._ff_block(mod.norm3(x))
                else:
                    x = mod.norm2(x + mod._mha_block(x, memory, None, None))
                    x = mod.norm3(x + mod._ff_block(x))
                decoder_output = x
        else:
            # Block embeddings
            block_ids = self._get_block_ids(block_names)  # [B]
            block_emb = self.block_emb(block_ids)         # [B, hidden_dim]
            tgt = block_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, B, hidden_dim]

            # Apply Transformer Decoder
            decoder_output = self.transformer_decoder(tgt, memory)  # [batch_size, B, hidden_dim]

        # 3. Generate output logits
        logits = self.head(decoder_output)  # [batch_size, B, 3/4]
        logits = logits.unsqueeze(1)        # [batch_size, 1, B, 3/4]

        return logits
           



def enumerate_decoder_block_keys(layer_names: List[str]) -> List[str]:
    keys: List[str] = []
    for layer_name in layer_names:
        keys.append(f"{layer_name}_sa_block")
        keys.append(f"{layer_name}_mha_block")
        keys.append(f"{layer_name}_ff_block")
    return keys