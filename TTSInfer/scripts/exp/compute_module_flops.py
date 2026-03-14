"""
Diffusion PolicyPrunerFLOPs

:
1. Diffusion Policy Encoder (100stepencoder)
2. Diffusion Policy Decoder (100step)
3. Pruner (steppruner)

:
python TTSInfer/scripts/compute_module_flops.py \
    --checkpoint checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
    --pruner_path sim_result/pruner_ckpt/20251030_223523/0/can_mh/pruner_model_18_0.0034_0.93.pt \
    --task_name can_mh \
    --device cuda:3 \
    --batch_size 1
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import click
import dill
import hydra
from omegaconf import OmegaConf
from typing import Dict, Any, Tuple
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# FLOPs
from thop import profile, clever_format

from TTSInfer.pruner.eval.eval_utils import (
    construct_obs_dict, load_workspace, load_pruner_model
)
from TTSInfer.pruner.train.transformer_pruner import enumerate_decoder_block_keys
from TTSInfer.pruner.train.gate_scheduler import apply_scheduler_single

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compute_module_flops")


def manual_attention_flops(seq_len_q: int, seq_len_kv: int, d_model: int, n_head: int) -> float:
    """
    MultiheadAttentionFLOPs
    
    Args:
        seq_len_q: query
        seq_len_kv: key/value
        d_model: 
        n_head: 
    
    Returns:
        total FLOPs
    """
    # Q, K, V projections
    flops_q = seq_len_q * d_model * d_model
    flops_k = seq_len_kv * d_model * d_model
    flops_v = seq_len_kv * d_model * d_model
    
    # Attention computation: Q @ K^T and softmax @ V
    d_k = d_model // n_head
    flops_qk = n_head * seq_len_q * seq_len_kv * d_k
    flops_av = n_head * seq_len_q * seq_len_kv * d_k
    
    # Output projection
    flops_out = seq_len_q * d_model * d_model
    
    total = flops_q + flops_k + flops_v + flops_qk + flops_av + flops_out
    return total


def manual_transformer_encoder_layer_flops(seq_len: int, d_model: int, n_head: int, dim_ff: int) -> float:
    """TransformerEncoderLayerFLOPs"""
    # Self-attention
    flops_sa = manual_attention_flops(seq_len, seq_len, d_model, n_head)
    # Feedforward
    flops_ff = seq_len * d_model * dim_ff + seq_len * dim_ff * d_model
    # LayerNorm ()
    return flops_sa + flops_ff


def manual_transformer_decoder_layer_flops(seq_len: int, memory_len: int, d_model: int, n_head: int, dim_ff: int) -> float:
    """TransformerDecoderLayerFLOPs"""
    # Self-attention
    flops_sa = manual_attention_flops(seq_len, seq_len, d_model, n_head)
    # Cross-attention
    flops_ca = manual_attention_flops(seq_len, memory_len, d_model, n_head)
    # Feedforward
    flops_ff = seq_len * d_model * dim_ff + seq_len * dim_ff * d_model
    # LayerNorm ()
    return flops_sa + flops_ca + flops_ff


def count_encoder_flops(
    model: nn.Module, 
    cond: torch.Tensor,
    num_steps: int = 100,
    device: torch.device = torch.device('cpu')
) -> float:
    """EncoderFLOPs"""
    logger.info(f"\n{'='*60}")
    logger.info(f"EncoderFLOPs")
    logger.info(f"{'='*60}")
    
    batch_size = cond.shape[0]
    total_flops = 0.0
    
    # 1. time_emb: num_steps
    logger.info(f"\n1. time_emb: {num_steps}")
    timesteps_batch = torch.arange(num_steps-1, -1, -1, dtype=torch.long, device=device)
    single_timestep = timesteps_batch[0:1]
    flops_time, _ = profile(model.time_emb, inputs=(single_timestep,), verbose=False)
    flops_time = flops_time * num_steps
    logger.info(f"   time_emb FLOPs: {flops_time/1e9:.4f} GFLOPs")
    total_flops += flops_time
    
    # 2. cond_obs_emb: 1
    if model.obs_as_cond and hasattr(model, 'cond_obs_emb'):
        logger.info(f"\n2. cond_obs_emb: 1")
        flops_cond, _ = profile(model.cond_obs_emb, inputs=(cond,), verbose=False)
        logger.info(f"   cond_obs_emb FLOPs: {flops_cond/1e9:.4f} GFLOPs")
        total_flops += flops_cond
    
    # 3. encoder: batch_size * num_steps
    logger.info(f"\n3. encoder: {batch_size * num_steps}")
    with torch.no_grad():
        if model.obs_as_cond and hasattr(model, 'cond_obs_emb'):
            time_emb_batch = model.time_emb(timesteps_batch)
            time_emb_batch = time_emb_batch.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(2)
            cond_obs_emb = model.cond_obs_emb(cond)
            cond_obs_emb_batch = cond_obs_emb.unsqueeze(1).expand(-1, num_steps, -1, -1)
            cond_embeddings_batch = torch.cat([time_emb_batch, cond_obs_emb_batch], dim=2)
            bs, ns, seq_len, hidden_dim = cond_embeddings_batch.shape
            cond_embeddings_flat = cond_embeddings_batch.view(bs * ns, seq_len, hidden_dim)
        else:
            time_emb_batch = model.time_emb(timesteps_batch)
            time_emb_batch = time_emb_batch.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(2)
            bs, ns, seq_len, hidden_dim = time_emb_batch.shape
            cond_embeddings_flat = time_emb_batch.view(bs * ns, seq_len, hidden_dim)
        
        tc = cond_embeddings_flat.shape[1]
        position_embeddings = model.cond_pos_emb[:, :tc, :]
        encoder_input = model.drop(cond_embeddings_flat + position_embeddings)
    
    # encoderFLOPs
    if isinstance(model.encoder, nn.TransformerEncoder):
        # TransformerEncoder:
        seq_len = encoder_input.shape[1]
        d_model = encoder_input.shape[2]
        first_layer = model.encoder.layers[0]
        n_head = first_layer.self_attn.num_heads
        dim_ff = first_layer.linear1.out_features
        n_encoder_layers = len(model.encoder.layers)
        
        logger.info(f"   Encoder: TransformerEncoder")
        logger.info(f"   Config: seq_len={seq_len}, d_model={d_model}, n_head={n_head}, dim_ff={dim_ff}, n_layers={n_encoder_layers}")
        
        # FLOPs
        flops_encoder_layer_single = manual_transformer_encoder_layer_flops(seq_len, d_model, n_head, dim_ff)
        flops_encoder_single = flops_encoder_layer_single * n_encoder_layers
        flops_encoder = flops_encoder_single * (batch_size * num_steps)
        
        logger.info(f"   encoder FLOPs: {flops_encoder_layer_single/1e6:.2f} MFLOPs")
        logger.info(f"   encoder FLOPs: {flops_encoder_single/1e6:.2f} MFLOPs")
    else:
        # TransformerSequentialthop
        logger.info(f"   Encoder: {type(model.encoder).__name__}")
        single_encoder_input = encoder_input[0:1]
        flops_encoder_single, _ = profile(model.encoder, inputs=(single_encoder_input,), verbose=False)
        flops_encoder = flops_encoder_single * (batch_size * num_steps)
    
    logger.info(f"   encoderFLOPs: {flops_encoder/1e9:.4f} GFLOPs")
    total_flops += flops_encoder
    
    logger.info(f"\nEncoder: {total_flops/1e9:.4f} GFLOPs")
    return total_flops


def count_decoder_base_flops(
    model: nn.Module,
    sample: torch.Tensor,
    memory: torch.Tensor,
    num_steps: int = 100,
    n_layers: int = 8,
) -> Tuple[float, Dict[str, float]]:
    """Base DecoderFLOPscache"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Base Decoder FLOPs (100)")
    logger.info(f"{'='*60}")
    
    # decoder
    with torch.no_grad():
        input_emb = model.input_emb(sample)
        t = input_emb.shape[1]
        position_embeddings = model.pos_emb[:, :t, :]
        x = model.drop(input_emb + position_embeddings)
    
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    memory_len = memory.shape[1]
    d_model = x.shape[2]
    
    first_layer = model.decoder.layers[0]
    n_head = first_layer.self_attn.num_heads
    
    logger.info(f"\nConfig:")
    logger.info(f"  d_model={d_model}, n_head={n_head}")
    logger.info(f"  seq_len={seq_len}, memory_len={memory_len}")
    
    # blockFLOPs
    # SA block: self-attention + LayerNorm + dropout
    flops_sa_attn = manual_attention_flops(seq_len, seq_len, d_model, n_head)
    flops_sa_norm = seq_len * d_model  # LayerNormFLOPsd*seq_len
    flops_sa_single = flops_sa_attn + flops_sa_norm
    
    # MHA block: cross-attention + LayerNorm + dropout
    flops_mha_attn = manual_attention_flops(seq_len, memory_len, d_model, n_head)
    flops_mha_norm = seq_len * d_model
    flops_mha_single = flops_mha_attn + flops_mha_norm
    
    # FF block: thopthopLinear
    class FFBlockWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, x):
            return self.layer._ff_block(self.layer.norm3(x))
    
    with torch.no_grad():
        x_after_sa = x + first_layer._sa_block(first_layer.norm1(x), None, None)
        x_after_mha = x_after_sa + first_layer._mha_block(first_layer.norm2(x_after_sa), memory, None, None)
    
    ff_wrapper = FFBlockWrapper(first_layer)
    flops_ff_single, _ = profile(ff_wrapper, inputs=(x_after_mha,), verbose=False)
    
    logger.info(f"\nlayerstepblock FLOPs:")
    logger.info(f"  SA block:  {flops_sa_single/1e6:.2f} MFLOPs")
    logger.info(f"  MHA block: {flops_mha_single/1e6:.2f} MFLOPs")
    logger.info(f"  FF block:  {flops_ff_single/1e6:.2f} MFLOPs")
    
    # decoder blocks FLOPs
    flops_sa_total = flops_sa_single * num_steps * n_layers
    flops_mha_total = flops_mha_single * num_steps * n_layers
    flops_ff_total = flops_ff_single * num_steps * n_layers
    
    logger.info(f"\nDecoder blocks FLOPs ({num_steps} steps × {n_layers} layers):")
    logger.info(f"  SA blocks:  {flops_sa_total/1e9:.4f} GFLOPs")
    logger.info(f"  MHA blocks: {flops_mha_total/1e9:.4f} GFLOPs")
    logger.info(f"  FF blocks:  {flops_ff_total/1e9:.4f} GFLOPs")
    
    logger.info(f"\n:")
    
    # 1. input_emb
    flops_input_emb, _ = profile(model.input_emb, inputs=(sample,), verbose=False)
    flops_input_emb = flops_input_emb * num_steps
    logger.info(f"  input_emb: {flops_input_emb/1e9:.4f} GFLOPs")
    
    # 2. ln_f
    with torch.no_grad():
        decoder_out = x_after_mha  # 
    flops_ln_f, _ = profile(model.ln_f, inputs=(decoder_out,), verbose=False)
    flops_ln_f = flops_ln_f * num_steps
    logger.info(f"  ln_f:      {flops_ln_f/1e9:.4f} GFLOPs")
    
    # 3. head
    with torch.no_grad():
        ln_out = model.ln_f(decoder_out)
    flops_head, _ = profile(model.head, inputs=(ln_out,), verbose=False)
    flops_head = flops_head * num_steps
    logger.info(f"  head:      {flops_head/1e9:.4f} GFLOPs")
    
    total_flops = flops_input_emb + flops_sa_total + flops_mha_total + flops_ff_total + flops_ln_f + flops_head
    
    breakdown = {
        'input_emb': flops_input_emb,
        'sa_blocks': flops_sa_total,
        'mha_blocks': flops_mha_total,
        'ff_blocks': flops_ff_total,
        'ln_f': flops_ln_f,
        'head': flops_head
    }
    
    logger.info(f"\nBase Decoder: {total_flops/1e9:.4f} GFLOPs")
    return total_flops, breakdown


def count_decoder_pruned_flops(
    model: nn.Module,
    sample: torch.Tensor,
    memory: torch.Tensor,
    strategy_dict: Dict[int, int],
    num_steps: int = 100,
    n_layers: int = 8,
) -> Tuple[float, Dict[str, float]]:
    """
    Pruned DecoderFLOPscache
    
    strategy_dictFLOPs:
    - strategy 0:  (100% FLOPs)
    - strategy 1: 3cache (0% FLOPs)
    - strategy 2: 24cache (0% FLOPs)
    - strategy 3: rollout cache (0% FLOPs, )
    
    Returns:
        total_flops: FLOPs
        breakdown: FLOPs
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Pruned Decoder FLOPs (cache)")
    logger.info(f"{'='*60}")
    
    # decoder
    with torch.no_grad():
        input_emb = model.input_emb(sample)
        t = input_emb.shape[1]
        position_embeddings = model.pos_emb[:, :t, :]
        x = model.drop(input_emb + position_embeddings)
    
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    memory_len = memory.shape[1]
    d_model = x.shape[2]
    
    first_layer = model.decoder.layers[0]
    n_head = first_layer.self_attn.num_heads
    
    logger.info(f"\nConfig:")
    logger.info(f"  d_model={d_model}, n_head={n_head}")
    logger.info(f"  seq_len={seq_len}, memory_len={memory_len}")
    
    # blockFLOPs
    # SA block: self-attention + LayerNorm + dropout
    flops_sa_attn = manual_attention_flops(seq_len, seq_len, d_model, n_head)
    flops_sa_norm = seq_len * d_model
    flops_sa = flops_sa_attn + flops_sa_norm
    
    # MHA block: cross-attention + LayerNorm + dropout
    flops_mha_attn = manual_attention_flops(seq_len, memory_len, d_model, n_head)
    flops_mha_norm = seq_len * d_model
    flops_mha = flops_mha_attn + flops_mha_norm
    
    # FF block: thopthopLinear
    class FFBlockWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, x):
            return self.layer._ff_block(self.layer.norm3(x))
    
    with torch.no_grad():
        x_after_sa = x + first_layer._sa_block(first_layer.norm1(x), None, None)
        x_after_mha = x_after_sa + first_layer._mha_block(first_layer.norm2(x_after_sa), memory, None, None)
    
    ff_wrapper = FFBlockWrapper(first_layer)
    flops_ff, _ = profile(ff_wrapper, inputs=(x_after_mha,), verbose=False)
    
    logger.info(f"\nblockFLOPs:")
    logger.info(f"  SA block:  {flops_sa/1e6:.2f} MFLOPs")
    logger.info(f"  MHA block: {flops_mha/1e6:.2f} MFLOPs")
    logger.info(f"  FF block:  {flops_ff/1e6:.2f} MFLOPs")
    
    # strategy_dict
    strategy_counts = {
        'sa': {0: 0, 1: 0, 2: 0, 3: 0},
        'mha': {0: 0, 1: 0, 2: 0, 3: 0},
        'ff': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    for step in range(num_steps):
        for layer_idx in range(n_layers):
            sa_idx = step * (n_layers * 3) + layer_idx * 3 + 0
            mha_idx = step * (n_layers * 3) + layer_idx * 3 + 1
            ff_idx = step * (n_layers * 3) + layer_idx * 3 + 2
            
            sa_strategy = strategy_dict.get(sa_idx, 0)
            mha_strategy = strategy_dict.get(mha_idx, 0)
            ff_strategy = strategy_dict.get(ff_idx, 0)
            
            strategy_counts['sa'][sa_strategy] += 1
            strategy_counts['mha'][mha_strategy] += 1
            strategy_counts['ff'][ff_strategy] += 1
    
    logger.info(f"\n:")
    logger.info(f"  SA blocks  - compute:{strategy_counts['sa'][0]}, 3cache:{strategy_counts['sa'][1]}, 24cache:{strategy_counts['sa'][2]}, rollout:{strategy_counts['sa'].get(3,0)}")
    logger.info(f"  MHA blocks - compute:{strategy_counts['mha'][0]}, 3cache:{strategy_counts['mha'][1]}, 24cache:{strategy_counts['mha'][2]}, rollout:{strategy_counts['mha'].get(3,0)}")
    logger.info(f"  FF blocks  - compute:{strategy_counts['ff'][0]}, 3cache:{strategy_counts['ff'][1]}, 24cache:{strategy_counts['ff'][2]}, rollout:{strategy_counts['ff'].get(3,0)}")
    
    # strategy 0computeFLOPs
    flops_sa_total = flops_sa * strategy_counts['sa'][0]
    flops_mha_total = flops_mha * strategy_counts['mha'][0]
    flops_ff_total = flops_ff * strategy_counts['ff'][0]
    
    logger.info(f"\nFLOPs:")
    logger.info(f"  SA blocks:  {flops_sa_total/1e9:.4f} GFLOPs")
    logger.info(f"  MHA blocks: {flops_mha_total/1e9:.4f} GFLOPs")
    logger.info(f"  FF blocks:  {flops_ff_total/1e9:.4f} GFLOPs")
    
    # decoderinput_emb, ln_f, head
    flops_input_emb, _ = profile(model.input_emb, inputs=(sample,), verbose=False)
    flops_input_emb = flops_input_emb * num_steps
    
    with torch.no_grad():
        decoder_out = x_after_mha  # 
    flops_ln_f, _ = profile(model.ln_f, inputs=(decoder_out,), verbose=False)
    flops_ln_f = flops_ln_f * num_steps
    
    with torch.no_grad():
        ln_out = model.ln_f(decoder_out)
    flops_head, _ = profile(model.head, inputs=(ln_out,), verbose=False)
    flops_head = flops_head * num_steps
    
    total_flops = flops_input_emb + flops_sa_total + flops_mha_total + flops_ff_total + flops_ln_f + flops_head
    
    breakdown = {
        'input_emb': flops_input_emb,
        'sa_blocks': flops_sa_total,
        'mha_blocks': flops_mha_total,
        'ff_blocks': flops_ff_total,
        'ln_f': flops_ln_f,
        'head': flops_head
    }
    
    logger.info(f"\nPruned Decoder: {total_flops/1e9:.4f} GFLOPs")
    
    # pruning ratio
    base_decoder_blocks = (flops_sa + flops_mha + flops_ff) * num_steps * n_layers
    pruned_decoder_blocks = flops_sa_total + flops_mha_total + flops_ff_total
    pruning_ratio = 1 - (pruned_decoder_blocks / base_decoder_blocks) if base_decoder_blocks > 0 else 0
    logger.info(f"Decoder blocks pruning ratio: {pruning_ratio*100:.2f}%")
    
    return total_flops, breakdown


def count_pruner_flops(
    pruner: nn.Module,
    encoder_buffer_batch: torch.Tensor,
    block_keys: list,
    num_steps: int = 100,
    batch_size: int = 1,
    device: torch.device = torch.device('cpu')
) -> float:
    """PrunerFLOPs"""
    logger.info(f"\n{'='*60}")
    logger.info(f"PrunerFLOPs")
    logger.info(f"{'='*60}")
    
    total_flops = 0.0
    memory_batch = encoder_buffer_batch.view(batch_size * num_steps, -1, encoder_buffer_batch.shape[-1])
    
    # 1. memory_proj (Linear)
    logger.info(f"\n1. memory_proj: {batch_size * num_steps}")
    single_memory = memory_batch[0:1]
    flops_memory_proj_single, _ = profile(pruner.memory_proj, inputs=(single_memory,), verbose=False)
    flops_memory_proj = flops_memory_proj_single * (batch_size * num_steps)
    logger.info(f"   FLOPs: {flops_memory_proj/1e9:.4f} GFLOPs")
    total_flops += flops_memory_proj
    
    # 2. block_emb (EmbeddingFLOPs0)
    logger.info(f"\n2. block_emb: 1")
    block_ids = torch.tensor([pruner.block_to_id[k] for k in block_keys], dtype=torch.long, device=device)
    flops_block_emb, _ = profile(pruner.block_emb, inputs=(block_ids,), verbose=False)
    logger.info(f"   FLOPs: {flops_block_emb/1e9:.4f} GFLOPs (Embedding)")
    total_flops += flops_block_emb
    
    # 3. transformer_decoder (TransformerDecoder)
    logger.info(f"\n3. transformer_decoder: 1 ({batch_size * num_steps})")
    with torch.no_grad():
        memory_proj_batch = pruner.memory_proj(memory_batch)
        block_emb = pruner.block_emb(block_ids)
        tgt_batch = block_emb.unsqueeze(0).expand(batch_size * num_steps, -1, -1)
    
    # TransformerDecoder
    if isinstance(pruner.transformer_decoder, nn.TransformerDecoder):
        # TransformerDecoderFLOPs
        seq_len_tgt = tgt_batch.shape[1]  # target
        seq_len_memory = memory_proj_batch.shape[1]  # memory
        d_model = tgt_batch.shape[2]
        
        first_layer = pruner.transformer_decoder.layers[0]
        n_head = first_layer.self_attn.num_heads
        dim_ff = first_layer.linear1.out_features
        n_decoder_layers = len(pruner.transformer_decoder.layers)
        
        logger.info(f"   Decoder: TransformerDecoder")
        logger.info(f"   Config: tgt_len={seq_len_tgt}, memory_len={seq_len_memory}, d_model={d_model}, n_head={n_head}, dim_ff={dim_ff}, n_layers={n_decoder_layers}")
        
        # FLOPs
        flops_decoder_layer_single = manual_transformer_decoder_layer_flops(seq_len_tgt, seq_len_memory, d_model, n_head, dim_ff)
        flops_transformer_decoder = flops_decoder_layer_single * n_decoder_layers * (batch_size * num_steps)
        
        logger.info(f"   decoder FLOPs: {flops_decoder_layer_single/1e6:.2f} MFLOPs")
        logger.info(f"   decoder FLOPs: {flops_decoder_layer_single * n_decoder_layers/1e6:.2f} MFLOPs")
    else:
        # TransformerDecoderthop
        flops_transformer_decoder, _ = profile(pruner.transformer_decoder, inputs=(tgt_batch, memory_proj_batch), verbose=False)
    
    logger.info(f"   transformer_decoderFLOPs: {flops_transformer_decoder/1e9:.4f} GFLOPs")
    total_flops += flops_transformer_decoder
    
    # 4. head (Linear)
    logger.info(f"\n4. head: 1")
    with torch.no_grad():
        decoder_output_batch = pruner.transformer_decoder(tgt_batch, memory_proj_batch)
    flops_head, _ = profile(pruner.head, inputs=(decoder_output_batch,), verbose=False)
    logger.info(f"   FLOPs: {flops_head/1e9:.4f} GFLOPs")
    total_flops += flops_head
    
    logger.info(f"\nPruner: {total_flops/1e9:.4f} GFLOPs")
    return total_flops


def get_block_keys_from_model(model: nn.Module, cfg: Any) -> list:
    """decoder block"""
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            n_layers = len(model.decoder.layers)
            layer_names = [f"decoder.layers.{i}" for i in range(n_layers)]
        else:
            layer_names = [f"decoder.layers.{i}" for i in range(8)]
    
    block_keys = enumerate_decoder_block_keys(layer_names)
    return block_keys


@click.command()
@click.option('-c', '--checkpoint', required=True, help='Policy checkpoint path')
@click.option('-p', '--pruner_path', default=None, help='Pruner ()')
@click.option('-t', '--task_name', required=True, help='Task name')
@click.option('-d', '--device', default='cuda:0', help='Device')
@click.option('-b', '--batch_size', default=1, type=int, help='Batch')
@click.option('-n', '--num_steps', default=100, type=int, help='Diffusion')
def main(checkpoint, pruner_path, task_name, device, batch_size, num_steps):
    """Diffusion PolicyPrunerFLOPs"""
    torch_device = torch.device(device)
    
    # 1.
    logger.info(f"checkpoint: {checkpoint}")
    workspace, cfg = load_workspace(checkpoint, output_dir='tmp_output')
    base_policy = workspace.model
    if getattr(cfg, 'training', None) is not None and getattr(cfg.training, 'use_ema', False):
        ema_model = getattr(workspace, 'ema_model', None)
        if ema_model is not None:
            base_policy = ema_model

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    
    base_policy = base_policy.to(torch_device)
    base_policy.eval()
    model = base_policy.model
    
    # 2.
    logger.info(f" (batch_size={batch_size})")
    obs_dict = construct_obs_dict(cfg, task_name, torch_device, batch_size=batch_size)
    nobs = base_policy.normalizer.normalize(obs_dict) if hasattr(base_policy, 'normalizer') else obs_dict
    
    # cond
    if model.obs_as_cond and hasattr(model, 'cond_obs_emb'):
        from diffusion_policy.common.pytorch_util import dict_apply
        this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = base_policy.obs_encoder(this_nobs) if hasattr(base_policy, 'obs_encoder') else None
        if nobs_features is not None:
            cond = nobs_features.reshape(batch_size, n_obs_steps, -1)
        else:
            cond = obs_dict['obs'][:, :n_obs_steps]
    else:
        cond = obs_dict['obs'][:, :n_obs_steps]
    
    # sample
    if hasattr(cfg, 'shape_meta') and 'action' in cfg.shape_meta:
        action_dim = cfg.shape_meta.action.shape[0]
    elif hasattr(cfg.task, 'action_dim'):
        action_dim = cfg.task.action_dim
    elif hasattr(base_policy, 'action_dim'):
        action_dim = base_policy.action_dim
    else:
        action_dim = 7
    
    horizon = model.horizon if hasattr(model, 'horizon') else 16
    if hasattr(base_policy, 'pred_action_steps_only') and base_policy.pred_action_steps_only:
        sample_shape = (batch_size, n_action_steps, action_dim)
    else:
        sample_shape = (batch_size, horizon, action_dim)
    sample = torch.randn(sample_shape, device=torch_device)
    
    logger.info(f"\n: cond={cond.shape}, sample={sample.shape}")
    
    # 3. Encoder FLOPs
    with torch.no_grad():
        encoder_flops = count_encoder_flops(model, cond, num_steps, torch_device)
    
    # 4. memorydecoder
    with torch.no_grad():
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=torch_device)
        time_emb = model.time_emb(timesteps).unsqueeze(1)
        if model.obs_as_cond and hasattr(model, 'cond_obs_emb'):
            cond_obs_emb = model.cond_obs_emb(cond)
            cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
        else:
            cond_embeddings = time_emb
        tc = cond_embeddings.shape[1]
        position_embeddings = model.cond_pos_emb[:, :tc, :]
        encoder_input = model.drop(cond_embeddings + position_embeddings)
        memory = model.encoder(encoder_input)
    
    # 5. Base Decoder FLOPs
    n_layers = len(model.decoder.layers)
    with torch.no_grad():
        decoder_base_flops, decoder_base_breakdown = count_decoder_base_flops(model, sample, memory, num_steps, n_layers)
    
    # 6. PrunerPruned Decoder FLOPs
    pruner_flops = 0.0
    decoder_pruned_flops = 0.0
    decoder_pruned_breakdown = {}
    
    if pruner_path is not None and os.path.exists(pruner_path):
        logger.info(f"\nPruner: {pruner_path}")
        pruner = load_pruner_model(pruner_path, cfg, torch_device, policy=base_policy, 
                                   reuse_block=False, tgt_sa=None, rollout_cache=True)
        pruner.eval()
        
        block_keys = get_block_keys_from_model(model, cfg)
        n_layers = len(model.decoder.layers)
        
        # encoder_buffer_batch
        with torch.no_grad():
            timesteps_batch = torch.arange(num_steps-1, -1, -1, dtype=torch.long, device=torch_device)
            time_emb_batch = model.time_emb(timesteps_batch)
            time_emb_batch = time_emb_batch.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(2)
            
            if model.obs_as_cond and hasattr(model, 'cond_obs_emb'):
                cond_obs_emb = model.cond_obs_emb(cond)
                cond_obs_emb_batch = cond_obs_emb.unsqueeze(1).expand(-1, num_steps, -1, -1)
                cond_embeddings_batch = torch.cat([time_emb_batch, cond_obs_emb_batch], dim=2)
            else:
                cond_embeddings_batch = time_emb_batch
            
            bs, ns, seq_len, hidden_dim = cond_embeddings_batch.shape
            cond_embeddings_flat = cond_embeddings_batch.view(bs * ns, seq_len, hidden_dim)
            tc = cond_embeddings_flat.shape[1]
            position_embeddings = model.cond_pos_emb[:, :tc, :]
            x_batch = model.drop(cond_embeddings_flat + position_embeddings)
            memory_batch = model.encoder(x_batch)
            encoder_buffer_batch = memory_batch.view(batch_size, num_steps, -1, memory_batch.shape[-1])
        
        # Pruner FLOPs
        with torch.no_grad():
            pruner_flops = count_pruner_flops(pruner, encoder_buffer_batch, block_keys, num_steps, batch_size, torch_device)
        
        # pruner
        logger.info(f"\nprunergate...")
        with torch.no_grad():
            memory_batch_flat = encoder_buffer_batch.view(batch_size * num_steps, -1, encoder_buffer_batch.shape[-1])
            memory_proj_batch = pruner.memory_proj(memory_batch_flat)
            block_ids = torch.tensor([pruner.block_to_id[k] for k in block_keys], dtype=torch.long, device=torch_device)
            block_emb = pruner.block_emb(block_ids)
            tgt_batch = block_emb.unsqueeze(0).expand(batch_size * num_steps, -1, -1)
            decoder_output_batch = pruner.transformer_decoder(tgt_batch, memory_proj_batch)
            logits_batch = pruner.head(decoder_output_batch)
            logits_batch = logits_batch.view(batch_size, num_steps, -1, 4)
            logits_batch = torch.flip(logits_batch, dims=[1])
            
            # apply_scheduler_single
            strategy_dict = apply_scheduler_single(logits=logits_batch, num_steps=num_steps, batch_idx=0)
        
        # Pruned Decoder FLOPs
        with torch.no_grad():
            decoder_pruned_flops, decoder_pruned_breakdown = count_decoder_pruned_flops(
                model, sample, memory, strategy_dict, num_steps, n_layers
            )
    
    # 7.
    logger.info(f"\n\n{'='*60}")
    logger.info(f"{'='*60}")
    logger.info(f"FLOPs")
    logger.info(f"{'='*60}")
    
    # Base Policy
    logger.info(f"\n[Base Policy]")
    base_total = encoder_flops + decoder_base_flops
    logger.info(f"  1. Encoder:  {encoder_flops/1e9:.4f} GFLOPs ({encoder_flops/base_total*100:.2f}%)")
    logger.info(f"  2. Decoder:  {decoder_base_flops/1e9:.4f} GFLOPs ({decoder_base_flops/base_total*100:.2f}%)")
    logger.info(f"     - SA blocks:  {decoder_base_breakdown['sa_blocks']/1e9:.4f} GFLOPs")
    logger.info(f"     - MHA blocks: {decoder_base_breakdown['mha_blocks']/1e9:.4f} GFLOPs")
    logger.info(f"     - FF blocks:  {decoder_base_breakdown['ff_blocks']/1e9:.4f} GFLOPs")
    logger.info(f"     - Other:      {(decoder_base_breakdown['input_emb']+decoder_base_breakdown['ln_f']+decoder_base_breakdown['head'])/1e9:.4f} GFLOPs")
    logger.info(f"  : {base_total/1e9:.4f} GFLOPs")
    
    # Pruned Policy
    if pruner_path is not None and os.path.exists(pruner_path):
        logger.info(f"\n[Pruned Policy]")
        pruned_total = encoder_flops + decoder_pruned_flops + pruner_flops
        logger.info(f"  1. Encoder:         {encoder_flops/1e9:.4f} GFLOPs ({encoder_flops/pruned_total*100:.2f}%)")
        logger.info(f"  2. Decoder (pruned):{decoder_pruned_flops/1e9:.4f} GFLOPs ({decoder_pruned_flops/pruned_total*100:.2f}%)")
        logger.info(f"     - SA blocks:  {decoder_pruned_breakdown['sa_blocks']/1e9:.4f} GFLOPs")
        logger.info(f"     - MHA blocks: {decoder_pruned_breakdown['mha_blocks']/1e9:.4f} GFLOPs")
        logger.info(f"     - FF blocks:  {decoder_pruned_breakdown['ff_blocks']/1e9:.4f} GFLOPs")
        logger.info(f"     - Other:      {(decoder_pruned_breakdown['input_emb']+decoder_pruned_breakdown['ln_f']+decoder_pruned_breakdown['head'])/1e9:.4f} GFLOPs")
        logger.info(f"  3. Pruner:          {pruner_flops/1e9:.4f} GFLOPs ({pruner_flops/pruned_total*100:.2f}%)")
        logger.info(f"  : {pruned_total/1e9:.4f} GFLOPs")
        
        logger.info(f"\n[Speedup Summary]")
        speedup = base_total / pruned_total if pruned_total > 0 else 0
        flops_reduction = (base_total - pruned_total) / base_total * 100 if base_total > 0 else 0
        decoder_speedup = decoder_base_flops / decoder_pruned_flops if decoder_pruned_flops > 0 else 0
        logger.info(f"  FLOPs: {flops_reduction:.2f}%")
        logger.info(f"  : {speedup:.2f}x")
        logger.info(f"  Decoder: {decoder_speedup:.2f}x")
    
    logger.info(f"\n{'='*60}")
    
    import json
    results = {
        'config': {
            'checkpoint': checkpoint,
            'pruner_path': pruner_path,
            'task_name': task_name,
            'device': str(torch_device),
            'batch_size': batch_size,
            'num_steps': num_steps,
        },
        'base_policy': {
            'encoder_gflops': float(encoder_flops/1e9),
            'decoder_gflops': float(decoder_base_flops/1e9),
            'decoder_breakdown': {k: float(v/1e9) for k, v in decoder_base_breakdown.items()},
            'total_gflops': float(base_total/1e9),
            'encoder_percentage': float(encoder_flops/base_total*100),
            'decoder_percentage': float(decoder_base_flops/base_total*100),
        }
    }
    
    if pruner_path is not None and os.path.exists(pruner_path):
        results['pruned_policy'] = {
            'encoder_gflops': float(encoder_flops/1e9),
            'decoder_pruned_gflops': float(decoder_pruned_flops/1e9),
            'pruner_gflops': float(pruner_flops/1e9),
            'total_gflops': float(pruned_total/1e9),
            'encoder_percentage': float(encoder_flops/pruned_total*100),
            'decoder_percentage': float(decoder_pruned_flops/pruned_total*100),
            'pruner_percentage': float(pruner_flops/pruned_total*100),
            'decoder_breakdown': {k: float(v/1e9) for k, v in decoder_pruned_breakdown.items()},
        }
        results['speedup'] = {
            'flops_reduction_percentage': float(flops_reduction),
            'overall_theoretical_speedup': float(speedup),
            'decoder_speedup': float(decoder_speedup),
        }
    
    output_file = 'module_flops_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n: {output_file}")


if __name__ == '__main__':
    main()
