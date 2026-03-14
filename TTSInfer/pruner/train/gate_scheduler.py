from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import time

GatesDict = Dict[int, Dict[str, torch.Tensor]]


def _extract_probs(logits: GatesDict) -> GatesDict:
    probs: GatesDict = {}
    for t, blk in logits.items():
        probs[t] = {}
        for k, lg in blk.items():
            probs[t][k] = F.softmax(lg, dim=-1)  
    return probs


def STE(probs_dict: GatesDict) -> GatesDict:
    """
     step×block  Straight-Through Estimator
    / {t: {block_key: Tensor([p_reuse, p_compute])}}
    """
    hard: GatesDict = {}
    for t, blk in probs_dict.items():
        hard[t] = {}
        for b, p in blk.items():
            # p: Tensor shape [2]
            index = p.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(p, memory_format=torch.legacy_contiguous_format).scatter_(dim=-1, index=index, value=1.0)
            # straight-through
            ret = y_hard - p.detach() + p
            hard[t][b] = ret
    return hard


def STE_tensor(probs: torch.Tensor) -> torch.Tensor:
    """
     [batch, T, B, N]  Straight-Through Estimator
    Binary gate (N=2) Ternary gate (N=3)
    
    Args:
        probs: [batch, T, B, N] N23
        
    Returns:
        hard_gates: [batch, T, B, N] 
    """
    #  [batch, T, B, 1]
    index = probs.max(dim=-1, keepdim=True)[1]
    
    # one-hot [batch, T, B, N]
    y_hard = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format)
    y_hard.scatter_(dim=-1, index=index, value=1.0)
    
    # Straight-Through Estimator:
    hard_gates = y_hard - probs.detach() + probs
    
    return hard_gates


def apply_scheduler(
    logits: GatesDict,
    num_steps: int,
) -> Tuple[GatesDict, GatesDict]:
    """
    STE
    (0)(num_steps-1)compute [0, 1]
    """
    probs = _extract_probs(logits)     # logits --> probs
    
    # step
    force_compute = torch.tensor([0.0, 1.0], dtype=probs[0][list(probs[0].keys())[0]].dtype, 
                                device=probs[0][list(probs[0].keys())[0]].device)

            # for block_key in probs[step]:
            #     probs[step][block_key] = force_compute.clone()

    gate_hard = STE(probs)
    return probs, gate_hard

def apply_scheduler_single(
    logits: torch.Tensor,
    num_steps: int,
    batch_idx: int = 0
) -> Dict[int, int]:
    """
     [batch, T, B, N] logits
     (N=2)  (N=3) gate
    (0)compute
    
    Ternary gate
    - 0: compute ()
    - 1: reuse_3cache (3cache)
    - 2: reuse_24cache (24cache)
    
    Args:
        logits: [batch, T, B, N] logitsN23
        num_steps: 
        batch_idx: batchbatch
        
    Returns:
        strategy_dict: {flat_idx: strategy} 
            - flat_idx: step * num_blocks + block_idx
            - strategy: 0=compute, 1=reuse_3cache, 2=reuse_24cache
    """
    with torch.no_grad():
        # batchlogits [T, B, N]
        logits_tb = logits[batch_idx]
        T, B, N = logits_tb.shape
        
        #  [T, B]
        strategy_matrix = logits_tb.argmax(dim=-1)  # [T, B]
        
        # Force compute on first step (initialize cache)
        strategy_matrix[0, :] = 0  # blockcompute

        
        # : {flat_idx: strategy}
        # v2CPU/numpyGPU->CPU
        strategy_matrix_flat = strategy_matrix.flatten().cpu().numpy()  # [T*B] numpy
        strategy_dict = {i: int(strategy_matrix_flat[i]) for i in range(T * B)}

        return strategy_dict


def apply_scheduler_batch(
    logits: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    softhard gates
     (N=2)  (N=3) gate

    Ternary gate
    - [..., 0]: compute ()
    - [..., 1]: reuse_3cache (3cache)
    - [..., 2]: reuse_24cache (24cache)

    Args:
        logits: [batch, T, B, N] logitsN23
        num_steps: 

    Returns:
        probs_scheduled: [batch, T, B, N] soft gate
        hard_gates: [batch, T, B, N] hard gate (one-hot with STE)
    """
    # Convert to probabilities [batch, T, B, N]
    probs = F.softmax(logits, dim=-1)
    N = probs.shape[-1]  # gate23

    if N ==3 : 
        force_compute = torch.tensor([1.0, 0.0, 0.0], dtype=probs.dtype, device=probs.device)

    # 3cache & 24cache & rollout cache
    elif N == 4:
        force_compute = torch.tensor([1.0, 0.0, 0.0,0.0], dtype=probs.dtype, device=probs.device)

    # Clone probs to avoid in-place modification
    probs_scheduled = probs.clone()

    # Force compute on first step (initialize cache)
    probs_scheduled[:, 0, :, :] = force_compute  # All 24 blocks compute on first step

    # Apply STE to obtain hard gates
    hard_gates = STE_tensor(probs_scheduled)

    return probs_scheduled, hard_gates

def apply_scheduler_batch_ablation(
    logits: torch.Tensor,
    num_steps: int,
    cache_type: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert to probabilities [batch, T, B, N]
    probs = F.softmax(logits, dim=-1)
    N = probs.shape[-1]  # gate23

    if cache_type == "3cache" or cache_type == "24cache":
        force_compute = torch.tensor([1.0, 0.0], dtype=probs.dtype, device=probs.device)

        # Clone probs to avoid in-place modification
        probs_scheduled = probs.clone()

        # Force compute on first step (initialize cache)
        probs_scheduled[:, 0, :, :] = force_compute  # All 24 blocks compute on first step

    else:
        probs_scheduled = probs
        
    # Apply STE to obtain hard gates
    hard_gates = STE_tensor(probs_scheduled)

    return probs_scheduled, hard_gates

# Compute real-time pruning ratio = Σ(flops_ratio * gk) / total_blocks ()
def calculate_pruning_ratio_tensor(
    hard_gate: torch.Tensor, 
    block_names: List[str], 
    flops_ratios: Dict[str, float] = None, 
    num_steps: Optional[int] = None
) -> float:
    """
    Compute real-time pruning ratio, Supports tensor-format hard_gate [batch, T, B, N]
     (N=2)  (N=3) gate
    
    Binary gate: [reuse, compute]
    Ternary gate: [compute, reuse_3cache, reuse_24cache]
    
    pruning_ratiocompute (cache)
    
    Args:
        hard_gate: [batch, T, B, N] N=23
        block_names: block name list, corresponding to B dimension
        flops_ratios: {'sa': 0.27, 'ca': 0.18, 'ffn': 0.55} etc. FLOP ratios
        num_steps: Total steps (optional)
    
    Returns:
        pruning_ratio: Pruning ratio (proportion of cache reuse)
    """
    batch_size, T, B, N = hard_gate.shape
    
    if N == 2:
        # Binary gate: hard_gate[:,:,:,0]Represents reuse probability
        reuse_probs = hard_gate[:, :, :, 0].mean()  # Average over all dimensions
        pruning_ratio = float(reuse_probs.item())
    elif N == 3:
        # Ternary gate: hard_gate[:,:,:,0]Represents compute probability
        # pruning_ratio = 1 - compute_ratio = reuse_3cache_ratio + reuse_24cache_ratio
        compute_probs = hard_gate[:, :, :, 0].mean()  # Proportion of full compute
        pruning_ratio = 1.0 - float(compute_probs.item())
    else:
        raise ValueError(f"Unsupported gate dimension: {N}")
    
    return pruning_ratio


def calculate_gate_statistics_stage2(
    hard_gate: torch.Tensor
) -> Dict[str, float]:
    """
    Compute detailed gate statistics (all steps)
    
    Args:
        hard_gate: [batch, T, B, N] 
    
    Returns:
        
        - Binary gate: {'p_reuse': float, 'p_compute': float, 'pruning_ratio': float}
        - Ternary gate: {'p_compute': float, 'p1_reuse_3cache': float, 'p2_reuse_24cache': float, 
                     'p_reuse_total': float, 'pruning_ratio': float}
    """
    
    stats = {}

    # Ternary gate
    p1_reuse_3cache = hard_gate[:, :, :, 1].mean().item()
    p2_reuse_24cache = hard_gate[:, :, :, 2].mean().item()
    p3_reuse_rlcache = hard_gate[:, :, :, 3].mean().item()
    p_reuse_total = p1_reuse_3cache + p2_reuse_24cache + p3_reuse_rlcache
    
    stats = {
        'p1_reuse_3cache': p1_reuse_3cache,
        'p2_reuse_24cache': p2_reuse_24cache,
        'p3_reuse_rlcache': p3_reuse_rlcache,
        'pruning_ratio': p_reuse_total
    }

    return stats


def calculate_pruning_ratio(
    gates: Union[GatesDict, torch.Tensor], 
    flops_ratios: Dict[str, float] = None, 
    num_steps: Optional[int] = None,
    block_names: Optional[List[str]] = None
) -> float:
    """
    pruning ratio
    block1pruning_ratio = reuse gate
    
    Args:
        gates: GatesDict
        flops_ratios: FLOP1
        num_steps: Total steps (optional)
        block_names: blockName list
    
    Returns:
        pruning_ratio: reuse gate
    """
    if isinstance(gates, torch.Tensor):
        return calculate_pruning_ratio_tensor(gates, block_names or [], flops_ratios, num_steps)
    
    steps = sorted(gates.keys())
    if not steps:
        return 0.0
    
    total_reuse_prob = 0.0
    total_blocks = 0
    
    for step in steps:
        step_gates = gates[step]
        for block_key, gate_probs in step_gates.items():
            # gate_probs[0]  reuse
            if isinstance(gate_probs, torch.Tensor) and len(gate_probs) >= 2:
                reuse_prob = float(gate_probs[0].item())
                total_reuse_prob += reuse_prob
                total_blocks += 1
    
    if total_blocks == 0:
        return 0.0
    
    pruning_ratio = total_reuse_prob / total_blocks
    return pruning_ratio

