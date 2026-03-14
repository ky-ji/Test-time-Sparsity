"""
Gate Scheduler - 

 pruner  logits 
"""
import torch
from typing import Dict, Tuple


def apply_scheduler_single(
    logits: torch.Tensor,
    num_steps: int,
    batch_idx: int = 0,
) -> Dict[int, int]:
    """
     batch 
    
    Args:
        logits:  [B, num_steps, num_blocks, num_strategies]
        num_steps: 
        batch_idx: batch 
        
    Returns:
         {block_idx: strategy}
    """
    #  argmax
    hard_decisions = logits[batch_idx].argmax(dim=-1)  # [num_steps, num_blocks]

    #  TTSInfer
    # - diffusion  step 0  compute cache rollout cache
    # -  step0  rollout/3cache/24cache
    if hard_decisions.shape[0] > 0:
        hard_decisions[0, :] = 0
    
    strategy_dict = {}
    num_blocks = hard_decisions.shape[1]
    
    for step in range(num_steps):
        for block in range(num_blocks):
            global_block_idx = step * num_blocks + block
            strategy_dict[global_block_idx] = int(hard_decisions[step, block].item())
    
    return strategy_dict


def apply_scheduler_batch(
    logits: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    
    
    Args:
        logits:  [B, num_steps, num_blocks, num_strategies]
        num_steps: 
        
    Returns:
        (soft_gate, hard_gate)
        - soft_gate: softmax  [B, num_steps, num_blocks, num_strategies]
        - hard_gate: one-hot  [B, num_steps, num_blocks, num_strategies]
    """
    soft_gate = torch.softmax(logits, dim=-1)
    
    # argmax -> one-hot
    hard_indices = logits.argmax(dim=-1, keepdim=True)
    hard_gate = torch.zeros_like(logits).scatter_(-1, hard_indices, 1.0)
    
    return soft_gate, hard_gate


def compute_cache_ratio(strategy_dict: Dict[int, int]) -> Dict[str, float]:
    """
    
    
    Args:
        strategy_dict: 
        
    Returns:
        
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for strategy in strategy_dict.values():
        counts[strategy] = counts.get(strategy, 0) + 1
    
    total = sum(counts.values())
    if total == 0:
        return {"compute": 0, "3cache": 0, "24cache": 0, "rollout": 0}
    
    return {
        "compute": counts[0] / total,
        "3cache": counts[1] / total,
        "24cache": counts[2] / total,
        "rollout": counts[3] / total,
    }
