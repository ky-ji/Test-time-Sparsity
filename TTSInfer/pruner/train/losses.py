from __future__ import annotations

from typing import Dict, List, Optional, Union, cast

import torch
import torch.nn.functional as F


GatesDict = Dict[int, Dict[str, torch.Tensor]]


def compute_consistency_loss(orig_action: torch.Tensor, pruned_action: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    if loss_type == "mse":
        # print("mseLC")
        return F.mse_loss(pruned_action, orig_action)
    if loss_type == "l1":
        return F.l1_loss(pruned_action, orig_action)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pruned_action, orig_action)
    #  KL
    raise ValueError(f"Unsupported loss_type: {loss_type}")



def compute_sparse_loss(
    global_lc: bool,
    hard_gates: torch.Tensor, 
    target_pruning_ratio: float,
    alpha_s: float = 1.0,
    loss_type: str = 'mse',
) -> torch.Tensor:
    """
    sample|reuse_probs.mean() - target_pruning_ratio|batch
    Binary gate (N=2) Ternary gate (N=3)
    
    Args:
        hard_gates: [batch, T, B, N] 
            - Binary gate (N=2): hard_gates[:,:,:,0]Represents reuse probability
            - Ternary gate (N=3): hard_gates[:,:,:,0]compute[:,:,:,1][:,:,:,2]reuse
        target_pruning_ratio: reuse
        alpha_s: 
        loss_type:  ('mse'  'l1')
        
    Returns:
        
    """

    if hard_gates.shape[3] == 2:
        reuse_probs = hard_gates[:, :, :, 1]
    elif hard_gates.shape[3] == 4:
        reuse_probs = hard_gates[:, :, :, 1] + hard_gates[:, :, :, 2] + hard_gates[:, :, :, 3]
    else:
        reuse_probs = hard_gates[:, :, :, 1] + hard_gates[:, :, :, 2]

    if global_lc:
        # sample |reuse_probs.mean() - target_pruning_ratio|
        batch_size = reuse_probs.shape[0]
        sample_losses = []
        
        for i in range(batch_size):
            # isampleblock
            sample_reuse_mean = reuse_probs[i].mean()  # TB
            if loss_type == 'l1':
                sample_loss = torch.abs(sample_reuse_mean - target_pruning_ratio)
            else:
                # print("mseLS")
                sample_loss = torch.square(sample_reuse_mean - target_pruning_ratio)
            sample_losses.append(sample_loss)
        
        # batch
        sparse_loss = alpha_s * torch.stack(sample_losses).mean()
    
    else:
        #  block  target
        #  block  (batch, T)  target  block
        # reuse_block_mean: [B]
        reuse_block_mean = reuse_probs.mean(dim=(0, 1))
        block_losses = torch.abs(reuse_block_mean - target_pruning_ratio)

        sparse_loss = alpha_s * block_losses.mean()
    
    return sparse_loss



def compose_total_loss(
    global_lc: bool,
    consistency_type: str,
    sparse_type: str,
    orig_action: torch.Tensor,
    pruned_action: torch.Tensor,
    gates: Union[GatesDict, torch.Tensor],
    lamb_sparse: float = 0.0,
    alpha_s: float = 1.0,
    target_pruning_ratio: float = 0.0,
):
    Lc = compute_consistency_loss(orig_action, pruned_action, loss_type=consistency_type)
    Ls = compute_sparse_loss(global_lc,gates, target_pruning_ratio, alpha_s, loss_type=sparse_type)
    total_loss = Lc + lamb_sparse * Ls
    return total_loss, Lc, Ls


