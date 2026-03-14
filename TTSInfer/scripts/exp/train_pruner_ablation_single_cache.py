"""
cache
--cache_typecache3cache24cacherollout_cache

:
python TTSInfer/scripts/train_pruner_ablation_single_cache.py \
--config_id 2stage1 \
--task_name tool_hang_ph \
--device cuda:4 \
--output_dir exp_ablation \
--pruner_epoch 18 \
--cache_type rollout_cache

cache_type: 3cache, 24cache, rollout_cache
"""

#!/usr/bin/env python
from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time
from typing import Dict, Any
import random
import copy
import yaml

import dill
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from TTSInfer.acceleration.ablation.pruner_warpper_train_ablation import CachePrunerWrapper
from TTSInfer.pruner.train.gate_scheduler import apply_scheduler, calculate_pruning_ratio
from TTSInfer.pruner.train.losses import compose_total_loss
from TTSInfer.pruner.train.train_utils import save_pruner_ckpt, save_json, set_seed, visualize_hard_gates, visualize_hard_gates_stage2, create_gate_animation, NormalizerManager
from TTSInfer.pruner.train.logger import Logger
from TTSInfer.pruner import utils
from TTSInfer.pruner.train.gate_scheduler import calculate_gate_statistics_stage2
from TTSInfer.pruner.stage2.trajectory_dataset import TrajectoryDataset
from TTSInfer.pruner.train.transformer_pruner import TransformerPruner
from TTSInfer.pruner.train.train_utils import enumerate_decoder_block_keys
from datetime import datetime, timezone
from TTSInfer.pruner.utils import get_task_ckpt_with_train_version


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def override_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config file parameters with command line arguments"""
    # Base config override
    if args.device is not None:
        config['basic']['device'] = args.device
    if args.task_name is not None:
        config['basic']['task_name'] = args.task_name
    if args.seed is not None:
        config['basic']['seed'] = args.seed
    if args.output_dir is not None:
        config['basic']['output_dir'] = args.output_dir
    
    # Checkpoint path
    if args.checkpoint is not None:
        config['checkpoint']['path'] = args.checkpoint
    
    if args.pruner_epoch is not None:
        config['basic']['pruner_epoch'] = args.pruner_epoch
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.target_prune_ratio is not None:
        config['pruning']['target_prune_ratio'] = args.target_prune_ratio
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    
    # cache_type
    if hasattr(args, 'cache_type') and args.cache_type is not None:
        config['ablation'] = config.get('ablation', {})
        config['ablation']['cache_type'] = args.cache_type
    
    return config

def create_args_from_config(config: Dict[str, Any]) -> argparse.Namespace:
    """Configargs"""
    args = argparse.Namespace()
    
    # Config
    args.use_swanlab = config['basic']['use_swanlab']
    args.use_wandb = config['basic']['use_wandb']
    args.use_tensorboard = config['basic']['use_tensorboard']
    args.seed = config['basic']['seed']
    args.output_dir = config['basic']['output_dir']
    args.task_name = config['basic']['task_name']
    args.train_version = config['basic']['train_version']
    args.pruner_epoch = config['basic']['pruner_epoch']
    
    args.checkpoint = utils.get_task_ckpt(config['basic']['task_name'],config['basic']['train_version'])
    
    args.epochs = config['training']['epochs']
    args.lr = config['training']['lr']
    args.warmup_steps = config['training']['warmup_steps']
    args.min_lr = config['training']['min_lr']
    args.patience = config['training']['patience']
    args.early_stop = config['training']['early_stop']
    
    args.num_batch_train_tra = config['stage2'].get('num_batch_train_tra', 3)
    args.num_batch_val_tra = config['stage2'].get('num_batch_val_tra', 2)
    args.train_tra_batch_size = config['stage2'].get('train_tra_batch_size', 32)
    args.val_tra_batch_size = config['stage2'].get('val_tra_batch_size', 8)
    args.epochs_stage2 = config['stage2'].get('epochs_tra', config['stage2'].get('epochs', 10))
    args.lr_stage2 = config['stage2'].get('lr', 1.0e-06)
    args.min_lr_stage2 = config['stage2'].get('lr_min', 1.0e-07)
    
    args.optimizer_config = config['training']['optimizer']
    args.weight_decay_config = args.optimizer_config['weight_decay']
    
    args.structure = config['model']['structure']
    args.hidden_dim = config['model']['hidden_dim']
    args.block_encoder_type = config['model']['block_encoder_type']
    args.attn_heads = config['model']['attn_heads']
    args.dim_feedforward = config['model']['dim_feedforward']
    
    args.target_prune_ratio = config['pruning']['target_prune_ratio']
    
    args.global_lc = config['loss']['global_lc']
    args.consistency = config['loss']['consistency']
    args.sparse = config['loss']['sparse']
    args.use_target_action = config['loss']['use_target_action']
    args.lamb_sparse = config['loss']['lamb_sparse']

    args.save_gate_animation = config['visual']['save_gate_animation']
    
    args.cache_type = config.get('ablation', {}).get('cache_type', '3cache')
    
    return args


def initialize_pruner_single_cache(cfg, policy, args, device):
    """
    cachepruner2computereuse
    
    Args:
        cfg: Config object
        policy: 
        args: 
        device: Device
        
    Returns:
        pruner: pruner
    """
    print(f"cache pruner (cache_type={args.cache_type})...")
    
    # Get model parameters
    num_steps = getattr(cfg, 'num_inference_steps', 100)
    
    # layer_names
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]
    
    block_keys = enumerate_decoder_block_keys(layer_names)
    
    #  obs_dimGet from DP model
    obs_dim = policy.model.cond_obs_emb.out_features if hasattr(policy.model, 'cond_obs_emb') else 512
    
    # pruner2computereuse
    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_keys,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        dim_feedforward=args.dim_feedforward,
        block_encoder_type=args.block_encoder_type,
        obs_dim=obs_dim,
        head_4=False  # cache2
    ).to(device)
    
    return pruner


def _run_training(args, device_str: str, config_dict: Dict[str, Any] = None) -> bool:
    """
    
    Args:
        args: 
        device_str: Device
        config_dict: Config
    """

    # Config
    print(f": {args.task_name}, Cache: {args.cache_type}, : {args.target_prune_ratio}")
    
    set_seed(args.seed)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # cache_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        'ablation_results',
        'single_cache',
        args.cache_type,
        timestamp,
        str(args.train_version),
        args.task_name
    )
    os.makedirs(output_path, exist_ok=True)
    
    # Config
    if config_dict is not None:
        config_save_path = os.path.join(output_path, 'training_config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ Config: {config_save_path}")
    
    # load policy/workspace
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    num_steps = getattr(cfg, 'num_inference_steps', 100)
    workspace = cls(cfg)
    workspace._output_dir = args.output_dir
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model.to(device)
    policy.eval()

    # ==================== Load Dataset and DataLoader ====================
    num_train_episodes = args.num_batch_train_tra * args.train_tra_batch_size
    num_val_episodes = args.num_batch_val_tra * args.val_tra_batch_size
    
    # trainvalepisodes
    trajectory_data_dir = f"pruner_tra_data_max/trajectories/{args.task_name}"
    
    # episode
    train_episode_indices = [i for i in range(num_train_episodes)]
    val_episode_indices = [i for i in range(num_train_episodes, num_train_episodes + num_val_episodes)]
    
    train_dataset = TrajectoryDataset(
        data_dir=trajectory_data_dir,
        device='cpu',  # GPU
        episode_indices=train_episode_indices  # episode
    )
    
    val_dataset = TrajectoryDataset(
        data_dir=trajectory_data_dir,
        device='cpu',
        episode_indices=val_episode_indices  # episode
    )
    
    episode_batch_size = args.train_tra_batch_size  
    val_episode_batch_size = args.val_tra_batch_size

    # ==================== Initialize Pruner ====================
    print(f"cachePruner (cache_type={args.cache_type})")
    pruner = initialize_pruner_single_cache(cfg, policy, args, device)

    # Create optimizer
    optimizer_config = getattr(args, 'optimizer_config', {})
    optim = torch.optim.AdamW(
        pruner.parameters(),
        lr=args.lr_stage2,
        betas=optimizer_config.get('betas', [0.9, 0.999]),
        weight_decay=1e-4,
    )

    # Create learning rate scheduler: warmup + linear annealing
    def warmup_linear_scheduler(epoch):
        if epoch < args.warmup_steps:
            # Warmup phase: Linear warmup to initial learning rate
            warmup_factor = args.min_lr_stage2/args.lr_stage2 + float(epoch) / float(max(1, args.warmup_steps))
            return warmup_factor
        else:
            # Linear decay phase: Linear decay from initial LR to minimum LR
            remaining_epochs = args.epochs - args.warmup_steps
            if remaining_epochs <= 0:
                return 1.0  # If no decay phase, keep original LR
            
            decay_progress = float(epoch - args.warmup_steps) / float(remaining_epochs)
            decay_progress = min(decay_progress, 1.0)  # 1.0
            
            # 1.0min_lr_ratio
            min_lr_ratio = args.min_lr_stage2 / args.lr_stage2 
            lr_factor = 1.0 - decay_progress * (1.0 - min_lr_ratio)
            return max(min_lr_ratio, lr_factor)
    
    scheduler = LambdaLR(optim, lr_lambda=warmup_linear_scheduler)

    logger = Logger(args)

    # Wrap forward/prediction logic with cache accelerationcache_type
    if args.cache_type == 'rollout_cache':
        CachePrunerWrapper.apply(policy, pruner=pruner, cache_type=args.cache_type, if_rollout=True)
    else:
        CachePrunerWrapper.apply(policy, pruner=pruner, cache_type=args.cache_type, if_rollout=False)

    # build pruner on top of policy/wrapper
    num_steps = getattr(workspace, 'num_inference_steps', getattr(policy, 'num_inference_steps', 100))
    block_keys = policy._cache_block_keys
    
    
    # ==================== Training Loop ====================
    
    for epoch in range(args.epochs_stage2):   
        # Training phase
        train_metrics = train_epoch_trajectory(
            epoch=epoch,
            pruner=pruner,
            policy=policy,
            train_dataset=train_dataset,
            episode_batch_size=episode_batch_size,
            num_steps=num_steps,
            block_keys=block_keys,
            args=args,
            optimizer=optim,
            scheduler=scheduler,
            logger=logger
        )

        scheduler.step()

        # Validation phase
        val_metrics = validate_epoch_trajectory(
            pruner=pruner,
            policy=policy,
            val_dataset=val_dataset,
            episode_batch_size=val_episode_batch_size,
            num_steps=num_steps,
            block_keys=block_keys,
            args=args,
        )
    
        logger.log_valid(epoch, val_metrics)
        
        # hard_gate10frame
        if val_metrics.get('gate_hard_frame10') is not None:
            visualization_path = os.path.join(output_path, f'valid_epoch_{epoch}_gate_visualization_frame10.png')
            # gate_hard_frame10 [steps, num_blocks, 2]
            visualize_hard_gates(
                gates=val_metrics['gate_hard_frame10'],
                output_path=visualization_path,
                title=f"Single Cache ({args.cache_type}) Frame10 ({args.task_name}, Epoch: {epoch})"
            )
        
        # hard_gate30frame
        if val_metrics.get('gate_hard_frame30') is not None:
            visualization_path = os.path.join(output_path, f'valid_epoch_{epoch}_gate_visualization_frame30.png')
            # gate_hard_frame30 [steps, num_blocks, 2]
            visualize_hard_gates(
                gates=val_metrics['gate_hard_frame30'],
                output_path=visualization_path,
                title=f"Single Cache ({args.cache_type}) Frame30 ({args.task_name}, Epoch: {epoch})"
            )

        if logger.early_stop(val_metrics, output_path, epoch, args, pruner):
            break
    
    final_save_path = os.path.join(output_path, 'final_pruner.pth')
    torch.save({
        'model_state_dict': pruner.state_dict(),
        'structure': args.structure,
        'hidden_dim': args.hidden_dim,
        'attn_heads': args.attn_heads,
        'dim_feedforward': args.dim_feedforward,
        'block_encoder_type': args.block_encoder_type,
        'cache_type': args.cache_type,  # cache
        'epoch': args.epochs_stage2,
    }, final_save_path)
    print(f"\n✓ : {final_save_path}")
    
    return True


def train_epoch_trajectory(epoch, pruner, policy, train_dataset, 
                          episode_batch_size, num_steps, block_keys, args, optimizer, scheduler, logger):

    pruner.train()
    policy.eval()
    
    train_losses = []
    train_consistency_losses = []
    train_sparse_losses = []
    train_pruning_ratios = []
    
    device = next(pruner.parameters()).device

    # episode batches
    for batch_frames in train_dataset.get_episode_batch_iterator(episode_batch_size=episode_batch_size):
        
        # episode batch
        if args.cache_type == 'rollout_cache':
            policy._cache['is_first_predict_action_in_chunk'] = True
        
        # batchframes
        for frame_idx, batched_frame in enumerate(batch_frames):
            optimizer.zero_grad()

            obs_batch = {}
            for key, value in batched_frame['obs'].items():
                if isinstance(value, torch.Tensor):
                    obs_batch[key] = value.to(device)
                else:
                    obs_batch[key] = value
            
            orig_action = batched_frame['action'].to(device)

            policy._cache["training"] = True

            pruned_action_batch = policy.predict_action(obs_batch)['action_pred']
               
            gate_hard = policy._cache['gate']
            
            # Calculate pruning ratio (cache2)
            current_pruning_ratio = (gate_hard[:, :, :, 1].sum() / gate_hard.numel() * 2).item()
            
            if frame_idx == 0 and args.cache_type == 'rollout_cache':
                policy._cache['is_first_predict_action_in_chunk'] = False
            
            # Compute loss
            loss, Lc, Ls = compose_total_loss(
                global_lc=args.global_lc,
                consistency_type=args.consistency,
                sparse_type=args.sparse,
                orig_action=orig_action,
                pruned_action=pruned_action_batch,
                gates=gate_hard,
                lamb_sparse=args.lamb_sparse,
                target_pruning_ratio=args.target_prune_ratio,
            )
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Record metrics
            train_losses.append(loss.item())
            train_consistency_losses.append(Lc.item())
            train_sparse_losses.append(Ls.item())
            train_pruning_ratios.append(current_pruning_ratio)

            logger.update_global_step()

            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
            
            gate_stats = {'pruning_ratio': current_pruning_ratio}
            logger.log_train(
                epoch=epoch,
                current_loss=loss,
                Lc=Lc,
                Ls=Ls,
                current_pruning_ratio=current_pruning_ratio,
                learning_rate=current_lr,
                gate_stats=gate_stats
            )
        
    
    avg_metrics = {
        'train_loss': np.mean(train_losses),
        'train_consistency_loss': np.mean(train_consistency_losses),
        'train_sparse_loss': np.mean(train_sparse_losses),
        'train_pruning_ratio': np.mean(train_pruning_ratios),
    }
    
    print(f"\n[Epoch {epoch}]  - "
          f"Loss: {avg_metrics['train_loss']:.4f}, "
          f"Prune Ratio: {avg_metrics['train_pruning_ratio']:.2%}")
    
    return avg_metrics



def validate_epoch_trajectory(pruner, policy, val_dataset, 
                             episode_batch_size, num_steps, block_keys, args):

    pruner.eval()
    policy.eval()

    policy._cache["training"] = False
    
    valid_losses = []
    valid_consistency_losses = []
    valid_sparse_losses = []
    valid_pruning_ratios = []
    
    device = next(pruner.parameters()).device
    
    # gate_hardbatchframe10frame30
    vis_gate_hard_frame10 = None
    vis_gate_hard_frame30 = None
    
    with torch.no_grad():
        batch_count = 0
        
        # episode batches
        for batch_frames in val_dataset.get_episode_batch_iterator(episode_batch_size=episode_batch_size):
            # episode batch
            if args.cache_type == 'rollout_cache':
                policy._cache['is_first_predict_action_in_chunk'] = True
            
            # batchframes
            for frame_idx, batched_frame in enumerate(batch_frames):
                # device
                obs_batch = {}
                for key, value in batched_frame['obs'].items():
                    if isinstance(value, torch.Tensor):
                        obs_batch[key] = value.to(device)
                    else:
                        obs_batch[key] = value
                
                orig_action = batched_frame['action'].to(device)

                policy._cache["training"] = False

                pruned_action_batch = policy.predict_action(obs_batch)['action_pred']

                gate_hard = policy._cache['gate']
                
                # batchframe10frame30gate
                if batch_count == 0 and frame_idx == 10:
                    vis_gate_hard_frame10 = gate_hard[0].detach().cpu()
                if batch_count == 0 and frame_idx == 30:
                    vis_gate_hard_frame30 = gate_hard[0].detach().cpu()
                
                # Calculate pruning ratio (cache2)
                current_pruning_ratio = (gate_hard[:, :, :, 1].sum() / gate_hard.numel() * 2).item()
                
                # framefirst
                if frame_idx == 0 and args.cache_type == 'rollout_cache':
                    policy._cache['is_first_predict_action_in_chunk'] = False
                
                # Compute loss
                loss, Lc, Ls = compose_total_loss(
                    global_lc=args.global_lc,
                    consistency_type=args.consistency,
                    sparse_type=args.sparse,
                    orig_action=orig_action,
                    pruned_action=pruned_action_batch,
                    gates=gate_hard,
                    lamb_sparse=args.lamb_sparse,
                    target_pruning_ratio=args.target_prune_ratio,
                )
                
                # Record
                valid_losses.append(loss.item())
                valid_consistency_losses.append(Lc.item() if Lc is not None else 0.0)
                valid_sparse_losses.append(Ls.item() if Ls is not None else 0.0)
                valid_pruning_ratios.append(current_pruning_ratio)
            
            batch_count += 1
    
    avg_metrics = {
        'valid_loss': np.mean(valid_losses),
        'valid_consistency_loss': np.mean(valid_consistency_losses),
        'valid_sparse_loss': np.mean(valid_sparse_losses),
        'valid_pruning_ratio': np.mean(valid_pruning_ratios),
        'gate_stats': {'pruning_ratio': np.mean(valid_pruning_ratios)},
        'gate_hard_frame10': vis_gate_hard_frame10,
        'gate_hard_frame30': vis_gate_hard_frame30,
    }
    
    print(f" - "
          f"Loss: {avg_metrics['valid_loss']:.4f}, "
          f"Prune Ratio: {avg_metrics['valid_pruning_ratio']:.2%}")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_id', type=str, required=True, help='ConfigID')
    parser.add_argument('--cache_type', type=str, required=True, 
                       choices=['3cache', '24cache', 'rollout_cache'],
                       help='Cache')
    
    parser.add_argument('--device', type=str, help='Device (Override config file)')
    parser.add_argument('--task_name', type=str, help='Task name (Override config file)')
    parser.add_argument('--seed', type=int, help='Random seed (Override config file)')
    parser.add_argument('--output_dir', type=str, help=' (Override config file)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (overrides config)')
    parser.add_argument('--pruner_epoch', type=int, help=' (Override config file)')
    parser.add_argument('--lr', type=float, help=' (Override config file)')
    parser.add_argument('--target_prune_ratio', type=float, help=' (Override config file)')
    parser.add_argument('--hidden_dim', type=int, help=' (Override config file)')
    
    cmd_args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'pruner_config', f'training_config_{cmd_args.config_id}.yaml')
    
    # Config
    if not os.path.exists(config_path):
        print(f": Config: {config_path}")
        sys.exit(1)
    
    print(f"Config: {config_path}")
    print(f"Cache: {cmd_args.cache_type}")

    # Config
    config = load_config(config_path)

    # Config
    config = override_config_with_args(config, cmd_args)
    
    # Configargs
    args = create_args_from_config(config)
    
    # Config
    success = _run_training(args, cmd_args.device, config_dict=config)
    
    if success:
        print("Done!")
    else:
        print("Done!")
        sys.exit(1)


if __name__ == '__main__':
    main()

