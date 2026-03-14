"""Trajectory-based pruner training for simulation."""

#!/usr/bin/env python
from __future__ import annotations

import sys
import os
# Add TTSInfer parent dir to path for diffusion_policy import
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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from TTSInfer.acceleration.rollout.pruner_warpper_train import CachePrunerWrapper
from TTSInfer.pruner.train.gate_scheduler import apply_scheduler, calculate_pruning_ratio
from TTSInfer.pruner.train.losses import compose_total_loss
from TTSInfer.pruner.train.train_utils import save_pruner_ckpt, save_json, set_seed, visualize_hard_gates, visualize_hard_gates_trajectory, create_gate_animation, NormalizerManager
from TTSInfer.pruner.train.logger import Logger
from TTSInfer.pruner import utils
from TTSInfer.pruner.train.gate_scheduler import calculate_gate_statistics_trajectory
from TTSInfer.pruner.trajectory.trajectory_dataset import TrajectoryDataset
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
    if args.train_version is not None:
        config['basic']['train_version'] = args.train_version
    if args.datatype is not None:
        config['basic']['datatype'] = args.datatype
    
    # DDIM
    if hasattr(args, 'ddim') and args.ddim is not None:
        config['basic']['ddim'] = args.ddim
    
    # Checkpoint path
    if args.checkpoint is not None:
        config['checkpoint']['path'] = args.checkpoint
    
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.target_prune_ratio is not None:
        config['pruning']['target_prune_ratio'] = args.target_prune_ratio
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    
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
    args.ddim = config['basic'].get('ddim', None)  # DDIMNonescheduler
    args.datatype = config['basic'].get('datatype', None)
    args.checkpoint = utils.get_task_ckpt(config['basic']['task_name'],config['basic']['train_version'])
    
    args.epochs = config['training']['epochs']
    args.lr = config['training']['lr']
    args.warmup_steps = config['training']['warmup_steps']
    args.min_lr = config['training']['min_lr']
    args.patience = config['training']['patience']
    args.early_stop = config['training']['early_stop']

    
    trajectory_training = config['trajectory_training']
    args.num_batch_train_tra = trajectory_training.get('num_batch_train_tra', 3)
    args.num_batch_val_tra = trajectory_training.get('num_batch_val_tra', 2)
    args.train_tra_batch_size = trajectory_training.get('train_tra_batch_size', 32)
    args.val_tra_batch_size = trajectory_training.get('val_tra_batch_size', 8)
    args.epochs_trajectory = trajectory_training.get('epochs', 10)
    args.lr_trajectory = trajectory_training.get('lr', 1.0e-06)
    args.min_lr_trajectory = trajectory_training.get('lr_min', 1.0e-07)
    
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
    
    return args


def setup_ddim_scheduler(policy, ddim_steps: int):
    """
    policyschedulerDDIMsetup
    
    Args:
        policy: 
        ddim_steps: DDIM
    """
    original_scheduler = policy.noise_scheduler
    
    print(f"scheduler: {type(original_scheduler).__name__}")
    print(f": {original_scheduler.config.num_train_timesteps}")
    print(f": {policy.num_inference_steps}")
    
    # DDIM schedulerConfig
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=original_scheduler.config.num_train_timesteps,
        beta_start=original_scheduler.config.beta_start,
        beta_end=original_scheduler.config.beta_end,
        beta_schedule=original_scheduler.config.beta_schedule,
        clip_sample=getattr(original_scheduler.config, 'clip_sample', True),
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type=original_scheduler.config.prediction_type
    )
    
    # scheduler
    policy.noise_scheduler = ddim_scheduler
    policy.num_inference_steps = ddim_steps
    
    print(f"✓ DDIM schedulersetup: {ddim_steps}")


def initialize_new_pruner(cfg, policy, args, device):
    """
    prunerstage1
    
    Args:
        cfg: Config object
        policy: 
        args: 
        device: Device
        
    Returns:
        pruner: pruner
    """
    print("prunerstage1...")
    
    # Get model parametersDDIMDDIM
    if args.ddim is not None:
        num_steps = args.ddim
    else:
        num_steps = getattr(cfg, 'num_inference_steps', 100)
    
    # layer_names
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]
    
    block_keys = enumerate_decoder_block_keys(layer_names)
    
    # obs_dim DP  cond_obs_emb
    obs_dim = policy.model.cond_obs_emb.out_features if hasattr(policy.model, 'cond_obs_emb') else 512
    
    # pruner4rollout cache
    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_keys,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        dim_feedforward=args.dim_feedforward,
        block_encoder_type=args.block_encoder_type,
        obs_dim=obs_dim,
        head_4=True
    ).to(device)

    return pruner


def train_with_config(task_name: str, device: str, config_path: str, output_dir: str = None, train_version: int = 0, use_direct_output_dir: bool = False) -> bool:
    """
    Config
    
    Args:
        task_name: Task name
        device: Device ( 'cuda:0')
        config_path: Config path
        output_dir: 
        train_version: checkpoint
        use_direct_output_dir: output_dir
    
    Returns:
        bool: 
    """
    # Config
    if not os.path.exists(config_path):
        print(f": Config: {config_path}")
        return False
    
    print(f"Config: {config_path}")

    # Config
    config = load_config(config_path)
    
    # Config
    config['basic']['task_name'] = task_name
    if output_dir:
        config['basic']['output_dir'] = output_dir
    
    # Configargs
    args = create_args_from_config(config)
    
    # checkpointtrain
    args.checkpoint = get_task_ckpt_with_train_version(task_name, train_version)
    
    return _run_training(args, device, use_direct_output_dir, config_dict=config)


def _run_training(args, device_str: str, use_direct_output_dir: bool = False, config_dict: Dict[str, Any] = None) -> bool:
    """
    
    Args:
        args: 
        device_str: Device
        use_direct_output_dir: output_dir
        config_dict: Config
    """

    # Config
    print(f": {args.task_name},  : {args.target_prune_ratio}")
    
    set_seed(args.seed)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # The public simulation pipeline always trains from scratch on trajectory data.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output_path = os.path.join(args.output_dir, 'pruner_ckpt', timestamp, str(args.train_version), args.task_name)
    os.makedirs(train_output_path, exist_ok=True)
    
    # Config
    if config_dict is not None:
        config_save_path = os.path.join(train_output_path, 'training_config.yaml')
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ Config: {config_save_path}")
    
    # load policy/workspace
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(cfg)
    workspace._output_dir = args.output_dir
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model.to(device)
    policy.eval()
    
    # DDIMschedulersetup
    if args.ddim is not None:
        print(f"\n{'='*60}")
        print(f"DDIM scheduler: {args.ddim}")
        print(f"{'='*60}")
        setup_ddim_scheduler(policy, args.ddim)
        num_steps = args.ddim
    else:
        num_steps = getattr(cfg, 'num_inference_steps', 100)
        print(f"\nscheduler: {num_steps}")

    # ==================== Load Dataset and DataLoader ====================
    num_train_episodes = args.num_batch_train_tra * args.train_tra_batch_size
    num_val_episodes = args.num_batch_val_tra * args.val_tra_batch_size
    
    # trainvalepisodes
    trajectory_data_dir = f"pruner_tra_data_{args.datatype}/trajectories/{args.task_name}"
    
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
    print(": Prunerrollout cache")
    pruner = initialize_new_pruner(cfg, policy, args, device)

     # Create optimizer
    optimizer_config = getattr(args, 'optimizer_config', {})
    optim = torch.optim.AdamW(
        pruner.parameters(),
        lr=args.lr_trajectory,
        betas=optimizer_config.get('betas', [0.9, 0.999]),
        weight_decay=1e-4,
    )

    # Create learning rate scheduler: warmup + linear annealing
    def warmup_linear_scheduler(epoch):
        if epoch < args.warmup_steps:
            # Warmup phase: Linear warmup to initial learning rate
            warmup_factor = args.min_lr_trajectory/args.lr_trajectory + float(epoch) / float(max(1, args.warmup_steps))
            return warmup_factor
        else:
            # Linear decay phase: Linear decay from initial LR to minimum LR
            remaining_epochs = args.epochs - args.warmup_steps
            if remaining_epochs <= 0:
                return 1.0  # If no decay phase, keep original LR
            
            decay_progress = float(epoch - args.warmup_steps) / float(remaining_epochs)
            decay_progress = min(decay_progress, 1.0)  # 1.0
            
            # 1.0min_lr_ratio
            min_lr_ratio = args.min_lr_trajectory / args.lr_trajectory 
            lr_factor = 1.0 - decay_progress * (1.0 - min_lr_ratio)
            return max(min_lr_ratio, lr_factor)
    
    scheduler = LambdaLR(optim, lr_lambda=warmup_linear_scheduler)

    logger = Logger(args)

    # Wrap forward/prediction logic with cache acceleration
    CachePrunerWrapper.apply(policy, pruner=pruner, if_rollout_cache=True)

    # build pruner on top of policy/wrapper
    num_steps = getattr(workspace, 'num_inference_steps', getattr(policy, 'num_inference_steps', 100))
    block_keys = policy._cache_block_keys
    
    
    # ==================== Training Loop ====================
    
    for epoch in range(args.epochs_trajectory):   
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
            visualization_path = os.path.join(train_output_path, f'valid_epoch_{epoch}_gate_visualization_frame10.png')
            visualize_hard_gates_trajectory(
                gate_tensor=val_metrics['gate_hard_frame10'],
                block_names=block_keys,
                output_path=visualization_path,
                title=f"Trajectory Training Hard Gates Frame10 ({args.task_name}, Epoch: {epoch}, Target Ratio: {args.target_prune_ratio:.2f})"
            )
        
        # hard_gate30frame
        if val_metrics.get('gate_hard_frame30') is not None:
            visualization_path = os.path.join(train_output_path, f'valid_epoch_{epoch}_gate_visualization_frame30.png')
            visualize_hard_gates_trajectory(
                gate_tensor=val_metrics['gate_hard_frame30'],
                block_names=block_keys,
                output_path=visualization_path,
                title=f"Trajectory Training Hard Gates Frame30 ({args.task_name}, Epoch: {epoch}, Target Ratio: {args.target_prune_ratio:.2f})"
            )

        if logger.early_stop(val_metrics, train_output_path, epoch, args, pruner):
            break
    
    final_save_path = os.path.join(train_output_path, 'final_pruner.pth')
    torch.save({
        'model_state_dict': pruner.state_dict(),
        'structure': args.structure,
        'hidden_dim': args.hidden_dim,
        'attn_heads': args.attn_heads,
        'dim_feedforward': args.dim_feedforward,
        'block_encoder_type': args.block_encoder_type,
        'epoch': args.epochs_trajectory,
    }, final_save_path)
    print(f"\n✓ : {final_save_path}")
    
    return True


def train_epoch_trajectory( epoch, pruner, policy, train_dataset, 
                         episode_batch_size, num_steps, block_keys, args, optimizer, scheduler, logger):

    pruner.train()
    policy.eval()
    
    train_losses = []
    train_consistency_losses = []
    train_sparse_losses = []
    train_pruning_ratios = []
    
    device = next(pruner.parameters()).device

    # episode batchesshuffle
    for batch_frames in train_dataset.get_episode_batch_iterator(episode_batch_size=episode_batch_size, shuffle=True):
        
        # episode batchrollout cache
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
            
            # Calculate pruning ratio
            gate_stats = calculate_gate_statistics_trajectory(gate_hard)
            current_pruning_ratio = gate_stats['pruning_ratio']
            
            if frame_idx == 0:
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



def validate_epoch_trajectory( pruner, policy, val_dataset, 
                             episode_batch_size, num_steps, block_keys, args):

    pruner.eval()
    policy.eval()

    policy._cache["training"] = False
    
    valid_losses = []
    valid_consistency_losses = []
    valid_sparse_losses = []
    valid_pruning_ratios = []
    last_gate_stats = None  
    
    device = next(pruner.parameters()).device
    
    # gate_hardbatchframe10frame30
    vis_gate_hard_frame10 = None
    vis_gate_hard_frame30 = None
    
    with torch.no_grad():
        batch_count = 0
        
        # episode batchesshuffle
        for batch_frames in val_dataset.get_episode_batch_iterator(episode_batch_size=episode_batch_size, shuffle=False):
            # episode batch
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
                
                # Calculate pruning ratio
                gate_stats = calculate_gate_statistics_trajectory(gate_hard)
                current_pruning_ratio = gate_stats['pruning_ratio']
                
                # framefirst
                if frame_idx == 0:
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
                last_gate_stats = gate_stats
            
    
    avg_metrics = {
        'valid_loss': np.mean(valid_losses),
        'valid_consistency_loss': np.mean(valid_consistency_losses),
        'valid_sparse_loss': np.mean(valid_sparse_losses),
        'valid_pruning_ratio': np.mean(valid_pruning_ratios),
        'gate_stats': last_gate_stats,
        'gate_hard_frame10': vis_gate_hard_frame10,
        'gate_hard_frame30': vis_gate_hard_frame30,
    }
    
    print(f" - "
          f"Loss: {avg_metrics['valid_loss']:.4f}, "
          f"Prune Ratio: {avg_metrics['valid_pruning_ratio']:.2%}")
    
    return avg_metrics


def move_batch_to_device(obs_batch, device):
    """Move observation batch to specified device"""
    if isinstance(obs_batch, dict):
        device_obs = {}
        for key, value in obs_batch.items():
            if isinstance(value, torch.Tensor):
                device_obs[key] = value.to(device)
            else:
                device_obs[key] = value
        return device_obs
    elif isinstance(obs_batch, torch.Tensor):
        return obs_batch.to(device)
    else:
        return obs_batch



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='Device (Override config file)')
    parser.add_argument('--task_name', type=str, help='Task name (Override config file)')
    parser.add_argument('--seed', type=int, help='Random seed (Override config file)')
    parser.add_argument('--output_dir', type=str, help=' (Override config file)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (overrides config)')
    parser.add_argument('--lr', type=float, help=' (Override config file)')
    parser.add_argument('--target_prune_ratio', type=float, help=' (Override config file)')
    parser.add_argument('--hidden_dim', type=int, help=' (Override config file)')
    parser.add_argument('--config', type=str, default=None, help='Path to the training config file')
    parser.add_argument('--use_direct_output_dir', action='store_true', help='output_dir (Override config file)')
    parser.add_argument('--train_version', type=int,default=0, help=' (Override config file)')
    parser.add_argument('--ddim', type=int, default=None, help='DDIM scheduler--ddim 4040DDIM')
    parser.add_argument('--datatype', type=str, default=None, help=' (Override config file)')
    
    cmd_args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = cmd_args.config or os.path.join(project_root, 'pruner_config', 'training_config.yaml')
    
    # Config
    if not os.path.exists(config_path):
        print(f": Config: {config_path}")
        sys.exit(1)
    
    print(f"Config: {config_path}")

    # Config
    config = load_config(config_path)

    # Config
    config = override_config_with_args(config, cmd_args)
    
    # Configargs
    args = create_args_from_config(config)
    
    # Config
    success = _run_training(args, cmd_args.device, cmd_args.use_direct_output_dir, config_dict=config)
    
    if success:
        print("Done!")
    else:
        print("Done!")
        sys.exit(1)


if __name__ == '__main__':
    main()
