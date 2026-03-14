"""
TTS Pruner Training Script - Episode-based Training with Rollout Cache
TTSInfer
1. prunerwrappergate
2. TrajectoryDatasetepisode-based
3. rollout cacheframe
"""
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent                # realworld-TTS/
project_root = repo_root.parent               # Test-time-Sparsity/ (where diffusion_policy submodule lives)
ttsinfer_root = project_root / "TTSInfer"
accel_root = repo_root / "acceleration"

for p in (current_dir, repo_root, project_root, ttsinfer_root, accel_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import yaml
import torch
import dill
import hydra
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from tqdm import tqdm
import wandb
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

#  TTSInfer  pruner
from pruner.train.transformer_pruner import TransformerPruner  # noqa: E402
from pruner.train.gate_scheduler import calculate_gate_statistics_stage2  # noqa: E402
from pruner.train.losses import compose_total_loss  # noqa: E402
from pruner.train.train_utils import set_seed, save_pruner_ckpt  # noqa: E402
from pruner.stage2.trajectory_dataset import TrajectoryDataset  # noqa: E402

#  diffusion_policy

# wrapperTTSInfer
from cache_pruner_wrapper_train import CachePrunerWrapper  # noqa: E402


def normalize_obs(obs_dict):
    """Normalize observations: convert uint8 images to float [0, 1]"""
    normalized = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert uint8 images to float and normalize to [0, 1]
            if value.dtype == torch.uint8:
                normalized[key] = value.float() / 255.0
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def color_augment(obs_dict, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    
    
    Args:
        obs_dict: 
        brightness:  [1-brightness, 1+brightness]
        contrast:  [1-contrast, 1+contrast]
        saturation:  [1-saturation, 1+saturation]
        hue:  [-hue, hue]
    
    Returns:
        
    """
    import random
    
    augmented = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 3:
            # 33
            # : [B, T, C, H, W]  [B, C, H, W]  [C, H, W]
            shape = value.shape
            
            #  RGB 3
            is_image = False
            if len(shape) >= 3:
                # Found
                if shape[-3] == 3:  # [..., C, H, W] 
                    is_image = True
                elif shape[-1] == 3:  # [..., H, W, C] 
                    is_image = True
            
            if is_image and shape[-3] == 3:  #  [..., C, H, W] 
                original_shape = value.shape
                
                #  [N, C, H, W]
                if len(shape) == 5:  # [B, T, C, H, W]
                    B, T, C, H, W = shape
                    value_flat = value.reshape(B * T, C, H, W)
                elif len(shape) == 4:  # [B, C, H, W]
                    value_flat = value
                else:  # [C, H, W]
                    value_flat = value.unsqueeze(0)
                
                if brightness > 0:
                    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
                    value_flat = value_flat * brightness_factor
                
                if contrast > 0:
                    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
                    mean = value_flat.mean(dim=[-3, -2, -1], keepdim=True)
                    value_flat = (value_flat - mean) * contrast_factor + mean
                
                # RGB ->
                if saturation > 0:
                    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
                    gray = 0.299 * value_flat[:, 0:1] + 0.587 * value_flat[:, 1:2] + 0.114 * value_flat[:, 2:3]
                    gray = gray.expand_as(value_flat)
                    value_flat = value_flat * saturation_factor + gray * (1 - saturation_factor)
                
                # RGB
                if hue > 0:
                    hue_shift = random.uniform(-hue, hue)
                    # RGB
                    if abs(hue_shift) > 0.01:
                        r, g, b = value_flat[:, 0:1], value_flat[:, 1:2], value_flat[:, 2:3]
                        cos_h = np.cos(hue_shift * np.pi)
                        sin_h = np.sin(hue_shift * np.pi)
                        new_r = r * cos_h + g * sin_h * 0.5 - b * sin_h * 0.5
                        new_g = g * cos_h + b * sin_h * 0.5 - r * sin_h * 0.5
                        new_b = b * cos_h + r * sin_h * 0.5 - g * sin_h * 0.5
                        value_flat = torch.cat([new_r, new_g, new_b], dim=1)
                
                #  [0, 1]
                value_flat = torch.clamp(value_flat, 0.0, 1.0)
                
                if len(original_shape) == 5:
                    value = value_flat.reshape(original_shape)
                elif len(original_shape) == 4:
                    value = value_flat
                else:
                    value = value_flat.squeeze(0)
                
                augmented[key] = value
            else:
                augmented[key] = value
        else:
            augmented[key] = value
    
    return augmented


def train_epoch_trajectory(
    epoch, pruner, policy, train_dataset,
    episode_batch_size, num_steps, block_keys, args, optimizer, scheduler, device,
    is_distributed=False, rank=0, world_size=1,
    grad_accum_steps=4, use_amp=True, scaler=None,
    trajectory_sample_ratio=1.0
):
    """
    Episode-based - TTSInferwrapperbatchpruner

    TTSInfer/scripts/train_eval/train_pruner_2stage.py
    1. episode batches
    2. episode batchis_first_predict_action_in_chunk=True
    3. batchframes
    4. framesetupis_first_predict_action_in_chunk=False
    5. Wrapperpredict_actionbatchprunergate
    6. Rollout cacheepisode batchframe
    
    
    -  grad_accum_steps  optimizer.step()
    -  AMP 
    -  loss gate
    -  epoch  trajectory_sample_ratio 
    """
    pruner.train()
    policy.eval()

    train_loss_sum = 0.0
    train_consistency_sum = 0.0
    train_sparse_sum = 0.0
    train_prune_sum = 0.0
    train_count = 0
    accumulated_steps = 0  # 

    # batch
    import math
    import random
    num_episodes = len(train_dataset)
    total_batches_full = math.ceil(num_episodes / episode_batch_size)
    
    # epoch
    # batch
    if trajectory_sample_ratio < 1.0:
        num_batches_to_process = max(1, int(total_batches_full * trajectory_sample_ratio))
    else:
        num_batches_to_process = total_batches_full
    
    # episode batchesshuffle
    batch_iterator = train_dataset.get_episode_batch_iterator(
        episode_batch_size=episode_batch_size, 
        shuffle=True
    )
    
    pbar_batch = tqdm(
        batch_iterator,
        total=num_batches_to_process,
        desc=f"Epoch {epoch} [Batches {num_batches_to_process}/{total_batches_full}]",
        leave=False,
        disable=(rank != 0)
    )
    
    batches_processed = 0

    for batch_frames in pbar_batch:

        # episode batchrollout cache
        policy._cache['is_first_predict_action_in_chunk'] = True

        # batchframes
        pbar_frame = tqdm(
            enumerate(batch_frames),
            total=len(batch_frames),
            desc="  Frames",
            leave=False,
            ncols=100,
            disable=(rank != 0)
        )

        for frame_idx, batched_frame in pbar_frame:
            obs_batch = {}
            for key, value in batched_frame['obs'].items():
                if isinstance(value, torch.Tensor):
                    obs_batch[key] = value.to(device)
                else:
                    obs_batch[key] = value

            obs_batch = normalize_obs(obs_batch)
            
            obs_batch = color_augment(obs_batch, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

            orig_action = batched_frame['action'].to(device)

            # setuptraining
            policy._cache["training"] = True

            # ✅ TTSInferwrapperbatchprunergate
            #  AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                pruned_action_batch = policy.predict_action(obs_batch)['action_pred']
                   
                gate_hard = policy._cache['gate']
                
                gate_stats = calculate_gate_statistics_stage2(gate_hard)
                current_pruning_ratio = gate_stats['pruning_ratio']
                
                # framefirst
                if frame_idx == 0:
                    policy._cache['is_first_predict_action_in_chunk'] = False
                    #  loss  gate pruner
                    train_prune_sum += current_pruning_ratio
                    train_count += 1
                    continue
                
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
                loss = loss / grad_accum_steps  # 
            
            # Backward pass scaler  AMP
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_steps += 1
            
            #  grad_accum_steps
            if accumulated_steps % grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            # loss
            train_loss_sum += loss.item() * grad_accum_steps
            train_consistency_sum += Lc.item()
            train_sparse_sum += Ls.item()
            train_prune_sum += current_pruning_ratio
            train_count += 1
            
            # frame
            pbar_frame.set_postfix({
                'loss': f'{loss.item() * grad_accum_steps:.4f}',
                'prune': f'{current_pruning_ratio:.2%}'
            })
        
        # batch
        if train_count > 0:
            pbar_batch.set_postfix({
                'avg_loss': f'{train_loss_sum / train_count:.4f}',
                'avg_prune': f'{train_prune_sum / train_count:.2%}'
            })
        
        #  batch
        batches_processed += 1
        if batches_processed >= num_batches_to_process:
            break
    
    if accumulated_steps % grad_accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if train_count == 0:
        avg_metrics = {
            'train_loss': 0.0,
            'train_consistency_loss': 0.0,
            'train_sparse_loss': 0.0,
            'train_pruning_ratio': 0.0,
        }
    else:
        avg_metrics = {
            'train_loss': train_loss_sum / train_count,
            'train_consistency_loss': train_consistency_sum / train_count,
            'train_sparse_loss': train_sparse_sum / train_count,
            'train_pruning_ratio': train_prune_sum / train_count,
        }

    if is_distributed:
        avg_metrics = _all_reduce_metrics(avg_metrics, device)

    return avg_metrics


def validate_epoch_trajectory(
    pruner, policy, val_dataset,
    episode_batch_size, num_steps, block_keys, args, device,
    is_distributed=False, rank=0, world_size=1
):
    """Episode-based - TTSInferwrapperbatchpruner"""
    if val_dataset is None:
        return {
            'valid_loss': 0.0,
            'valid_consistency_loss': 0.0,
            'valid_sparse_loss': 0.0,
            'valid_pruning_ratio': 0.0,
            'gate_stats': None,
        }
    pruner.eval()
    policy.eval()

    policy._cache["training"] = False

    valid_loss_sum = 0.0
    valid_consistency_sum = 0.0
    valid_sparse_sum = 0.0
    valid_prune_sum = 0.0
    valid_count = 0
    last_gate_stats = None

    import math
    num_val_episodes = len(val_dataset)
    total_batches = math.ceil(num_val_episodes / episode_batch_size)

    with torch.no_grad():
        # episode batchesshuffle
        batch_iterator = val_dataset.get_episode_batch_iterator(episode_batch_size=episode_batch_size, shuffle=False)
        pbar_val = tqdm(batch_iterator, total=total_batches, desc="Validation", leave=False, disable=(rank != 0))

        for batch_frames in pbar_val:
            # episode batch
            policy._cache['is_first_predict_action_in_chunk'] = True

            # batchframes
            for frame_idx, batched_frame in tqdm(
                enumerate(batch_frames),
                total=len(batch_frames),
                desc="  Val Frames",
                leave=False,
                ncols=100,
                disable=(rank != 0)
            ):
                # device
                obs_batch = {}
                for key, value in batched_frame['obs'].items():
                    if isinstance(value, torch.Tensor):
                        obs_batch[key] = value.to(device)
                    else:
                        obs_batch[key] = value

                obs_batch = normalize_obs(obs_batch)

                orig_action = batched_frame['action'].to(device)

                policy._cache["training"] = False

                # ✅ TTSInferwrapperbatchprunergate
                pruned_action_batch = policy.predict_action(obs_batch)['action_pred']

                gate_hard = policy._cache['gate']
                
                gate_stats = calculate_gate_statistics_stage2(gate_hard)
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

                valid_loss_sum += loss.item()
                valid_consistency_sum += Lc.item() if Lc is not None else 0.0
                valid_sparse_sum += Ls.item() if Ls is not None else 0.0
                valid_prune_sum += current_pruning_ratio
                valid_count += 1
                last_gate_stats = gate_stats
            
            if valid_count > 0:
                pbar_val.set_postfix({
                    'avg_loss': f'{valid_loss_sum / valid_count:.4f}',
                    'avg_prune': f'{valid_prune_sum / valid_count:.2%}'
                })

    if valid_count == 0:
        avg_metrics = {
            'valid_loss': 0.0,
            'valid_consistency_loss': 0.0,
            'valid_sparse_loss': 0.0,
            'valid_pruning_ratio': 0.0,
            'gate_stats': last_gate_stats,
        }
    else:
        avg_metrics = {
            'valid_loss': valid_loss_sum / valid_count,
            'valid_consistency_loss': valid_consistency_sum / valid_count,
            'valid_sparse_loss': valid_sparse_sum / valid_count,
            'valid_pruning_ratio': valid_prune_sum / valid_count,
            'gate_stats': last_gate_stats,
        }

    if is_distributed:
        avg_metrics = _all_reduce_metrics(avg_metrics, device)

    return avg_metrics


def _get_dist_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    return is_distributed, rank, world_size, local_rank


def _all_reduce_metrics(metrics: dict, device: torch.device) -> dict:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    reduced = {}
    for k, v in metrics.items():
        if k == "gate_stats":
            reduced[k] = v
            continue
        t = torch.tensor(float(v), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
        reduced[k] = t.item()
    return reduced


def _broadcast_stop_flag(stop: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return stop
    t = torch.tensor(1 if stop else 0, device=device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return bool(t.item())


def _get_rank_episode_indices(num_episodes: int, world_size: int, rank: int):
    """episodesrankrank batch"""
    if world_size <= 1:
        return None
    indices = list(range(num_episodes))
    per_rank = (num_episodes + world_size - 1) // world_size
    total_needed = per_rank * world_size
    if total_needed > num_episodes:
        indices.extend(indices[: (total_needed - num_episodes)])
    start = rank * per_rank
    end = start + per_rank
    return indices[start:end]


def _count_episodes(data_dir: Path, max_episodes=None) -> int:
    episodes_dir = data_dir / "episodes"
    episode_dirs = sorted(episodes_dir.glob("episode_*"))
    if max_episodes is not None:
        return min(len(episode_dirs), max_episodes)
    return len(episode_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Policy checkpoint path')
    parser.add_argument('--trajectory_dir', type=str, required=True, help='Trajectory')
    parser.add_argument('--output_dir', type=str, default='output/pruner_tts', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = _get_dist_env()
    is_main = (rank == 0)

    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['basic'].get('seed', 42)
    set_seed(seed + rank)

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if is_main:
        logger.info(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = config['basic']['task_name']
    target_ratio = config['pruning']['target_prune_ratio']
    output_dir = Path(args.output_dir) / task_name / f"{target_ratio:.3f}" / timestamp
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f": {output_dir}")

    #  wandb
    use_wandb = config['basic'].get('use_wandb', False)
    if use_wandb and is_main:
        wandb.init(
            project="TTS-Real",
            name=f"{task_name}_prune{target_ratio:.2f}_{timestamp}",
            config={**config, "version": "TTS-Trajectory"},
            tags=["pruner", "real-world", "TTS", "trajectory", task_name],
            dir=str(output_dir),  # wandb 
            settings=wandb.Settings(git_root=None)
        )
        logger.info("✓ Wandb initialized")

    if is_main:
        logger.info(f": {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model.to(device)
    policy.eval()
    policy.requires_grad_(False)  #  policy 
    if is_main:
        logger.info("")

    #  workspace  init
    if is_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    # ✅ Trajectoryepisode-based
    if is_main:
        logger.info(f"\nTrajectory: {args.trajectory_dir}")
    trajectory_path = Path(args.trajectory_dir)

    train_dir = trajectory_path / "train"
    val_dir = trajectory_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Trajectory!\n"
            f" convert_zarr_to_trajectory.py \n"
            f": {train_dir}  {val_dir}"
        )

    train_config = config.get('trajectory_training', config.get('training', {}))
    num_train_episodes = train_config.get('num_train_episodes', None)
    num_val_episodes = train_config.get('num_val_episodes', None)

    # rankrankvalval
    train_indices = None
    if is_distributed:
        train_count = _count_episodes(train_dir, num_train_episodes)
        train_indices = _get_rank_episode_indices(train_count, world_size, rank)
        if is_main:
            logger.info(f":  {len(train_indices)} train episodes")

    train_dataset = TrajectoryDataset(
        data_dir=str(train_dir),
        device='cpu',
        max_episodes=num_train_episodes,
        episode_indices=train_indices,
        n_obs_steps=cfg.n_obs_steps  # ✅ 
    )

    # val
    val_dataset = TrajectoryDataset(
        data_dir=str(val_dir),
        device='cpu',
        max_episodes=num_val_episodes,
        episode_indices=None,  # 
        n_obs_steps=cfg.n_obs_steps  # ✅ 
    )

    if is_main:
        logger.info(f"✓ episodes: {len(train_dataset)}")
        logger.info(f"✓ episodes: {len(val_dataset)}")
        logger.info(f"✓ : {train_dataset.unified_length} frames/episode")

    # frame_batch_size  trajectory Config frame_batch_sizeepisode_batch_size
    # -trajectory Config training  assembly_chocolate.yaml
    #  training.batch_size
    episode_batch_size = train_config.get('frame_batch_size', train_config.get('batch_size', 32))
    if is_main:
        logger.info(f"✓ Episode batch size: {episode_batch_size}")

    # ✅ PrunerTTSInfer4gate
    if is_main:
        logger.info("\n TTS Pruner rollout cache")
    pruner_config = config['model']
    num_steps = 100

    #  obs_dimGet from DP model
    model = policy.model
    obs_dim = model.cond_obs_emb.out_features if hasattr(model, 'cond_obs_emb') else 512

    #  wrapper  block_names
    CachePrunerWrapper.apply(policy, pruner=None, if_rollout_cache=True, training=True)
    block_names = policy._cache_block_keys
    if is_main:
        logger.info(f" {len(block_names)}  blocks")

    #  Pruner4gate
    block_encoder_type = pruner_config.get('block_encoder_type', 'SA')
    
    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_names,
        hidden_dim=pruner_config['hidden_dim'],
        decoder_layers=pruner_config.get('decoder_layers', 1),
        block_encoder_type=block_encoder_type,
        attn_heads=pruner_config.get('attn_heads', 8),
        dim_feedforward=pruner_config.get('dim_feedforward', 1024),
        obs_dim=obs_dim,
        dropout=pruner_config.get('dropout', 0.1),
        head_4=True,  # 4gatecompute, 3cache, 24cache, rollout_cache
    ).to(device)

    if is_main:
        logger.info("✓ TTS Pruner 4gate")

    if is_distributed and dist.is_initialized():
        pruner = DDP(
            pruner,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True
        )

    # ✅ wrapperprunerTTSInfer
    CachePrunerWrapper.apply(policy, pruner=pruner, if_rollout_cache=True, training=True)
    if is_main:
        logger.info("✓ Wrappergatepruner")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        pruner.parameters(),
        lr=train_config['lr'],
        betas=train_config['optimizer']['betas'],
        weight_decay=train_config['optimizer']['weight_decay'].get('transformer', 0.001)
    )

    # Create learning rate scheduler
    warmup_steps = train_config.get('warmup_steps', 0)

    def warmup_linear_scheduler(epoch):
        if epoch < warmup_steps:
            warmup_factor = train_config.get('min_lr', 1e-7) / train_config['lr'] + float(epoch) / float(max(1, warmup_steps))
            return warmup_factor
        else:
            remaining_epochs = train_config['epochs'] - warmup_steps
            if remaining_epochs <= 0:
                return 1.0

            decay_progress = float(epoch - warmup_steps) / float(remaining_epochs)
            decay_progress = min(decay_progress, 1.0)

            min_lr_ratio = train_config.get('min_lr', 1e-7) / train_config['lr']
            lr_factor = 1.0 - decay_progress * (1.0 - min_lr_ratio)
            return max(min_lr_ratio, lr_factor)

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_linear_scheduler)

    class TrainArgs:
        def __init__(self, config):
            self.target_prune_ratio = config['pruning']['target_prune_ratio']
            self.consistency = config['loss']['consistency']
            self.sparse = config['loss']['sparse']
            self.lamb_sparse = config['loss']['lamb_sparse']
            self.global_lc = config['loss'].get('global_lc', False)

    train_args = TrainArgs(config)

    grad_accum_steps = train_config.get('grad_accum_steps', 4)  # 
    use_amp = train_config.get('use_amp', True)  # 
    trajectory_sample_ratio = train_config.get('trajectory_sample_ratio', 1.0)  # 
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_loss = float('inf')
    patience_counter = 0
    patience = train_config.get('patience', 10)

    if is_main:
        logger.info(f"\n TTS Pruner (Episode-based) - {train_config['epochs']}  epochs...")
        logger.info(f"Config: ={grad_accum_steps}, AMP={use_amp}, ={trajectory_sample_ratio:.0%}")
        logger.info("=" * 70)

    # Epoch
    pbar_epoch = tqdm(range(train_config['epochs']), desc="Training Progress", ncols=120, disable=(not is_main))
    
    for epoch in pbar_epoch:
        train_metrics = train_epoch_trajectory(
            epoch, pruner, policy, train_dataset,
            episode_batch_size, num_steps, block_names, train_args, optimizer, scheduler, device,
            is_distributed=is_distributed, rank=rank, world_size=world_size,
            grad_accum_steps=grad_accum_steps, use_amp=use_amp, scaler=scaler,
            trajectory_sample_ratio=trajectory_sample_ratio
        )

        valid_metrics = validate_epoch_trajectory(
            pruner, policy, val_dataset,
            episode_batch_size, num_steps, block_names, train_args, device,
            is_distributed=is_distributed, rank=rank, world_size=world_size
        )

        scheduler.step()

        # epoch
        if is_main:
            pbar_epoch.set_postfix({
                'train_loss': f'{train_metrics["train_loss"]:.4f}',
                'val_loss': f'{valid_metrics["valid_loss"]:.4f}',
                'train_prune': f'{train_metrics["train_pruning_ratio"]:.2%}',
                'val_prune': f'{valid_metrics["valid_pruning_ratio"]:.2%}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        if is_main:
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Epoch {epoch+1}/{train_config['epochs']}")
            tqdm.write(f"  Train - Loss: {train_metrics['train_loss']:.4f}, Prune: {train_metrics['train_pruning_ratio']:.2%}")
            tqdm.write(f"  Valid - Loss: {valid_metrics['valid_loss']:.4f}, Prune: {valid_metrics['valid_pruning_ratio']:.2%}")
            tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if use_wandb and is_main:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics['train_loss'],
                "train/consistency_loss": train_metrics['train_consistency_loss'],
                "train/sparse_loss": train_metrics['train_sparse_loss'],
                "train/prune_ratio": train_metrics['train_pruning_ratio'],
                "valid/loss": valid_metrics['valid_loss'],
                "valid/prune_ratio": valid_metrics['valid_pruning_ratio'],
                "lr": optimizer.param_groups[0]['lr'],
            })

        if is_main:
            epoch_checkpoint_path = output_dir / f'epoch_{epoch+1:03d}_loss_{valid_metrics["valid_loss"]:.4f}.pt'
            state_dict = pruner.module.state_dict() if hasattr(pruner, "module") else pruner.state_dict()
            save_pruner_ckpt(
                path=str(epoch_checkpoint_path),
                state_dict=state_dict
            )
            tqdm.write(f"  ✓ epoch: {epoch_checkpoint_path.name}")

        if is_main:
            if valid_metrics['valid_loss'] < best_val_loss:
                best_val_loss = valid_metrics['valid_loss']
                best_epoch = epoch + 1
                patience_counter = 0

                checkpoint_path = output_dir / 'best_pruner.pt'
                state_dict = pruner.module.state_dict() if hasattr(pruner, "module") else pruner.state_dict()
                save_pruner_ckpt(
                    path=str(checkpoint_path),
                    state_dict=state_dict
                )
                tqdm.write(f"  ✓ : {checkpoint_path}")
            else:
                patience_counter += 1

            if train_config.get('early_stop', True) and patience_counter >= patience:
                tqdm.write(f"\nEarly stopping triggered after {epoch+1} epochs")
                stop_training = True
            else:
                stop_training = False
        else:
            stop_training = False

        stop_training = _broadcast_stop_flag(stop_training, device)
        if stop_training:
            break

        if is_main:
            tqdm.write("=" * 70)

    if is_main:
        logger.info(f"\n! : {best_val_loss:.4f}")
        logger.info(f": {output_dir / 'best_pruner.pt'}")

        last_epoch = epoch + 1
        if train_config.get('early_stop', True) and patience_counter >= patience:
            logger.info(" early_stop")
        elif 'best_epoch' in locals() and best_epoch == last_epoch:
            logger.info("epochs")
        else:
            logger.info("")

        if use_wandb:
            wandb.log({"final/best_val_loss": best_val_loss})
            wandb.save(str(output_dir / 'best_pruner.pt'))
            wandb.finish()
            logger.info("✓ Wandb logging finished")


if __name__ == '__main__':
    main()
