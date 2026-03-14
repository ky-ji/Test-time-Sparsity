"""
Convert Real-world Zarr Dataset to Trajectory Format for TTS Training

This script converts CogActImageDataset (zarr format) to TrajectoryDataset format
used by TTSInfer for episode-based training with rollout cache.

Usage:
    python convert_zarr_to_trajectory.py \
        --zarr_path /path/to/data.zarr \
        --output_dir /path/to/output \
        --checkpoint /path/to/policy.ckpt \
        --train_ratio 0.9
"""

import sys
import os
from pathlib import Path
import argparse
import json
import torch
import dill
import hydra
import zarr
import numpy as np
from tqdm import tqdm

# Add necessary paths
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent                # realworld-TTS/
project_root = repo_root.parent               # Test-time-Sparsity/ (where diffusion_policy submodule lives)

for p in (current_dir, repo_root, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diffusion_policy.dataset.cogact_image_dataset import CogActImageDataset
from diffusion_policy.common.pytorch_util import dict_apply


def load_policy_config(checkpoint_path):
    """Load policy configuration from checkpoint"""
    print(f"Loading policy config from: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    return cfg


def create_trajectory_structure(output_dir):
    """Create directory structure for trajectory dataset"""
    output_path = Path(output_dir)
    episodes_dir = output_path / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    return output_path, episodes_dir


def save_episode(episode_dir, frames, episode_idx, episode_length):
    """
    Save a single episode in TrajectoryDataset format

    Args:
        episode_dir: Directory to save episode
        frames: List of frame dicts with {obs, action, episode_idx, frame_idx}
        episode_idx: Episode index
        episode_length: Number of frames in episode
    """
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory data
    trajectory_path = episode_dir / "trajectory.pt"
    torch.save({
        'frames': frames,
        'episode_idx': episode_idx,
        'length': episode_length
    }, trajectory_path)

    # Save metadata
    metadata = {
        'episode_idx': episode_idx,
        'length': episode_length,
        'num_frames': episode_length
    }
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def convert_zarr_to_trajectory(
    zarr_path: str,
    output_dir: str,
    checkpoint_path: str,
    train_ratio: float = 0.9,
    seed: int = 42
):
    """
    Convert zarr dataset to trajectory format

    Args:
        zarr_path: Path to zarr dataset
        output_dir: Output directory for trajectory data
        checkpoint_path: Path to policy checkpoint (for loading config)
        train_ratio: Ratio of training data (rest is validation)
        seed: Random seed
    """
    print("=" * 80)
    print("Converting Zarr Dataset to Trajectory Format")
    print("=" * 80)
    print(f"Zarr path:    {zarr_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Train ratio:  {train_ratio}")
    print(f"Seed:         {seed}")
    print("=" * 80)

    # Load policy config
    cfg = load_policy_config(checkpoint_path)

    # Create output structure
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    train_path, train_episodes_dir = create_trajectory_structure(train_dir)
    val_path, val_episodes_dir = create_trajectory_structure(val_dir)

    # Load zarr dataset
    print("\nLoading zarr dataset...")
    train_dataset_raw = CogActImageDataset(
        shape_meta=cfg.task.shape_meta,
        zarr_path=zarr_path,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        n_obs_steps=cfg.n_obs_steps,
        seed=seed,
        val_ratio=1.0 - train_ratio  # Use val_ratio for splitting
    )

    val_dataset_raw = train_dataset_raw.get_validation_dataset()

    print(f"✓ Loaded dataset")
    print(f"  Train samples: {len(train_dataset_raw)}")
    print(f"  Val samples:   {len(val_dataset_raw)}")

    # Get episode information from zarr
    zarr_root = zarr.open(zarr_path, 'r')
    episode_ends = zarr_root['meta']['episode_ends'][:]

    # Calculate episode starts
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    num_episodes = len(episode_ends)

    print(f"\nDataset structure:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Episode lengths: min={min(episode_ends - episode_starts)}, "
          f"max={max(episode_ends - episode_starts)}, "
          f"mean={np.mean(episode_ends - episode_starts):.1f}")

    # Split episodes into train/val
    num_train_episodes = int(num_episodes * train_ratio)
    train_episode_indices = list(range(num_train_episodes))
    val_episode_indices = list(range(num_train_episodes, num_episodes))

    print(f"\nSplitting episodes:")
    print(f"  Train episodes: {len(train_episode_indices)} (indices 0-{num_train_episodes-1})")
    print(f"  Val episodes:   {len(val_episode_indices)} (indices {num_train_episodes}-{num_episodes-1})")

    # Convert training episodes
    print("\n" + "=" * 80)
    print("Converting Training Episodes")
    print("=" * 80)

    train_stats = convert_episodes(
        zarr_root=zarr_root,
        episode_indices=train_episode_indices,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        episodes_dir=train_episodes_dir,
        split_name="train",
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon
    )

    # Save train dataset summary
    train_summary = {
        'num_episodes': len(train_episode_indices),
        'total_frames': train_stats['total_frames'],
        'min_length': train_stats['min_length'],
        'max_length': train_stats['max_length'],
        'mean_length': train_stats['mean_length'],
        'split': 'train'
    }
    with open(train_path / "dataset_summary.json", 'w') as f:
        json.dump(train_summary, f, indent=2)

    # Convert validation episodes
    print("\n" + "=" * 80)
    print("Converting Validation Episodes")
    print("=" * 80)

    val_stats = convert_episodes(
        zarr_root=zarr_root,
        episode_indices=val_episode_indices,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        episodes_dir=val_episodes_dir,
        split_name="val",
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon
    )

    # Save val dataset summary
    val_summary = {
        'num_episodes': len(val_episode_indices),
        'total_frames': val_stats['total_frames'],
        'min_length': val_stats['min_length'],
        'max_length': val_stats['max_length'],
        'mean_length': val_stats['mean_length'],
        'split': 'val'
    }
    with open(val_path / "dataset_summary.json", 'w') as f:
        json.dump(val_summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Train data: {train_dir}")
    print(f"  Episodes: {train_summary['num_episodes']}")
    print(f"  Frames:   {train_summary['total_frames']}")
    print(f"  Length:   {train_summary['min_length']}-{train_summary['max_length']} (mean: {train_summary['mean_length']:.1f})")
    print(f"\nVal data: {val_dir}")
    print(f"  Episodes: {val_summary['num_episodes']}")
    print(f"  Frames:   {val_summary['total_frames']}")
    print(f"  Length:   {val_summary['min_length']}-{val_summary['max_length']} (mean: {val_summary['mean_length']:.1f})")
    print("=" * 80)


def convert_episodes(zarr_root, episode_indices, episode_starts, episode_ends, episodes_dir, split_name, n_obs_steps=1, horizon=16):
    """
    Convert a set of episodes to trajectory format

    zarrCogActImageDatasettrain/val split

    Args:
        zarr_root: Zarr root object
        episode_indices: List of episode indices to convert
        episode_starts: Array of episode start indices
        episode_ends: Array of episode end indices
        episodes_dir: Directory to save episodes
        split_name: "train" or "val"
        n_obs_steps: Number of observation steps
        horizon: Action horizon length (TTSInferhorizon)

    Returns:
        Dict with statistics
    """
    total_frames = 0
    episode_lengths = []

    # zarr
    data_group = zarr_root['data']

    for ep_idx in tqdm(episode_indices, desc=f"Converting {split_name} episodes"):
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        episode_length = end_idx - start_idx

        # Collect all frames in this episode (raw single-frame observations)
        raw_frames = []
        for frame_idx in range(episode_length):
            zarr_idx = start_idx + frame_idx

            try:
                # zarr
                frame_data = {}
                for key in data_group.keys():
                    data_item = data_group[key][zarr_idx]
                    if isinstance(data_item, np.ndarray):
                        frame_data[key] = torch.from_numpy(data_item)
                    else:
                        # tensor
                        frame_data[key] = torch.tensor(data_item)

                # obsaction
                obs = {}
                action = None

                for key, value in frame_data.items():
                    if key == 'action':
                        action = value
                    elif key == 'timestamp':
                        continue
                    else:
                        # ✅ 3D4D tensorCHW
                        if value.ndim >= 3 and value.shape[-1] in [1, 3, 4]:  # 
                            #  HWC  CHW
                            if value.ndim == 3:  # [H, W, C]
                                value = value.permute(2, 0, 1)  # -> [C, H, W]
                            elif value.ndim == 4:  # [T, H, W, C] (unlikely in single frame)
                                value = value.permute(0, 3, 1, 2)  # -> [T, C, H, W]
                        obs[key] = value

                if action is None:
                    print(f"\nWarning: No action found for episode {ep_idx}, frame {frame_idx}")
                    continue

                #  n_obs_steps
                raw_frame = {
                    'obs': obs,
                    'action': action,
                    'episode_idx': ep_idx,
                    'frame_idx': frame_idx
                }
                raw_frames.append(raw_frame)

            except (IndexError, KeyError) as e:
                print(f"\nWarning: Could not access zarr index {zarr_idx} for episode {ep_idx}, frame {frame_idx}: {e}")
                continue

        if len(raw_frames) == 0:
            print(f"\nWarning: Episode {ep_idx} has no valid frames, skipping")
            continue

        # ✅  + horizonactionTTSInfer/CogActImageDataset
        # TrajectoryDatasetn_obs_stepsactionhorizon
        frames = []
        for i in range(len(raw_frames)):
            frame_idx = raw_frames[i]['frame_idx']

            #  horizon  action SequenceSampler
            action_sequence = []
            for h in range(horizon):
                future_idx = i + h
                if future_idx < len(raw_frames):
                    action_sequence.append(raw_frames[future_idx]['action'])
                else:
                    # SequenceSampler
                    action_sequence.append(raw_frames[-1]['action'])
            
            # Stack into [horizon, action_dim]
            action_horizon = torch.stack(action_sequence, dim=0)

            frame = {
                'obs': raw_frames[i]['obs'],      # shape: [C, H, W] for images, [D] for low_dim
                'action': action_horizon,          # horizonactionshape: [horizon, action_dim]
                'episode_idx': ep_idx,
                'frame_idx': frame_idx
            }
            frames.append(frame)

        if len(frames) == 0:
            print(f"\nWarning: Episode {ep_idx} has no valid frames, skipping")
            continue

        # Save episode
        episode_dir = episodes_dir / f"episode_{ep_idx}"
        save_episode(episode_dir, frames, ep_idx, len(frames))

        total_frames += len(frames)
        episode_lengths.append(len(frames))

    stats = {
        'total_frames': total_frames,
        'min_length': min(episode_lengths) if episode_lengths else 0,
        'max_length': max(episode_lengths) if episode_lengths else 0,
        'mean_length': np.mean(episode_lengths) if episode_lengths else 0
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert zarr dataset to trajectory format")
    parser.add_argument('--zarr_path', type=str, required=True, help='Path to zarr dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trajectory data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to policy checkpoint')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    convert_zarr_to_trajectory(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
