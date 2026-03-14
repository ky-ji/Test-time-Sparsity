"""
Trajectory Dataset for Stage 2 Pruner Training

(episode)(frame)

1. episodeframeepisode
2. episode
3. episodeframe


- episode
- episode
- truncate_lengthpad_length
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
     - collect_trajectory_data.py
    
    :
    - episodetrajectory.pt
    - frame: {obs: Dict[str, Tensor], action: Tensor, episode_idx: int, frame_idx: int}
    
    :
    - episodeframepad_length
    - episode
    """
    
    def __init__(
        self,
        data_dir: str,
        pad_length: Optional[int] = None,
        truncate_length: Optional[int] = None,
        pad_value: float = 0.0,
        device: str = 'cpu',
        max_episodes: Optional[int] = None,
        episode_indices: Optional[List[int]] = None,
        n_obs_steps: int = 1
    ):
        """
        Args:
            data_dir:  (episodes/)
            pad_length: episodeNone
            truncate_length: episodepad_length
            pad_value: 
            device: Device (cpu)
            max_episodes: episodeNoneepisodes
            episode_indices: episodeNone
        """
        self.data_dir = Path(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        self.pad_value = pad_value
        self.device = device
        self.n_obs_steps = n_obs_steps
        
        if not self.episodes_dir.exists():
            raise FileNotFoundError(f"Episodes directory not found: {self.episodes_dir}")
        
        # dataset summary
        summary_path = self.data_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = None
        
        # episode
        self.episode_paths = []
        self.episode_metadata = []
        
        episode_dirs = sorted(self.episodes_dir.glob("episode_*"))
        
        # episode_indicesepisodes
        if episode_indices is not None:
            print(f"Loading specified {len(episode_indices)} episodes by indices")
            episode_dirs_dict = {int(ep_dir.name.split('_')[1]): ep_dir for ep_dir in episode_dirs}
            episode_dirs = [episode_dirs_dict[idx] for idx in episode_indices if idx in episode_dirs_dict]
        
        for episode_dir in episode_dirs:
            trajectory_path = episode_dir / "trajectory.pt"
            metadata_path = episode_dir / "metadata.json"
            
            if trajectory_path.exists() and metadata_path.exists():
                self.episode_paths.append(trajectory_path)
                
                # metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.episode_metadata.append(metadata)
            
            # max_episodes
            if max_episodes is not None and len(self.episode_paths) >= max_episodes:
                print(f"Reached max_episodes limit: {max_episodes}")
                break
        
        if len(self.episode_paths) == 0:
            raise ValueError(f"No valid episodes found in {self.episodes_dir}")
        
        print(f"Found {len(self.episode_paths)} episodes in {data_dir}")
        
        if truncate_length is not None:
            self.unified_length = truncate_length
            print(f"Using truncate_length: {self.unified_length}")
        elif pad_length is not None:
            self.unified_length = pad_length
            print(f"Using pad_length: {self.unified_length}")
        else:
            # episode
            min_length = min(meta['length'] for meta in self.episode_metadata)
            self.unified_length = min_length
            print(f"Auto-detected min_length: {self.unified_length} (using shortest episode, others will be truncated)")
        
        print("Loading and unifying all episodes...")
        self.episodes = []
        for i, (episode_path, metadata) in enumerate(zip(self.episode_paths, self.episode_metadata)):
            episode_data = torch.load(episode_path, map_location='cpu')
            frames = episode_data['frames']
            
            unified_frames = self._unify_episode_length(frames, metadata['length'])
            self.episodes.append({
                'episode_idx': metadata['episode_idx'],
                'frames': unified_frames,  # List of {obs, action, episode_idx, frame_idx}
                'original_length': metadata['length'],
                'unified_length': self.unified_length
            })
            
            if (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(self.episode_paths)} episodes")
        
        print(f"✓ Successfully loaded {len(self.episodes)} episodes")
        print(f"✓ Unified length: {self.unified_length} frames per episode")
    
    def _unify_episode_length(self, frames: List[Dict], original_length: int) -> List[Dict]:
        """
        episode
        
        Args:
            frames: frame
            original_length: 
            
        Returns:
            frameunified_length
        """
        if original_length > self.unified_length:
            # unified_length
            return frames[:self.unified_length]
        else:
            # unified_length
            # unified_length
            return frames
    
    def __len__(self) -> int:
        """episode"""
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        episode
        
        Returns:
            Dict with keys:
                - episode_idx: int
                - frames: List of {obs, action, episode_idx, frame_idx}
                - original_length: int
                - unified_length: int
        """
        return self.episodes[idx]
    
    def get_episode_iterator(self, frame_batch_size: int = 1):
        """
        episodeepisodeframe_batch
        
        Args:
            frame_batch_size: frame batch
            
        Yields:
            yieldepisodeframe batches (List of batched frames)
        """
        for episode in self.episodes:
            frames = episode['frames']
            episode_batches = []
            
            # framesbatches
            for i in range(0, len(frames), frame_batch_size):
                frame_batch = frames[i:i + frame_batch_size]
                
                # frame_batchbatched
                # frame_batchList[Dict]Dict[List]
                batched_frame = self._collate_frames(frame_batch)
                episode_batches.append(batched_frame)
            
            yield episode_batches
    
    def get_episode_batch_iterator(self, episode_batch_size: int, shuffle: bool = False):
        """
        episode batchbatchframe

        
        1. episode_batchepisodes 0-31, 32-63, ...
        2. episode_batchframe_idxframe 0, 1, 2, ...
        3. episode_batchrolloutrollout cache

        Args:
            episode_batch_size: batchepisode
            shuffle: episodes

        Yields:
            yieldepisode_batchframes
            : List[Dict], Dictbatchframe_idx
        """
        num_episodes = len(self.episodes)

        # episode
        episode_indices = list(range(num_episodes))

        # shuffle
        if shuffle:
            import random
            random.shuffle(episode_indices)

        num_batches = (num_episodes + episode_batch_size - 1) // episode_batch_size

        # episode batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * episode_batch_size
            end_idx = min(start_idx + episode_batch_size, num_episodes)

            # episodesshuffle
            batch_indices = episode_indices[start_idx:end_idx]
            episodes_in_batch = [self.episodes[i] for i in batch_indices]
            num_frames = self.unified_length

            # batchframes
            batch_frames = []
            for frame_idx in range(num_frames):
                # batchepisodesframe_idx
                frame_list = []
                for episode in episodes_in_batch:
                    #  max(0, frame_idx - n_obs_steps + 1)  frame_idx
                    start_window = max(0, frame_idx - self.n_obs_steps + 1)
                    window_frames = episode['frames'][start_window:frame_idx + 1]

                    #  n_obs_steps
                    if len(window_frames) < self.n_obs_steps:
                        first_frame = episode['frames'][0]
                        padding_needed = self.n_obs_steps - len(window_frames)
                        window_frames = [first_frame] * padding_needed + window_frames

                    obs_window = {}
                    for key in window_frames[0]['obs'].keys():
                        # Stack: [n_obs_steps, ...]
                        stacked = torch.stack([f['obs'][key] for f in window_frames], dim=0)
                        obs_window[key] = stacked

                    # action
                    frame_with_window = {
                        'obs': obs_window,
                        'action': episode['frames'][frame_idx]['action'],
                        'episode_idx': episode['frames'][frame_idx]['episode_idx'],
                        'frame_idx': episode['frames'][frame_idx]['frame_idx']
                    }
                    frame_list.append(frame_with_window)

                # Collatebatch
                batched_frame = self._collate_frames(frame_list)
                batch_frames.append(batched_frame)

            # Yieldepisode_batchframes
            yield batch_frames
    
    def _collate_frames(self, frame_batch: List[Dict]) -> Dict:
        """
        frame list collatebatch
        
        Args:
            frame_batch: List of {obs, action, episode_idx, frame_idx}
            
        Returns:
            Dict with:
                - obs: Dict[str, Tensor] with batch dimension
                - action: Tensor with batch dimension
                - episode_idx: Tensor
                - frame_idx: Tensor
        """
        # obs keys
        obs_keys = frame_batch[0]['obs'].keys()
        
        batched_obs = {}
        for key in obs_keys:
            # Stackframeobs key
            obs_list = [frame['obs'][key] for frame in frame_batch]
            batched_obs[key] = torch.stack(obs_list, dim=0)  # [batch_size, ...]
        
        # Stack actions
        action_list = [frame['action'] for frame in frame_batch]
        batched_action = torch.stack(action_list, dim=0)  # [batch_size, ...]
        
        # Collect metadata
        episode_idx_list = [frame['episode_idx'] for frame in frame_batch]
        frame_idx_list = [frame['frame_idx'] for frame in frame_batch]
        
        return {
            'obs': batched_obs,
            'action': batched_action,
            'episode_idx': torch.tensor(episode_idx_list),
            'frame_idx': torch.tensor(frame_idx_list)
        }
    
    def get_stats(self) -> Dict:
        """"""
        stats = {
            'num_episodes': len(self.episodes),
            'unified_length': self.unified_length,
            'total_frames': len(self.episodes) * self.unified_length,
        }
        
        original_lengths = [ep['original_length'] for ep in self.episodes]
        stats['original_lengths'] = {
            'min': min(original_lengths),
            'max': max(original_lengths),
            'mean': sum(original_lengths) / len(original_lengths)
        }
        
        return stats


def create_trajectory_dataset(
    data_dir: str,
    pad_length: Optional[int] = None,
    truncate_length: Optional[int] = None,
    pad_value: float = 0.0,
    device: str = 'cpu'
) -> TrajectoryDataset:
    """
    
    
    Args:
        data_dir: 
        pad_length: 
        truncate_length: 
        pad_value: 
        device: Device
        
    Returns:
        TrajectoryDataset
    """
    return TrajectoryDataset(
        data_dir=data_dir,
        pad_length=pad_length,
        truncate_length=truncate_length,
        pad_value=pad_value,
        device=device
    )


def create_trajectory_dataloaders(
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    pad_length: Optional[int] = None,
    truncate_length: Optional[int] = None,
    pad_value: float = 0.0,
    device: str = 'cpu'
) -> Tuple[TrajectoryDataset, Optional[TrajectoryDataset]]:
    """
    
    
    Args:
        train_data_dir: 
        val_data_dir: 
        pad_length: 
        truncate_length: 
        pad_value: 
        device: Device
        
    Returns:
        (train_dataset, val_dataset) tuple
    """
    print("=" * 80)
    print("Creating Trajectory Datasets")
    print("=" * 80)
    
    print("\n[Train Dataset]")
    train_dataset = create_trajectory_dataset(
        data_dir=train_data_dir,
        pad_length=pad_length,
        truncate_length=truncate_length,
        pad_value=pad_value,
        device=device
    )
    
    train_stats = train_dataset.get_stats()
    print("\nTrain Dataset Stats:")
    print(f"  - Episodes: {train_stats['num_episodes']}")
    print(f"  - Unified length: {train_stats['unified_length']}")
    print(f"  - Total frames: {train_stats['total_frames']}")
    print(f"  - Original length range: [{train_stats['original_lengths']['min']}, {train_stats['original_lengths']['max']}]")
    print(f"  - Original length mean: {train_stats['original_lengths']['mean']:.2f}")
    
    val_dataset = None
    if val_data_dir is not None:
        print("\n[Validation Dataset]")
        val_dataset = create_trajectory_dataset(
            data_dir=val_data_dir,
            pad_length=pad_length,
            truncate_length=truncate_length,
            pad_value=pad_value,
            device=device
        )
        
        val_stats = val_dataset.get_stats()
        print("\nValidation Dataset Stats:")
        print(f"  - Episodes: {val_stats['num_episodes']}")
        print(f"  - Unified length: {val_stats['unified_length']}")
        print(f"  - Total frames: {val_stats['total_frames']}")
        print(f"  - Original length range: [{val_stats['original_lengths']['min']}, {val_stats['original_lengths']['max']}]")
        print(f"  - Original length mean: {val_stats['original_lengths']['mean']:.2f}")
    
    print("\n" + "=" * 80)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Trajectory Dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--pad_length', type=int, default=None, help='Pad length')
    parser.add_argument('--truncate_length', type=int, default=None, help='Truncate length')
    parser.add_argument('--frame_batch_size', type=int, default=4, help='Frame batch size')
    
    args = parser.parse_args()
    
    dataset = create_trajectory_dataset(
        data_dir=args.data_dir,
        pad_length=args.pad_length,
        truncate_length=args.truncate_length
    )
    
    print("\n" + "="*80)
    print("Testing Dataset")
    print("="*80)
    
    # __getitem__
    print(f"\nTest 1: Get first episode")
    episode = dataset[0]
    print(f"  Episode idx: {episode['episode_idx']}")
    print(f"  Unified length: {episode['unified_length']}")
    print(f"  Original length: {episode['original_length']}")
    print(f"  First frame obs keys: {list(episode['frames'][0]['obs'].keys())}")
    print(f"  First frame action shape: {episode['frames'][0]['action'].shape}")
    

    
    # frame iterator
    print(f"\nTest 3: Frame Iterator (NEW) - Cross-episode batching")
    print(f"  Using first 3 episodes, batch size = num_episodes")
    frame_count = 0
    for frame_batches in dataset.get_frame_iterator(num_episodes=3):
        frame_batch = frame_batches[0]
        print(f"\n  Frame {frame_count}:")
        print(f"    Batch contains data from multiple episodes at the same frame position")
        print(f"    - obs keys: {list(frame_batch['obs'].keys())}")
        for key, val in frame_batch['obs'].items():
            print(f"      {key}: {val.shape}")
        print(f"    - action shape: {frame_batch['action'].shape}")
        print(f"    - episode_idx: {frame_batch['episode_idx']}")
        print(f"    - frame_idx: {frame_batch['frame_idx']}")

    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

