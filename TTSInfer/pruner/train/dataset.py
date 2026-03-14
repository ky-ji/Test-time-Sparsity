from typing import Dict, Any, Optional
import torch
import random
import pickle
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class PrunerTrainingDataset(Dataset):  
    def __init__(self, cfg, device=None, use_real_data=False, use_target_action=False, 
                 source_dataset=None, n_obs_steps=2, num_samples=1000, fixed_indices=None,
                 precomputed_data_path: Optional[str] = None):
        """
        Pruner
        
        Args:
            cfg: Config object
            device: Device
            use_real_data:  ()
            use_target_action: 
            source_dataset: 
            n_obs_steps: 
            num_samples: 
            fixed_indices: 
            precomputed_data_path:  ()
        """
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cpu')
        self.use_real_data = use_real_data
        self.use_target_action = use_target_action
        self.source_dataset = source_dataset
        self.n_obs_steps = n_obs_steps
        self.num_samples = num_samples
        self.fixed_indices = fixed_indices
        self.precomputed_data_path = precomputed_data_path
        
        if precomputed_data_path and os.path.exists(precomputed_data_path):
            print(f": {precomputed_data_path}")
            self._load_precomputed_data()
        else:
            if precomputed_data_path:
                print(f": {precomputed_data_path}")
                print("")
            
            if fixed_indices is not None:
                assert len(fixed_indices) >= num_samples, f" ({len(fixed_indices)})  ({num_samples})"
                self.sample_indices = fixed_indices[:num_samples]  # num_samples
            else:
                self.sample_indices = None
            
            self.data = []
            print(f" {num_samples} ...")
            for i in range(num_samples):
                sample = self._generate_sample(i)
                self.data.append(sample)
                if (i + 1) % 1000 == 0:
                    print(f" {i + 1}/{num_samples} ")
    
    def _load_precomputed_data(self):
        """"""
        try:
            with open(self.precomputed_data_path, 'rb') as f:
                all_data = pickle.load(f)
            
            if self.num_samples < len(all_data):
                print(f" {len(all_data)}  {self.num_samples} ")
                # Set random seed
                random.seed(42)
                sampled_indices = random.sample(range(len(all_data)), self.num_samples)
                self.data = [all_data[i] for i in sampled_indices]
            else:
                self.data = all_data[:self.num_samples]  # 
            
            # CPUDevice
            # for i, sample in enumerate(self.data):
            #     self.data[i] = self._move_sample_to_device(sample)
            
            self.num_samples = len(self.data)
            print(f" {self.num_samples} ")
            
        except Exception as e:
            print(f": {e}")
            raise e
    
    def _move_sample_to_device(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Device"""
        device_sample = {}
        for key, value in sample.items():
            if key == 'obs':
                # obs
                device_sample[key] = {}
                for obs_key, obs_value in value.items():
                    if isinstance(obs_value, torch.Tensor):
                        device_sample[key][obs_key] = obs_value.to(self.device)
                    else:
                        device_sample[key][obs_key] = obs_value
            elif isinstance(value, torch.Tensor):
                device_sample[key] = value.to(self.device)
            else:
                device_sample[key] = value
        return device_sample
    
    def _generate_sample(self, sample_idx):
        """"""
        if self.use_target_action:
            # targetactionCPU
            obs, target_action = sample_obs_action_from_dataset(
                self.source_dataset, self.device, self.n_obs_steps, 
                fixed_idx=self.sample_indices[sample_idx] if self.sample_indices is not None else None)
            return {'obs': obs, 'target_action': target_action, 'has_target': torch.tensor(True)}
            
        else:
            # policyactionplaceholder target_actionCPU
            obs = sample_obs_from_dataset(
                self.source_dataset, self.device, self.n_obs_steps,
                fixed_idx=self.sample_indices[sample_idx] if self.sample_indices is not None else None)
            
            # source_datasetNone
            if self.source_dataset is None:
                raise ValueError("source_datasetNone")
                
            # tensoractionCPU
            sample_action = self.source_dataset[0]['action']
            placeholder_action = torch.zeros_like(sample_action)
            return {'obs': obs, 'target_action': placeholder_action, 'has_target': torch.tensor(False)}
            
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_demo_dataset(cfg, device, task_name: str,
                        data_type: str = 'train',
                       base_dir: str = 'pruner_data',
                       num_samples: Optional[int] = None,
                       num_train: Optional[int] = None) -> PrunerTrainingDataset:
    """
    
    
    Args:
        cfg: Config object
        device: Device
        task_name: Task name
        seed: Random seed
        data_type:  ('train'  'valid')
        base_dir: 
        num_samples: None
        num_train: 
        
    Returns:
        PrunerTrainingDataset
    """
    # num_train
    if num_train is not None:
        data_path = Path(base_dir) / task_name /  f"train{num_train}" / f"{data_type}_data.pkl"
    else:
        base_path = Path(base_dir) / task_name 
        
        # train*
        train_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('train')]
        
        if train_dirs:
            # Foundtrain
            train_dir = train_dirs[0]
            data_path = train_dir / f"{data_type}_data.pkl"
            print(f": {data_path}")
        else:
            data_path = base_path / f"{data_type}_data.pkl"
            print(f": {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f": {data_path}")
    
    if num_samples is None:
        try:
            with open(data_path, 'rb') as f:
                temp_data = pickle.load(f)
            num_samples = len(temp_data)
            print(f": {num_samples} ")
        except Exception as e:
            print(f": {e}")
            num_samples = 1000  # 
    
    return PrunerTrainingDataset(
        cfg=cfg,
        device=device,
        use_target_action=True,  # target action
        num_samples=num_samples,
        precomputed_data_path=str(data_path)
    )


def sample_obs_from_dataset(dataset, device, n_obs_steps: int, fixed_idx=None) -> Dict[str, torch.Tensor]:
    """obs"""
    if fixed_idx is not None:
        idx = fixed_idx
    else:
        idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[idx]
    
    # obsn_obs_steps
    obs = sample['obs'][:n_obs_steps]  # shape: (n_obs_steps, obs_dim)
    
    obs_dict = {
        'obs': obs  # CPUDevice
    }
    
    # past_action
    if 'past_action' in sample:
        past_action = sample['past_action'][:n_obs_steps]
        obs_dict['past_action'] = past_action  # CPU
    
    return obs_dict


def sample_obs_action_from_dataset(dataset, device, n_obs_steps: int, fixed_idx=None):
    """obstarget action"""
    if fixed_idx is not None:
        idx = fixed_idx
    else:
        idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[idx]
    
    obs_dict = {}
    # n_obs_steps
    if isinstance(sample['obs'], dict):
        for key,value in sample['obs'].items():
            obs_dict[key] = value[:n_obs_steps]
    else:
        obs = sample['obs'][:n_obs_steps]
        obs_dict = {'obs': obs}  # CPU
    
    action = sample['action']
    target_action = action  # CPU 
    
    return obs_dict, target_action


