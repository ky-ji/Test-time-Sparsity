from __future__ import annotations

import json
import os
import random
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from diffusion_policy.model.common.normalizer import LinearNormalizer


GatesDict = Dict[int, Dict[str, torch.Tensor]]

class NormalizerManager:
    """
    Normalizercheckpointnormalizer
    
    :
    1. Lowdim: obs
    2. Hybrid: obslowdim + image
    
    :
        # normalizer
        norm_manager = NormalizerManager.from_checkpoint("checkpoint.ckpt")
        
        normalized = norm_manager.normalize(obs_dict, action)
        
        original = norm_manager.unnormalize(normalized['obs'], normalized['action'])
    """
    
    def __init__(self, normalizer=None):
        """
        NormalizerManager
        
        Args:
            normalizer: LinearNormalizer
        """
        self.normalizer = normalizer
        self._is_lowdim = None
        self._available_keys = None
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, verbose: bool = True):
        """
        checkpointnormalizer
        
        Args:
            checkpoint_path: checkpoint
            verbose: 
            
        Returns:
            NormalizerManager: 
        """
        import torch
        import dill
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        
        # checkpoint
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
        
        # normalizer
        normalizer = LinearNormalizer()
        
        # modelstate_dictnormalizer
        model_state_dict = payload['state_dicts']['model']
        normalizer_state_dict = {}
        
        for key, value in model_state_dict.items():
            if key.startswith('normalizer.'):
                # 'normalizer.'
                normalizer_key = key[len('normalizer.'):]
                normalizer_state_dict[normalizer_key] = value
        
        if len(normalizer_state_dict) == 0:
            print("Warning: No normalizer keys found in checkpoint!")
            print(f"Available model keys (first 20): {list(model_state_dict.keys())[:20]}")
            raise ValueError("No normalizer parameters found in checkpoint")
        
        # normalizer
        normalizer.load_state_dict(normalizer_state_dict)
        
        manager = cls(normalizer)
        
        # normalizer
        if verbose:
            manager.print_info()
        
        return manager
    
    @classmethod
    def from_data(cls, data_dict: Dict[str, torch.Tensor], mode='limits', **kwargs):
        """
        normalizer
        
        Args:
            data_dict: 'obs''action'
            mode:  ('limits'  'gaussian')
            **kwargs: fit
            
        Returns:
            NormalizerManager: 
        """
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        
        if 'range_eps' not in kwargs:
            kwargs['range_eps'] = 5e-2  # 
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dict, last_n_dims=1, mode=mode, **kwargs)
        
        return cls(normalizer)
    
    def _update_cache(self):
        """normalizer"""
        if self.normalizer is None:
            raise ValueError("Normalizer not loaded. Use from_checkpoint() or from_data() first.")
        
        input_stats = self.normalizer.get_input_stats()
        self._available_keys = set(input_stats.keys())
        self._is_lowdim = 'obs' in self._available_keys
    
    @property
    def is_lowdim(self) -> bool:
        """lowdim"""
        if self._is_lowdim is None:
            self._update_cache()
        return self._is_lowdim
    
    @property
    def available_keys(self) -> set:
        """normalizer"""
        if self._available_keys is None:
            self._update_cache()
        return self._available_keys
    
    def print_info(self):
        """normalizer"""
        if self.normalizer is None:
            print("No normalizer loaded.")
            return
        
        input_stats = self.normalizer.get_input_stats()
        print(f"✓ Normalizer loaded successfully")
        print(f"  Mode: {'Lowdim' if self.is_lowdim else 'Hybrid'}")
        print(f"  Total features: {len(input_stats)}")
        print(f"  Feature keys: {list(input_stats.keys())}")
        
        for key, stats in input_stats.items():
            if 'min' in stats:
                dim = stats['min'].shape[0]
                min_val = stats['min'].min().item()
                max_val = stats['max'].max().item()
                print(f"    - {key}: dim={dim}, range=[{min_val:.4f}, {max_val:.4f}]")
    
    def normalize(self, obs_dict_or_batch, action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        
        
        :
        1. normalize(obs_dict, action) - obsaction
        2. normalize(batch) - batchaction
        
        batch:
        - Lowdim: {'obs': {'obs': tensor}, 'target_action': tensor, ...}
        - Hybrid: {'obs': {'feature1': tensor, 'feature2': tensor, ...}, 'target_action': tensor, ...}
        
        Args:
            obs_dict_or_batch:   batch
            action: obs_dict
            
        Returns:
            : {'obs': ..., 'action': ...}
        """
        if self.normalizer is None:
            raise ValueError("Normalizer not loaded. Use from_checkpoint() or from_data() first.")
        
        if isinstance(obs_dict_or_batch, dict) and 'obs' in obs_dict_or_batch:
            # 2: batch
            batch = obs_dict_or_batch
            obs_dict = batch['obs']
            
            # action
            action_key = None
            for key in ['action', 'target_action']:
                if key in batch:
                    action_key = key
                    action = batch[key]
                    break
            
            if action_key is None and action is None:
                print("Warning: No action found in batch, only normalizing obs")
        else:
            # 1: obs_dictaction
            obs_dict = obs_dict_or_batch
        
        result = {}
        
        if self.is_lowdim:
            # Lowdim: obs {'obs': {'obs': tensor}}
            if isinstance(obs_dict, dict) and 'obs' in obs_dict:
                # obs
                obs_tensor = obs_dict['obs']
                original_device = obs_tensor.device
                result['obs'] = self.normalizer['obs'].normalize(obs_tensor).to(original_device)
            elif isinstance(obs_dict, torch.Tensor):
                # tensor
                original_device = obs_dict.device
                result['obs'] = self.normalizer['obs'].normalize(obs_dict).to(original_device)
            else:
                raise ValueError(f"Lowdim normalizer expects nested obs dict or tensor, but got: {type(obs_dict)}")
        
        else:
            # Hybrid: obs {'feature1': tensor, 'feature2': tensor, ...}
            if not isinstance(obs_dict, dict):
                raise ValueError(f"Hybrid normalizer expects dict for obs, but got: {type(obs_dict)}")
            
            normalized_obs = {}
            
            for key, value in obs_dict.items():
                if key in self.available_keys:
                    # Device
                    original_device = value.device if hasattr(value, 'device') else None
                    normalized_value = self.normalizer[key].normalize(value)
                    if original_device is not None:
                        normalized_value = normalized_value.to(original_device)
                    normalized_obs[key] = normalized_value
                else:
                    # normalizer
                    print(f"Warning: No normalizer found for '{key}', keeping original values")
                    normalized_obs[key] = value
            
            result['obs'] = normalized_obs
        
        if action is not None:
            if 'action' not in self.available_keys:
                raise ValueError(f"Normalizer does not contain 'action' key. Available: {self.available_keys}")
            original_device = action.device
            result['action'] = self.normalizer['action'].normalize(action).to(original_device)
        
        return result
    
    def unnormalize(self, normalized_obs_dict: Dict[str, torch.Tensor] = None, 
                   normalized_action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        
        
        Args:
            normalized_obs_dict: 
            normalized_action: 
            
        Returns:
            
        """
        if self.normalizer is None:
            raise ValueError("Normalizer not loaded. Use from_checkpoint() or from_data() first.")
        
        result = {}
        
        if normalized_obs_dict is not None:
            if self.is_lowdim:
                # Lowdim
                if isinstance(normalized_obs_dict, dict) and 'obs' in normalized_obs_dict:
                    obs_tensor = normalized_obs_dict['obs']
                    original_device = obs_tensor.device
                    result['obs'] = self.normalizer['obs'].unnormalize(obs_tensor).to(original_device)
                else:
                    original_device = normalized_obs_dict.device
                    result['obs'] = self.normalizer['obs'].unnormalize(normalized_obs_dict).to(original_device)
            else:
                # Hybrid
                unnormalized_obs = {}
                for key, value in normalized_obs_dict.items():
                    if key in self.available_keys:
                        original_device = value.device if hasattr(value, 'device') else None
                        unnormalized_value = self.normalizer[key].unnormalize(value)
                        if original_device is not None:
                            unnormalized_value = unnormalized_value.to(original_device)
                        unnormalized_obs[key] = unnormalized_value
                    else:
                        unnormalized_obs[key] = value
                result['obs'] = unnormalized_obs
        
        if normalized_action is not None:
            if 'action' not in self.available_keys:
                raise ValueError(f"Normalizer does not contain 'action' key. Available: {self.available_keys}")
            original_device = normalized_action.device
            result['action'] = self.normalizer['action'].unnormalize(normalized_action).to(original_device)
        
        return result
    
    def normalize_obs_only(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """"""
        return self.normalize(obs_dict)['obs']
    
    def normalize_action_only(self, action: torch.Tensor) -> torch.Tensor:
        """"""
        return self.normalize({}, action)['action']
    
    def unnormalize_obs_only(self, normalized_obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """"""
        return self.unnormalize(normalized_obs_dict)['obs']
    
    def unnormalize_action_only(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """"""
        return self.unnormalize(None, normalized_action)['action']
    
    def get_feature_info(self, feature_name: str) -> Dict:
        """
        
        
        Args:
            feature_name: 
            
        Returns:
            
        """
        if self.normalizer is None:
            raise ValueError("Normalizer not loaded.")
        
        if feature_name not in self.available_keys:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {self.available_keys}")
        
        input_stats = self.normalizer.get_input_stats()
        return input_stats[feature_name]
    
    def save_normalizer(self, path: str):
        """
        normalizer
        
        Args:
            path: 
        """
        if self.normalizer is None:
            raise ValueError("No normalizer to save.")
        
        import torch
        torch.save(self.normalizer.state_dict(), path)
        print(f"Normalizer saved to {path}")
    
    def load_normalizer(self, path: str):
        """
        normalizer
        
        Args:
            path: 
        """
        import torch
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        
        self.normalizer = LinearNormalizer()
        state_dict = torch.load(path, map_location='cpu')
        self.normalizer.load_state_dict(state_dict)
        
        self._is_lowdim = None
        self._available_keys = None
        
        print(f"Normalizer loaded from {path}")


def save_pruner_ckpt(path: str, state_dict: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def visualize_hard_gates(gates: Union[GatesDict, torch.Tensor], output_path: str, title: str = "Hard Gates Visualization") -> None:
    """
    hard_gateSupports binary (N=2) and ternary (N=3) gates
    
    Args:
        gates: 
               1. {step: {block_key: [gate_values]}}
                  Binary gate: [reuse_prob, compute_prob]
                  Ternary gate: [compute_prob, p1_reuse_prob, p2_reuse_prob]
               2. Tensor[T, B, N] T=steps, B=blocks, N=gate_dim
        output_path: 
        title: 
    """
    if isinstance(gates, torch.Tensor):
        # Tensor: [T, B, N]
        if gates.dim() != 3:
            print(f"gate tensor3 {gates.dim()}")
            return
        
        gate_tensor = gates.detach().cpu().numpy()  # [T, B, N]
        num_steps, num_blocks, gate_dim = gate_tensor.shape
        
        # 0=P1, 1=P2, 2=
        decision_matrix = np.zeros((num_blocks, num_steps))
        
        compute_count = 0
        p1_reuse_count = 0
        p2_reuse_count = 0
        
        for step_idx in range(num_steps):
            for block_idx in range(num_blocks):
                gate_values = gate_tensor[step_idx, block_idx, :]  # [N]
                
                if gate_dim == 2:
                    # Binary gate: [reuse_prob, compute_prob]
                    compute_prob = gate_values[0]
                    if compute_prob > 0.5:
                        decision_matrix[block_idx, step_idx] = 2  # 
                        compute_count += 1
                    else:
                        decision_matrix[block_idx, step_idx] = 0  # P1P1
                        p1_reuse_count += 1
                else:
                    # Ternary gate: [compute_prob, p1_reuse_prob, p2_reuse_prob]
                    max_idx = np.argmax(gate_values[:3])
                    
                    if max_idx == 0:
                        decision_matrix[block_idx, step_idx] = 2  # 
                        compute_count += 1
                    elif max_idx == 1:
                        decision_matrix[block_idx, step_idx] = 0  # P1
                        p1_reuse_count += 1
                    else:
                        decision_matrix[block_idx, step_idx] = 1  # P2
                        p2_reuse_count += 1
        
        # block
        block_labels = [f'Block_{i}' for i in range(num_blocks)]
        steps_list = list(range(num_steps))
        
    elif isinstance(gates, dict):
        if not gates:
            print("No gates to visualize")
            return
        
        steps_list = sorted(gates.keys())
        blocks = list(gates[steps_list[0]].keys())
        
        # blockstep
        num_blocks = len(blocks)
        num_steps = len(steps_list)
        
        # gate
        first_gate = gates[steps_list[0]][blocks[0]]
        gate_dim = len(first_gate)
        
        # 0=P1, 1=P2, 2=
        decision_matrix = np.zeros((num_blocks, num_steps))
        
        compute_count = 0
        p1_reuse_count = 0
        p2_reuse_count = 0
        
        for step_idx, step in enumerate(steps_list):
            for block_idx, block in enumerate(blocks):
                gate_values = gates[step][block]
                
                if gate_dim == 2:
                    # Binary gate: [reuse_prob, compute_prob]
                    compute_prob = gate_values[1].item()
                    if compute_prob > 0.5:
                        decision_matrix[block_idx, step_idx] = 2  # 
                        compute_count += 1
                    else:
                        decision_matrix[block_idx, step_idx] = 0  # P1P1
                        p1_reuse_count += 1
                else:
                    # Ternary gate: [compute_prob, p1_reuse_prob, p2_reuse_prob]
                    probs = [gate_values[i].item() for i in range(3)]
                    max_idx = np.argmax(probs)
                    
                    if max_idx == 0:
                        decision_matrix[block_idx, step_idx] = 2  # 
                        compute_count += 1
                    elif max_idx == 1:
                        decision_matrix[block_idx, step_idx] = 0  # P1
                        p1_reuse_count += 1
                    else:
                        decision_matrix[block_idx, step_idx] = 1  # P2
                        p2_reuse_count += 1
        
        # blocky
        block_labels = []
        for block in blocks:
            if 'sa_block' in block:
                layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
                block_labels.append(f'L{layer_num}_SA')
            elif 'mha_block' in block:
                layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
                block_labels.append(f'L{layer_num}_MHA')
            elif 'ff_block' in block:
                layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
                block_labels.append(f'L{layer_num}_FF')
            else:
                block_labels.append(block[:10])  # 
    else:
        print(f"gates: {type(gates)}")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Ternary gateP2 reuse
    is_binary_gate = (p2_reuse_count == 0)
    
    if is_binary_gate:
        # Binary gate0=(REUSE), 2=(COMPUTE)
        # decision_matrix0=reuse, 2=compute02
        cmap = ListedColormap(['green', 'white', 'blue'])  # white1
        im = plt.imshow(decision_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
    else:
        # Ternary gate0=(P1), 1=(P2), 2=()
        cmap = ListedColormap(['green', 'orange', 'blue'])
        im = plt.imshow(decision_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
    
    total_decisions = num_blocks * num_steps
    compute_ratio = compute_count / total_decisions * 100
    p1_ratio = p1_reuse_count / total_decisions * 100
    p2_ratio = p2_reuse_count / total_decisions * 100
    total_pruning_ratio = (p1_reuse_count + p2_reuse_count) / total_decisions * 100
    
    # setup
    if is_binary_gate:
        title_with_stats = f"{title}\nCompute: {compute_ratio:.1f}% | Reuse: {p1_ratio:.1f}% | Total Pruning: {total_pruning_ratio:.1f}%"
    else:
        title_with_stats = f"{title}\nCompute: {compute_ratio:.1f}% | P1 Reuse: {p1_ratio:.1f}% | P2 Reuse: {p2_ratio:.1f}% | Total Pruning: {total_pruning_ratio:.1f}%"
    
    plt.title(title_with_stats, fontsize=14, fontweight='bold')
    plt.xlabel('Diffusion Steps', fontsize=12)
    plt.ylabel('Transformer Blocks', fontsize=12)
    
    # setup
    plt.xticks(range(0, num_steps, max(1, num_steps//10)), 
               [str(steps_list[i]) for i in range(0, num_steps, max(1, num_steps//10))])
    
    plt.yticks(range(num_blocks), block_labels, fontsize=8)
    
    if is_binary_gate:
        cbar = plt.colorbar(im, ticks=[0, 2])
        cbar.set_ticklabels(['REUSE', 'COMPUTE'])
        cbar.set_label('Decision (Binary)', fontsize=12)
    else:
        cbar = plt.colorbar(im, ticks=[0, 1, 2])
        cbar.set_ticklabels(['P1 REUSE', 'P2 REUSE', 'COMPUTE'])
        cbar.set_label('Decision', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gate: {output_path}")
    print(f": {total_decisions}")
    print(f"COMPUTE: {compute_count} ({compute_ratio:.1f}%)")
    print(f"P1 REUSE: {p1_reuse_count} ({p1_ratio:.1f}%)")
    print(f"P2 REUSE: {p2_reuse_count} ({p2_ratio:.1f}%)")
    print(f": {total_pruning_ratio:.1f}%")
    print(f": {num_blocks} blocks × {num_steps} steps")


def visualize_hard_gates_stage2(gate_tensor: torch.Tensor, block_names: List[str], output_path: str, title: str = "Stage2 Hard Gates Visualization") -> None:
    """
    stage2hard_gate
    
    Args:
        gate_tensor: [batch, T, B, 4] 
                    gate[..., 0] = compute
                    gate[..., 1] = reuse_3cache
                    gate[..., 2] = reuse_24cache
                    gate[..., 3] = reuse_rollout_cache
        block_names: blockName list
        output_path: 
        title: 
    """
    if gate_tensor is None or gate_tensor.numel() == 0:
        print("No gates to visualize")
        return
    
    # batch[T, B, 4]
    gates = gate_tensor  # [T, B, 4]
    
    num_steps, num_blocks, gate_dim = gates.shape
    
    if gate_dim != 4:
        print(f"Warning: Expected 4-way gates, got {gate_dim}-way gates")
        return
    
    # blockstep
    # 0=3cache, 1=24cache, 2=rollout, 3=
    decision_matrix = np.zeros((num_blocks, num_steps))
    
    compute_count = 0
    reuse_3cache_count = 0
    reuse_24cache_count = 0
    reuse_rollout_count = 0
    
    for step_idx in range(num_steps):
        for block_idx in range(num_blocks):
            gate_values = gates[step_idx, block_idx, :]  # [4]
            
            max_idx = torch.argmax(gate_values).item()
            
            if max_idx == 0:
                decision_matrix[block_idx, step_idx] = 3  # 
                compute_count += 1
            elif max_idx == 1:
                decision_matrix[block_idx, step_idx] = 0  # 3cache
                reuse_3cache_count += 1
            elif max_idx == 2:
                decision_matrix[block_idx, step_idx] = 1  # 24cache
                reuse_24cache_count += 1
            else:  # max_idx == 3
                decision_matrix[block_idx, step_idx] = 2  # rollout
                reuse_rollout_count += 1
    
    plt.figure(figsize=(18, 10))
    
    # 0=(3cache), 1=(24cache), 2=(rollout), 3=()
    cmap = ListedColormap(['green', 'orange', 'purple', 'blue'])
    
    im = plt.imshow(decision_matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=3)
    
    total_decisions = num_blocks * num_steps
    compute_ratio = compute_count / total_decisions * 100
    reuse_3_ratio = reuse_3cache_count / total_decisions * 100
    reuse_24_ratio = reuse_24cache_count / total_decisions * 100
    reuse_rollout_ratio = reuse_rollout_count / total_decisions * 100
    total_pruning_ratio = (reuse_3cache_count + reuse_24cache_count + reuse_rollout_count) / total_decisions * 100
    
    # setup
    title_with_stats = (f"{title}\n"
                       f"Compute: {compute_ratio:.1f}% | "
                       f"3Cache: {reuse_3_ratio:.1f}% | "
                       f"24Cache: {reuse_24_ratio:.1f}% | "
                       f"Rollout: {reuse_rollout_ratio:.1f}% | "
                       f"Total Pruning: {total_pruning_ratio:.1f}%")
    plt.title(title_with_stats, fontsize=14, fontweight='bold')
    plt.xlabel('Diffusion Steps', fontsize=12)
    plt.ylabel('Transformer Blocks', fontsize=12)
    
    # setup
    plt.xticks(range(0, num_steps, max(1, num_steps//10)), 
               [str(i) for i in range(0, num_steps, max(1, num_steps//10))])
    
    # blocky
    block_labels = []
    for block in block_names:
        if 'sa_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_SA')
        elif 'mha_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_MHA')
        elif 'ff_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_FF')
        else:
            block_labels.append(block[:10])  # 
    
    plt.yticks(range(num_blocks), block_labels, fontsize=8)
    
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['3Cache', '24Cache', 'Rollout', 'COMPUTE'])
    cbar.set_label('Decision', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stage2 Gate: {output_path}")
    print(f": {total_decisions}")
    print(f"COMPUTE: {compute_count} ({compute_ratio:.1f}%)")
    print(f"3Cache REUSE: {reuse_3cache_count} ({reuse_3_ratio:.1f}%)")
    print(f"24Cache REUSE: {reuse_24cache_count} ({reuse_24_ratio:.1f}%)")
    print(f"Rollout REUSE: {reuse_rollout_count} ({reuse_rollout_ratio:.1f}%)")
    print(f": {total_pruning_ratio:.1f}%")
    print(f": {num_blocks} blocks × {num_steps} steps")


def create_gate_animation_stage2(gates_sequence: List[Tuple[int, torch.Tensor]], 
                                block_names: List[str],
                                output_path: str, 
                                title_prefix: str = "Stage2 Hard Gates Evolution", 
                                duration: float = 0.5) -> None:
    """
    stage2hard gateGIF
    
    Args:
        gates_sequence: [(step_number, gates)] gatestensor [1, T, B, 4]
        block_names: block name list, corresponding to B dimension
        output_path: GIF
        title_prefix: 
        duration: 
    """
    if not gates_sequence:
        print("No gates sequence to animate")
        return
    
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    
    # gates
    _, first_gates = gates_sequence[0]
    _, num_steps, num_blocks, gate_dim = first_gates.shape
    
    if gate_dim != 4:
        print(f"Warning: Expected 4-way gates for stage2, got {gate_dim}-way gates. Using standard animation.")
        return create_gate_animation(gates_sequence, block_names, output_path, title_prefix, duration)
    
    frames_data = []
    frames_stats = []
    step_numbers = []
    
    for step_num, gates in gates_sequence:
        # gates: [1, T, B, 4]
        gates_data = gates[0]  # [T, B, 4]
        
        # blockstep
        # 0=3cache, 1=24cache, 2=rollout, 3=
        decision_matrix = np.zeros((num_blocks, num_steps))
        
        compute_count = 0
        reuse_3cache_count = 0
        reuse_24cache_count = 0
        reuse_rollout_count = 0
        
        for step_idx in range(num_steps):
            for block_idx in range(num_blocks):
                gate_values = gates_data[step_idx, block_idx, :]  # [4]
                max_idx = torch.argmax(gate_values).item()
                
                if max_idx == 0:
                    decision_matrix[block_idx, step_idx] = 3  # 
                    compute_count += 1
                elif max_idx == 1:
                    decision_matrix[block_idx, step_idx] = 0  # 3cache
                    reuse_3cache_count += 1
                elif max_idx == 2:
                    decision_matrix[block_idx, step_idx] = 1  # 24cache
                    reuse_24cache_count += 1
                else:  # max_idx == 3
                    decision_matrix[block_idx, step_idx] = 2  # rollout
                    reuse_rollout_count += 1
        
        frames_data.append(decision_matrix)
        
        total_decisions = num_blocks * num_steps
        compute_ratio = compute_count / total_decisions * 100
        reuse_3_ratio = reuse_3cache_count / total_decisions * 100
        reuse_24_ratio = reuse_24cache_count / total_decisions * 100
        reuse_rollout_ratio = reuse_rollout_count / total_decisions * 100
        total_pruning_ratio = (reuse_3cache_count + reuse_24cache_count + reuse_rollout_count) / total_decisions * 100
        
        frames_stats.append({
            'compute_ratio': compute_ratio,
            'reuse_3_ratio': reuse_3_ratio,
            'reuse_24_ratio': reuse_24_ratio,
            'reuse_rollout_ratio': reuse_rollout_ratio,
            'total_pruning_ratio': total_pruning_ratio
        })
        step_numbers.append(step_num)
    
    # figureaxis
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # 0=(3cache), 1=(24cache), 2=(rollout), 3=()
    cmap = ListedColormap(['green', 'orange', 'purple', 'blue'])
    
    im = ax.imshow(frames_data[0], cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=3)
    
    # setup
    ax.set_xlabel('Diffusion Steps', fontsize=12)
    ax.set_ylabel('Transformer Blocks', fontsize=12)
    
    # setup
    ax.set_xticks(range(0, num_steps, max(1, num_steps//10)))
    ax.set_xticklabels([str(i) for i in range(0, num_steps, max(1, num_steps//10))])
    
    # blocky
    block_labels = []
    for block in block_names:
        if 'sa_block' in block:
            block_labels.append(block.replace('decoder.layers.', 'L').replace('.sa_block', '.SA'))
        elif 'mha_block' in block:
            block_labels.append(block.replace('decoder.layers.', 'L').replace('.mha_block', '.MHA'))
        elif 'ff_block' in block:
            block_labels.append(block.replace('decoder.layers.', 'L').replace('.ff_block', '.FF'))
        else:
            block_labels.append(block.replace('decoder.layers.', 'L'))
    
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels(block_labels, fontsize=8)
    
    # colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['3Cache Reuse', '24Cache Reuse', 'Rollout Reuse', 'Compute'])
    
    stats = frames_stats[0]
    title_text = ax.text(0.5, 1.08, '', transform=ax.transAxes,
                        ha='center', va='top', fontsize=11, weight='bold')
    
    def update(frame):
        """"""
        im.set_array(frames_data[frame])
        stats = frames_stats[frame]
        step_num = step_numbers[frame]
        
        title = (f"{title_prefix} - Rollout Step {step_num}\n"
                f"Compute: {stats['compute_ratio']:.1f}% | "
                f"3Cache: {stats['reuse_3_ratio']:.1f}% | "
                f"24Cache: {stats['reuse_24_ratio']:.1f}% | "
                f"Rollout: {stats['reuse_rollout_ratio']:.1f}% | "
                f"Total Pruning: {stats['total_pruning_ratio']:.1f}%")
        title_text.set_text(title)
        return [im, title_text]
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames_data),
                                  interval=duration*1000, blit=True, repeat=True)
    
    # GIF
    anim.save(output_path, writer='pillow', fps=1/duration)
    plt.close(fig)
    
    print(f"Stage2 animation saved to {output_path}")
    print(f"Contains {len(frames_data)} rollout steps")


def create_gate_animation(gates_sequence: List[Tuple[int, Union[GatesDict, torch.Tensor]]], 
                         block_names: List[str],
                         output_path: str, 
                         title_prefix: str = "Hard Gates Evolution", 
                         duration: float = 0.5) -> None:
    """
    hard gateGIF
    
    Args:
        gates_sequence: [(step_number, gates)] gatestensor [1, T, B, N]
        block_names: block name list, corresponding to B dimension
        output_path: GIF
        title_prefix: 
        duration: 
    """
    if not gates_sequence:
        print("No gates sequence to animate")
        return
    
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    
    # gates
    _, first_gates = gates_sequence[0]
    is_tensor_format = isinstance(first_gates, torch.Tensor)
    
    if is_tensor_format:
        # : [1, T, B, N] where N=2 or 3
        _, num_steps, num_blocks, gate_dim = first_gates.shape
        steps = list(range(num_steps))
        blocks = block_names
    else:
        steps = sorted(first_gates.keys())
        blocks = list(first_gates[steps[0]].keys())
        num_blocks = len(blocks)
        num_steps = len(steps)
        gate_dim = len(first_gates[steps[0]][blocks[0]])
        
    frames_data = []
    frames_stats = []
    step_numbers = []
    
    for step_num, gates in gates_sequence:
        decision_matrix = np.zeros((num_blocks, num_steps))
        
        compute_count = 0
        p1_reuse_count = 0
        p2_reuse_count = 0
        
        if is_tensor_format:
            # : gates  [1, T, B, N]
            gates_data = gates[0]  # [T, B, N]
            
            for step_idx in range(num_steps):
                for block_idx in range(num_blocks):
                    if gate_dim == 2:
                        # Binary gate: [reuse_prob, compute_prob]
                        compute_prob = gates_data[step_idx, block_idx, 1].item()
                        if compute_prob > 0.5:
                            decision_matrix[block_idx, step_idx] = 2  # 
                            compute_count += 1
                        else:
                            decision_matrix[block_idx, step_idx] = 0  # P1
                            p1_reuse_count += 1
                    else:
                        # Ternary gate: [compute_prob, p1_reuse_prob, p2_reuse_prob]
                        probs = gates_data[step_idx, block_idx].cpu().numpy()
                        max_idx = np.argmax(probs)
                        
                        if max_idx == 0:
                            decision_matrix[block_idx, step_idx] = 2  # 
                            compute_count += 1
                        elif max_idx == 1:
                            decision_matrix[block_idx, step_idx] = 0  # P1
                            p1_reuse_count += 1
                        else:
                            decision_matrix[block_idx, step_idx] = 1  # P2
                            p2_reuse_count += 1
        else:
            for step_idx, step in enumerate(steps):
                for block_idx, block in enumerate(blocks):
                    if step in gates and block in gates[step]:
                        gate_values = gates[step][block]
                        
                        if gate_dim == 2:
                            # Binary gate
                            compute_prob = gate_values[1].item()
                            if compute_prob > 0.5:
                                decision_matrix[block_idx, step_idx] = 2  # 
                                compute_count += 1
                            else:
                                decision_matrix[block_idx, step_idx] = 0  # P1
                                p1_reuse_count += 1
                        else:
                            # Ternary gate
                            probs = [gate_values[i].item() for i in range(3)]
                            max_idx = np.argmax(probs)
                            
                            if max_idx == 0:
                                decision_matrix[block_idx, step_idx] = 2  # 
                                compute_count += 1
                            elif max_idx == 1:
                                decision_matrix[block_idx, step_idx] = 0  # P1
                                p1_reuse_count += 1
                            else:
                                decision_matrix[block_idx, step_idx] = 1  # P2
                                p2_reuse_count += 1
        
        frames_data.append(decision_matrix)
        
        total_decisions = num_blocks * num_steps
        compute_ratio = compute_count / total_decisions * 100
        p1_ratio = p1_reuse_count / total_decisions * 100
        p2_ratio = p2_reuse_count / total_decisions * 100
        total_pruning_ratio = (p1_reuse_count + p2_reuse_count) / total_decisions * 100
        
        frames_stats.append({
            'compute_ratio': compute_ratio,
            'p1_ratio': p1_ratio,
            'p2_ratio': p2_ratio,
            'total_pruning_ratio': total_pruning_ratio
        })
        step_numbers.append(step_num)
    
    # figureaxis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 0=(P1), 1=(P2), 2=()
    cmap = ListedColormap(['green', 'orange', 'blue'])
    
    im = ax.imshow(frames_data[0], cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
    
    # setup
    ax.set_xlabel('Diffusion Steps', fontsize=12)
    ax.set_ylabel('Transformer Blocks', fontsize=12)
    
    # setup
    ax.set_xticks(range(0, num_steps, max(1, num_steps//10)))
    ax.set_xticklabels([str(steps[i]) for i in range(0, num_steps, max(1, num_steps//10))])
    
    # blocky
    block_labels = []
    for block in blocks:
        if 'sa_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_SA')
        elif 'mha_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_MHA')
        elif 'ff_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_FF')
        else:
            block_labels.append(block[:10])
    
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels(block_labels, fontsize=8)
    
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.set_ticklabels(['P1 REUSE', 'P2 REUSE', 'COMPUTE'])
    cbar.set_label('Decision', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    initial_stats = frames_stats[0]
    title_text = ax.set_title(f"{title_prefix} - Step: {step_numbers[0]}\n"
                             f"Compute: {initial_stats['compute_ratio']:.1f}% | "
                             f"P1: {initial_stats['p1_ratio']:.1f}% | "
                             f"P2: {initial_stats['p2_ratio']:.1f}% | "
                             f"Total Pruning: {initial_stats['total_pruning_ratio']:.1f}%", 
                             fontsize=14, fontweight='bold')
    
    def animate(frame_idx):
        """"""
        im.set_array(frames_data[frame_idx])
        
        stats = frames_stats[frame_idx]
        title_text.set_text(f"{title_prefix} - Step: {step_numbers[frame_idx]}\n"
                           f"Compute: {stats['compute_ratio']:.1f}% | "
                           f"P1: {stats['p1_ratio']:.1f}% | "
                           f"P2: {stats['p2_ratio']:.1f}% | "
                           f"Total Pruning: {stats['total_pruning_ratio']:.1f}%")
        
        return [im, title_text]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), 
        interval=int(duration * 1000), blit=True, repeat=True
    )
    
    # GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer='pillow', fps=int(1/duration), dpi=150)
    plt.close()
    
    print(f"Gate: {output_path}")
    print(f" {len(frames_data)}  {duration}s")
    print(f": {num_blocks} blocks × {num_steps} steps")
    print(f"Gate: {'' if gate_dim == 3 else ''}")
    
    final_stats = frames_stats[-1]
    print(f" - Compute: {final_stats['compute_ratio']:.1f}%, "
          f"P1: {final_stats['p1_ratio']:.1f}%, "
          f"P2: {final_stats['p2_ratio']:.1f}%, "
          f"Total Pruning: {final_stats['total_pruning_ratio']:.1f}%")


def create_multi_sample_gate_animation(multi_sample_gates: Dict[int, List[Tuple[int, torch.Tensor]]], 
                                     block_names: List[str],
                                     output_path: str, 
                                     title_prefix: str = "Multi-Sample Gate Evolution", 
                                     duration: float = 0.5) -> None:
    """
    2x2hard gateGIF
    
    Args:
        multi_sample_gates: {sample_idx: [(step_number, gates)]} 
        block_names: block name list, corresponding to B dimension
        output_path: GIF
        title_prefix: 
        duration: 
    """
    valid_samples = {k: v for k, v in multi_sample_gates.items() if v}
    
    if not valid_samples:
        print("No valid sample gates sequence to animate")
        return
    
    # 42x2
    sample_indices = sorted(valid_samples.keys())[:4]
    
    if len(sample_indices) < 4:
        print(f"Warning: Only {len(sample_indices)} samples available, padding with empty subplots")
    
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    
    first_sample_idx = sample_indices[0]
    _, first_gates = valid_samples[first_sample_idx][0]
    _, num_steps, num_blocks, gate_dim = first_gates.shape
    
    all_samples_frames = {}
    max_frames = 0
    
    for sample_idx in sample_indices:
        gates_sequence = valid_samples[sample_idx]
        frames_data = []
        step_numbers = []
        
        for step_num, gates in gates_sequence:
            decision_matrix = np.zeros((num_blocks, num_steps))
            
            # : gates  [1, T, B, N] where N=2 or 3
            gates_data = gates[0]  # [T, B, N]
            
            for step_idx in range(num_steps):
                for block_idx in range(num_blocks):
                    if gate_dim == 2:
                        # Binary gate: [reuse_prob, compute_prob]
                        compute_prob = gates_data[step_idx, block_idx, 1].item()
                        if compute_prob > 0.5:
                            decision_matrix[block_idx, step_idx] = 2  # 
                        else:
                            decision_matrix[block_idx, step_idx] = 0  # P1
                    else:
                        # Ternary gate: [compute_prob, p1_reuse_prob, p2_reuse_prob]
                        probs = gates_data[step_idx, block_idx].cpu().numpy()
                        max_idx = np.argmax(probs)
                        
                        if max_idx == 0:
                            decision_matrix[block_idx, step_idx] = 2  # 
                        elif max_idx == 1:
                            decision_matrix[block_idx, step_idx] = 0  # P1
                        else:
                            decision_matrix[block_idx, step_idx] = 1  # P2
            
            frames_data.append(decision_matrix)
            step_numbers.append(step_num)
        
        all_samples_frames[sample_idx] = {
            'frames': frames_data,
            'step_numbers': step_numbers
        }
        max_frames = max(max_frames, len(frames_data))
    
    # 2x2figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(title_prefix, fontsize=20, fontweight='bold')
    
    # 0=(P1), 1=(P2), 2=()
    cmap = ListedColormap(['green', 'orange', 'blue'])
    ims = []
    title_texts = []
    
    # blocky
    block_labels = []
    for block in block_names:
        if 'sa_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_SA')
        elif 'mha_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_MHA')
        elif 'ff_block' in block:
            layer_num = block.split('.')[2] if 'decoder.layers.' in block else 'X'
            block_labels.append(f'L{layer_num}_FF')
        else:
            block_labels.append(block[:10])
    
    # 4
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            subplot_idx = i * 2 + j
            
            if subplot_idx < len(sample_indices):
                sample_idx = sample_indices[subplot_idx]
                initial_frame = all_samples_frames[sample_idx]['frames'][0]
                
                im = ax.imshow(initial_frame, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
                ims.append(im)
                
                # setup
                ax.set_xlabel('Diffusion Steps', fontsize=10)
                ax.set_ylabel('Transformer Blocks', fontsize=10)
                
                # setup
                ax.set_xticks(range(0, num_steps, max(1, num_steps//5)))
                ax.set_xticklabels([str(x) for x in range(0, num_steps, max(1, num_steps//5))], fontsize=8)
                
                ax.set_yticks(range(num_blocks))
                ax.set_yticklabels(block_labels, fontsize=6)
                
                ax.grid(True, alpha=0.3)
                
                initial_step = all_samples_frames[sample_idx]['step_numbers'][0]
                title_text = ax.set_title(f"Sample {sample_idx} - Step: {initial_step}", 
                                        fontsize=12, fontweight='bold')
                title_texts.append(title_text)
                
                if j == 1:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks([0, 1, 2])
                    cbar.set_ticklabels(['P1 REUSE', 'P2 REUSE', 'COMPUTE'])
                    cbar.set_label('Decision', fontsize=10)
            else:
                ax.set_visible(False)
                ims.append(None)
                title_texts.append(None)
    
    def animate(frame_idx):
        """"""
        update_objects = []
        
        for subplot_idx, sample_idx in enumerate(sample_indices):
            if subplot_idx >= 4:
                break
                
            sample_data = all_samples_frames[sample_idx]
            
            actual_frame_idx = frame_idx % len(sample_data['frames'])
            
            current_matrix = sample_data['frames'][actual_frame_idx]
            current_step = sample_data['step_numbers'][actual_frame_idx]
            
            if ims[subplot_idx] is not None:
                ims[subplot_idx].set_array(current_matrix)
                update_objects.append(ims[subplot_idx])
            
            if title_texts[subplot_idx] is not None:
                total_decisions = num_blocks * num_steps
                compute_decisions = np.sum(current_matrix)
                compute_ratio = compute_decisions / total_decisions * 100
                
                title_texts[subplot_idx].set_text(
                    f"Sample {sample_idx} - Step: {current_step} (Compute: {compute_ratio:.1f}%)"
                )
                update_objects.append(title_texts[subplot_idx])
        
        return update_objects
    
    anim = animation.FuncAnimation(
        fig, animate, frames=max_frames, 
        interval=int(duration * 1000), blit=True, repeat=True
    )
    
    # GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer='pillow', fps=int(1/duration), dpi=150)
    plt.close()
    
    print(f"Gate: {output_path}")
    print(f" {max_frames}  {duration}s")
    print(f": {len(sample_indices)}, : {sample_indices}")
    print(f": {num_blocks} blocks × {num_steps} steps")


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def set_seed(seed: int):
    seed = int(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

