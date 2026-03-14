"""

"""
import os
import time
import logging
import hydra
import dill
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
from omegaconf import OmegaConf
from copy import deepcopy
from TTSInfer.pruner.train.dataset import create_demo_dataset
from TTSInfer.pruner.train.train_utils import create_gate_animation, create_multi_sample_gate_animation
from TTSInfer.pruner.train.transformer_pruner import TransformerPruner,enumerate_decoder_block_keys
import copy
import pathlib

# imageio
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# setup
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from plotting_style import apply_paper_style, save_publication_figure
except ImportError:
    def apply_paper_style():
        pass
    def save_publication_figure(fig, path, dpi=300):
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

logger = logging.getLogger(__name__)


def construct_obs_dict(cfg: Any,task_name: str, device: torch.device, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """
    ConfigConstruct observation dict
    
    Args:
        cfg: Config object
        device: Device
        batch_size: 
        
    Returns:
        
    """
    To = getattr(cfg, 'n_obs_steps', 1)
    
    if hasattr(cfg, 'shape_meta'):
        logger.info("Configshape_metaobs_dict")
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            tensor_shape = [batch_size, To] + list(shape['shape'])
            obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        logger.info("Configobs_dimobs_dict")
        obs_dim = getattr(cfg, 'obs_dim', None)
        if obs_dim is None and hasattr(cfg, 'task'):
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            if isinstance(task_cfg, dict) and 'obs_dim' in task_cfg:
                obs_dim = task_cfg['obs_dim']
                logger.info(f"cfg.task.obs_dim: {obs_dim}")
        
        if obs_dim is None:
            raise ValueError("obs_dim")
        
        obs_dict_sample = create_demo_dataset(
        cfg=cfg,
        device=torch.device('cpu'),  # CPU
        task_name=task_name,
        data_type='train',
        base_dir='pruner_data',
        num_samples=batch_size,
       )

        obs = obs_dict_sample[0]['obs']['obs'].unsqueeze(0).to(device)  # Device
        obs_dict = {'obs': obs}

        # obs_dict = {
        #     'obs': torch.zeros((batch_size, To, obs_dim), device=device, dtype=torch.float32)
        # }

        # obs_dict = obs_dict[0]['obs']
        
        # past_action
        if getattr(cfg, 'use_past_action', False):
            action_dim = getattr(cfg, 'action_dim', None)
            if action_dim is None and hasattr(cfg, 'task'):
                task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
                if isinstance(task_cfg, dict):
                    action_dim = task_cfg.get('action_dim', None)
            if action_dim is not None:
                obs_dict['past_action'] = torch.zeros((batch_size, To, action_dim), device=device, dtype=torch.float32)

    return obs_dict

def get_real_obs_dict(cfg: Any, device: torch.device, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """
    
    
    Args:
        cfg: Config object
        device: Device
        batch_size: 
        
    Returns:
        
    """

    from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner
    
    # runner
    temp_runner = KitchenLowdimRunner(
        output_dir="/tmp",
        dataset_dir=cfg.task.dataset.dataset_dir if hasattr(cfg, 'task') else "",
        n_train=1,  # 
        n_test=0,
        max_steps=1,  # 
        n_obs_steps=getattr(cfg, 'n_obs_steps', 1),
        past_action=getattr(cfg, 'use_past_action', False),
        n_envs=1
    )
    
    env = temp_runner.env
    obs = env.reset()  # 
    
    if hasattr(cfg, 'shape_meta'):
        #  -
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            if key in obs:
                tensor_shape = [batch_size] + list(obs[key].shape)
                obs_dict[key] = torch.from_numpy(obs[key]).float().to(device).unsqueeze(0)
            else:
                # key
                tensor_shape = [batch_size, getattr(cfg, 'n_obs_steps', 1)] + list(shape['shape'])
                obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        obs_dict = {
            'obs': torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
        }
        
        # past_action
        if getattr(cfg, 'use_past_action', False):
            action_dim = getattr(cfg, 'action_dim', None)
            if action_dim is None and hasattr(cfg, 'task'):
                task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
                if isinstance(task_cfg, dict):
                    action_dim = task_cfg.get('action_dim', None)
            if action_dim is not None:
                To = getattr(cfg, 'n_obs_steps', 1)
                obs_dict['past_action'] = torch.zeros((batch_size, To, action_dim), device=device, dtype=torch.float32)
    
    logger.info("")
    return obs_dict
        


def get_real_obs_from_dataset(cfg: Any, device: torch.device, batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """
    
    """
    try:
        import zarr
        import numpy as np
        
        dataset_path = cfg.task.dataset.dataset_dir
        
        dataset_root = zarr.open(dataset_path, 'r')
        
        # episode
        episode_idx = np.random.randint(0, len(dataset_root['data']))
        episode_data = dataset_root['data'][episode_idx]
        time_idx = np.random.randint(0, len(episode_data['obs']))
        
        if hasattr(cfg, 'shape_meta'):
            obs_dict = {}
            for key, shape in cfg.shape_meta['obs'].items():
                if key in episode_data:
                    obs_data = episode_data[key][time_idx]
                    obs_dict[key] = torch.from_numpy(obs_data).float().to(device).unsqueeze(0)
                else:
                    tensor_shape = [batch_size, getattr(cfg, 'n_obs_steps', 1)] + list(shape['shape'])
                    obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
        else:
            obs_data = episode_data['obs'][time_idx]
            obs_dict = {
                'obs': torch.from_numpy(obs_data.astype(np.float32)).to(device).unsqueeze(0)
            }
        
        logger.info(f": episode {episode_idx}, time {time_idx}")
        return obs_dict
        
    except Exception as e:
        logger.warning(f": {e}")
        return construct_obs_dict(cfg, device, batch_size)

def safe_estimate_flops(policy_obj: Any, obs_dict: Dict[str, torch.Tensor], cfg: Any, device: torch.device) -> float:
    """
    FLOPswrapped model
    
    Args:
        policy_obj: Policy object
        obs_dict: 
        cfg: Config
        device: Device
        
    Returns:
        FLOPs
    """
    try:
        # wrapped policyFLOPs
        if hasattr(policy_obj, '_cache') and hasattr(policy_obj, '_cacheable_layers'):
            logger.warning("wrapped policyFLOPs")
            return 0.0
        
        # FLOPs
        total_params = sum(p.numel() for p in policy_obj.parameters() if p.requires_grad)
        
        # FLOPs2
        estimated_flops = total_params * 2
        
        logger.info(f"FLOPs - : {total_params/1e6:.2f}M, FLOPs: {estimated_flops/1e9:.2f}G")
        return float(estimated_flops)
        
    except Exception as e:
        logger.warning(f"FLOPs0: {e}")
        return 0.0


def benchmark_policy(policy: Any, obs_dict: Dict[str, torch.Tensor], device: torch.device, 
                    num_trials: int = 10, policy_name: str = "Policy") -> Tuple[float, float]:
    """
    
    
    Args:
        policy: Policy object
        obs_dict: 
        device: Device
        num_trials: 
        policy_name: 
        
    Returns:
        (, )
    """
    logger.info(f"\n----- {policy_name} -----")
    times = [] 
    
    def _cuda_sync_if_needed():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for i in range(num_trials):
        _cuda_sync_if_needed()
        start = time.time()
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)

        _cuda_sync_if_needed()
        dt = time.time() - start
        times.append(dt)
        logger.info(f" {i+1}/{num_trials}: {dt:.4f}s")
    
    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    logger.info(f": {avg_time:.4f}s ± {std_time:.4f}s")
    return avg_time, std_time


def benchmark_policy_with_trajectory(policy: Any, trajectory_dataset, device: torch.device, 
                                     num_trials: int = 10, policy_name: str = "Policy", 
                                     use_rollout_cache: bool = False) -> Tuple[float, float]:
    """
    
    
    
    - 12
    - 1rollout cache
    - 2predict_actionpruned policyuse_rollout_cache=Truecache
    
    Args:
        policy: Policy object
        trajectory_dataset: 
        device: Device
        num_trials: 
        policy_name: 
        use_rollout_cache: rollout cachepruned policy
        
    Returns:
        (, )
    """
    logger.info(f"\n=== {policy_name} () ===")
    times = []
    
    def _cuda_sync_if_needed():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    if len(trajectory_dataset) == 0:
        logger.error("")
        return 0.0, 0.0
    
    # episode2
    first_episode = trajectory_dataset[0]  # Dict: {episode_idx, frames, original_length, unified_length}
    frames = first_episode['frames']
    
    if len(frames) < 2:
        logger.error(f"Episode 0 2: {len(frames)}")
        return 0.0, 0.0
    
    frame0 = frames[0]
    frame1 = frames[1]
    
    # device
    obs0 = {}
    obs1 = {}
    for key, value in frame0['obs'].items():
        if isinstance(value, torch.Tensor):
            obs0[key] = value.unsqueeze(0).to(device)  # batch
        else:
            obs0[key] = value
    
    for key, value in frame1['obs'].items():
        if isinstance(value, torch.Tensor):
            obs1[key] = value.unsqueeze(0).to(device)  # batch
        else:
            obs1[key] = value
    
    logger.info(f"episode 02{len(frames)}")
    
    for i in range(num_trials):
        # Reset cachepruned policy
        if hasattr(policy, '_cache'):
            policy._cache['is_first_predict_action_in_chunk'] = True
            if use_rollout_cache and 'rollout_cache' in policy._cache:
                policy._cache['rollout_cache'].clear()
        
        with torch.no_grad():
            # Frame 0: rollout cache
            _ = policy.predict_action(obs0)
            
            # first frame
            if hasattr(policy, '_cache'):
                policy._cache['is_first_predict_action_in_chunk'] = False
            
            # Frame 1: predict_action
            _cuda_sync_if_needed()
            start = time.time()
            _ = policy.predict_action(obs1)
            _cuda_sync_if_needed()
            dt = time.time() - start
            
        times.append(dt)
        logger.info(f" {i+1}/{num_trials}: {dt:.4f}s")
    
    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    logger.info(f": {avg_time:.4f}s ± {std_time:.4f}s")
    return avg_time, std_time


def warmup_policy(policy: Any, obs_dict: Dict[str, torch.Tensor], warmup_rounds: int = 1) -> bool:
    """
    
    
    Args:
        policy: Policy object
        obs_dict: 
        warmup_rounds: 5
        
    Returns:
        
    """
    try:
        logger.info(f" {warmup_rounds} ...")
        
        with torch.no_grad():
            for i in range(warmup_rounds):
                _ = policy.predict_action(obs_dict)
                logger.debug(f" {i+1}/{warmup_rounds} ")
                
        logger.info("")
        return True
    except Exception as e:
        logger.error(f": {e}")
        import traceback
        traceback.print_exc()
        return False


def prepare_env_runner_config(cfg: Any, skip_video: bool = False) -> Any:
    """
    Prepare env runner config
    
    Args:
        cfg: Config object
        skip_video: 
        
    Returns:
        Env runner config
    """
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    
    if not isinstance(env_runner_cfg, dict):
        raise ValueError("env_runner_cfgMust be a dict")
    
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("Skip video rendering in env runner")
    
    return env_runner_cfg


def get_actions_per_inference(policy: Any, cfg: Any) -> int:
    """
    Actions per inference
    
    Args:
        policy: Policy object
        cfg: Config
        
    Returns:
        Actions per inference
    """
    actions_per_inference = getattr(policy, 'n_action_steps', None)
    if actions_per_inference is None:
        try:
            actions_per_inference = int(getattr(cfg, 'n_action_steps', 1))
        except Exception:
            actions_per_inference = 1
    return actions_per_inference 

    

def load_workspace(checkpoint_path: str, output_dir: str) -> tuple:
    """Load workspace and config"""
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"Loaded config target: {cfg._target_}")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace._output_dir = output_dir
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    return workspace, cfg


def load_pruner_model(pruner_path: str, cfg: Any, device: torch.device, policy=None, reuse_block: bool = False, tgt_sa=None):
    """Load trained pruner model"""
    # Get model parameters
    num_steps = getattr(cfg, 'num_inference_steps', 100)
    
    # Infer layer info from policy config
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]
    
    block_keys = enumerate_decoder_block_keys(layer_names)

    checkpoint = torch.load(pruner_path, map_location='cpu')
    
    #  checkpoint
    hidden_dim = checkpoint['hidden_dim']
    attn_heads = checkpoint['attn_heads']
    dim_feedforward = checkpoint['dim_feedforward']
    block_encoder_type = checkpoint['block_encoder_type']
    # Backward compat: old checkpoints may contain if_dejavu, ignore it
    _ = checkpoint.get('if_dejavu', True)

    #  obs_dimGet from DP model
    obs_dim = policy.model.cond_obs_emb.out_features if hasattr(policy.model, 'cond_obs_emb') else 512

    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_keys,
        hidden_dim=hidden_dim,
        attn_heads=attn_heads,
        dim_feedforward=dim_feedforward,
        block_encoder_type=block_encoder_type,
        obs_dim=obs_dim,
        reuse_block=reuse_block,
        tgt_sa=tgt_sa
    ).to(device)
  
    from TTSInfer.pruner.trajectory.trajectory_utils import expand_pruner_head_for_rollout_cache
    pruner = expand_pruner_head_for_rollout_cache(pruner, init_bias_trajectory=0.0)
    
    if os.path.exists(pruner_path):
        #checkpoint = torch.load(pruner_path, map_location='cpu', weights_only=False)
        #checkpoint = torch.load(pruner_path, map_location=device)
        
        # Filter out temporary attributes added by FLOPs calculator(total_ops, total_params)
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not k.endswith(('total_ops', 'total_params'))}
        
        # Load with strict=False, ignoring mismatched keys
        missing_keys, unexpected_keys = pruner.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading pruner: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading pruner: {unexpected_keys}")
        
        logger.info(f"Loaded pruner model: {pruner_path}")
        logger.info(f"  - epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  - valid_loss: {checkpoint.get('valid_loss', 'unknown')}")
        logger.info(f"  - pruning_ratio: {checkpoint.get('valid_pruning_ratio', 'unknown')}")
    else:
        logger.warning(f"Pruner model file not found: {pruner_path}")
        return None
    
    pruner.to(device)
    pruner.eval()
    return pruner


def create_policies(base_policy: Any, device: torch.device, pruner_path: Optional[str] = None, cfg: Any = None, one_gate: bool = False, reuse_block: bool = False, tgt_sa=None) -> Tuple[Any, Optional[Any]]:
    """Create original and pruned policies"""

    from TTSInfer.acceleration.rollout.pruner_warpper_test_stream import CachePrunerWrapper

    # Original policy
    original_policy = deepcopy(base_policy)
    original_policy.to(device)
    original_policy.eval()

    # Pruned policy
    pruned_policy = deepcopy(base_policy)
    pruned_policy.to(device)
    pruned_policy.eval()
    
    logger.info("Using online pruning strategy (CachePrunerWrapper + Dynamic Pruner)")

    # Load pruner model
    pruner = None
    if pruner_path:
        pruner = load_pruner_model(pruner_path, cfg, device, base_policy, reuse_block, tgt_sa)
    
    if pruner is None:
        logger.warning("Failed to load pruner model")
    else:
        # Apply cache+pruner wrapper with dynamic pruner
        CachePrunerWrapper.apply(
            pruned_policy, 
            pruner=pruner, 
            training=False, 
            one_gate=one_gate,
            if_rollout_cache=True
        )
        
            
    return original_policy, pruned_policy


def process_runner_log(runner_log: dict) -> dict:
    """"""
    json_log = {}
    for key, value in runner_log.items():
        is_video = False
        try:
            from wandb.sdk.data_types.video import Video as WandbVideo
            is_video = isinstance(value, WandbVideo)
        except Exception:
            is_video = False
        
        if is_video:
            try:
                json_log[key] = getattr(value, '_path', None) or getattr(value, 'path', None) or str(value)
            except Exception:
                json_log[key] = str(value)
        else:
            json_log[key] = value
    
    return json_log


class GateTracker:
    """rolloutgate"""
    def __init__(self, sample_indices=[0, 20, 50, 55]):
        self.sample_indices = sample_indices  # 
        self.multi_sample_gates = {idx: [] for idx in sample_indices}  # {sample_idx: [(rollout_step, gates)]}
        self.block_names = []
        self.current_rollout_step = 0
        
        #  -
        self.gates_sequence = []  # 
        
    def reset(self):
        """"""
        self.multi_sample_gates = {idx: [] for idx in self.sample_indices}
        self.gates_sequence = []
        self.current_rollout_step = 0
        
    def track_gate(self, policy, sample_idx=0):
        """gate"""
        if hasattr(policy, '_cache') and policy._cache.get("gate") is not None:
            gates = policy._cache["gate"]  # [batch, T, B, 2]
            if isinstance(gates, torch.Tensor) and gates.dim() == 4:
                # sample (batchsample)
                if sample_idx < gates.shape[0]:
                    sample_gates = gates[sample_idx:sample_idx+1]  # [1, T, B, 2]
                    
                    # block ()
                    if not self.block_names and hasattr(policy, 'model'):
                        self.block_names = self._extract_block_names(policy.model)
                    
                    # rollout stepgates
                    self.gates_sequence.append((self.current_rollout_step, sample_gates.clone()))
                    
                    logger.debug(f"Tracked gates for rollout step {self.current_rollout_step}, "
                               f"sample {sample_idx}, gates shape: {sample_gates.shape}")
                               
    def track_multi_sample_gates(self, policy):
        """gate"""
        if hasattr(policy, '_cache') and policy._cache.get("gate") is not None:
            gates = policy._cache["gate"]  # [batch, T, B, 2]
            if isinstance(gates, torch.Tensor) and gates.dim() == 4:
                batch_size = gates.shape[0]
                
                # block ()
                if not self.block_names and hasattr(policy, 'model'):
                    self.block_names = self._extract_block_names(policy.model)
                
                for sample_idx in self.sample_indices:
                    if sample_idx < batch_size:
                        sample_gates = gates[sample_idx:sample_idx+1]  # [1, T, B, 2]
                        self.multi_sample_gates[sample_idx].append(
                            (self.current_rollout_step, sample_gates.clone())
                        )
                        
                        # 0gates_sequence
                        if sample_idx == 0:
                            self.gates_sequence.append((self.current_rollout_step, sample_gates.clone()))
                        
                self.current_rollout_step += 1
                        
                logger.debug(f"Tracked multi-sample gates for rollout step {self.current_rollout_step-1}, "
                           f"batch_size: {batch_size}, tracked samples: {[idx for idx in self.sample_indices if idx < batch_size]}")
    
    def _extract_block_names(self, model):
        """block"""
        block_names = []
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                block_names.extend([
                    f"decoder.layers.{i}.sa_block",
                    f"decoder.layers.{i}.mha_block", 
                    f"decoder.layers.{i}.ff_block"
                ])
        return block_names
    
    def save_animation(self, output_path: str, title_prefix: str = "Rollout Gate Evolution"):
        """gate"""
        if self.gates_sequence and self.block_names:
            create_gate_animation(
                gates_sequence=self.gates_sequence,
                block_names=self.block_names,
                output_path=output_path,
                title_prefix=title_prefix,
                duration=0.4  # 
            )
            logger.info(f"Gate: {output_path}")
        else:
            logger.warning("gate")
            
    def save_multi_sample_animations(self, output_dir: str, title_prefix: str = "Rollout Gate Evolution"):
        """gate2x2GIF"""
        if not self.block_names:
            logger.warning("block")
            return
            
        valid_samples = {k: v for k, v in self.multi_sample_gates.items() if v}
        
        if not valid_samples:
            logger.warning("gate")
            return
            
        # 2x2
        animation_path = os.path.join(output_dir, 'rollout_gate_evolution_multi_sample.gif')
        
        try:
            create_multi_sample_gate_animation(
                multi_sample_gates=self.multi_sample_gates,
                block_names=self.block_names,
                output_path=animation_path,
                title_prefix=title_prefix,
                duration=0.4
            )
            logger.info(f"Gate: {animation_path}")
            logger.info(f": {sorted(valid_samples.keys())}")
        except Exception as e:
            logger.error(f": {e}")
            logger.info("...")
            self._save_individual_sample_animations(output_dir, title_prefix)
    
    def _save_individual_sample_animations(self, output_dir: str, title_prefix: str = "Rollout Gate Evolution"):
        """"""
        saved_animations = []
        for sample_idx in self.sample_indices:
            if sample_idx in self.multi_sample_gates and self.multi_sample_gates[sample_idx]:
                animation_path = os.path.join(output_dir, f'rollout_gate_evolution_sample_{sample_idx}.gif')
                
                try:
                    create_gate_animation(
                        gates_sequence=self.multi_sample_gates[sample_idx],
                        block_names=self.block_names,
                        output_path=animation_path,
                        title_prefix=f"{title_prefix} - Sample {sample_idx}",
                        duration=0.4
                    )
                    saved_animations.append(animation_path)
                    logger.info(f" {sample_idx} Gate: {animation_path}")
                except Exception as e:
                    logger.error(f" {sample_idx} : {e}")
            else:
                logger.warning(f" {sample_idx} gate")
                
        if saved_animations:
            logger.info(f"Gate {len(saved_animations)} ")
        else:
            logger.warning("")


class SoftGateTracker:
    """rolloutsoft gate"""
    def __init__(self, sample_indices=[0], tanh_factor=6.0, contrast_mode: str = 'tanh', mid_gamma: float = 0.7):
        self.sample_indices = sample_indices  # 
        self.multi_sample_soft_gates = {idx: [] for idx in sample_indices}  # {sample_idx: [(rollout_step, soft_gates)]}
        self.block_names = []
        self.current_rollout_step = 0
        self.tanh_factor = tanh_factor  # tanh
        self.contrast_mode = contrast_mode  # 'tanh'  'mid_gamma'
        # mid_gamma 0<gamma<1  0.5 >1
        self.mid_gamma = float(mid_gamma)
        
    def reset(self):
        """"""
        self.multi_sample_soft_gates = {idx: [] for idx in self.sample_indices}
        self.current_rollout_step = 0
    
    def track_soft_gate(self, policy):
        """soft gate"""
        cache_ctx = getattr(policy, "_cache", None)
        if cache_ctx is None:
            return
            
        # soft gate
        soft_gate = cache_ctx.get("soft_gate", None)
        if soft_gate is not None and len(soft_gate.shape) == 4:  # [batch, T, B, 2]
            batch_size = soft_gate.shape[0]
            
            # block
            if not self.block_names:
                self.block_names = self._extract_block_names(policy.model)
            
            for sample_idx in self.sample_indices:
                if sample_idx < batch_size:
                    # soft gate
                    sample_soft_gate = soft_gate[sample_idx].detach().cpu().numpy()  # [T, B, 2]
                    # rollout stepsoft gate
                    self.multi_sample_soft_gates[sample_idx].append((self.current_rollout_step, sample_soft_gate))
            
        self.current_rollout_step += 1
    
    def _extract_block_names(self, model):
        """block"""
        block_names = []
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                block_names.extend([
                    f"decoder.layers.{i}.sa_block",
                    f"decoder.layers.{i}.mha_block", 
                    f"decoder.layers.{i}.ff_block"
                ])
        return block_names
    
    def _plot_soft_gate_heatmap(self, soft_gates: np.ndarray, out_path: str) -> None:
        """soft gatediffusion stepblock"""
        apply_paper_style()
        
        # soft_gates: [T, B, 2] - compute
        compute_probs = soft_gates[:, :, 1]  # [T, B] - compute
        
        enhanced_probs = self._enhance_contrast(compute_probs)
        
        # diffusion stepblock
        compute_probs_transposed = enhanced_probs.T  # [B, T]
        
        figsize = (12, 9)
        fig = plt.figure(figsize=figsize, dpi=300)
        
        # setupmatplotlib
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.edgecolor'] = 'none'
        
        #sns.light_palette("seagreen", as_cmap=True)
        # vlag
        #cmap = sns.color_palette("light:b", as_cmap=True)
        ax = sns.heatmap(
            compute_probs_transposed,
            vmin=0.0,
            vmax=1.0,
            center=0.5,
            cmap="RdPu",
            square=False,
            cbar=False,
            linewidths=0,
            linecolor='none',
            xticklabels=False,
            yticklabels=False,
            rasterized=True,
        )
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        
        # tick markslabels
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # setupcolorbar
        # if ax.collections and getattr(ax.collections[0], 'colorbar', None) is not None:
        #     cbar = ax.collections[0].colorbar
        #     cbar.ax.tick_params(labelsize=12)
        #     cbar.set_label('')
        
        plt.tight_layout()
        
        # PDF
        save_publication_figure(fig, out_path, dpi=300)
        plt.close(fig)
    
    def save_soft_gate_heatmaps(self, output_dir: str, task_name: str = ""):
        """rollout stepsoft gate"""
        if not any(self.multi_sample_soft_gates.values()):
            logger.warning("soft gate")
            return
        
        # soft_gate
        soft_gate_dir = os.path.join(output_dir, 'soft_gate')
        pathlib.Path(soft_gate_dir).mkdir(parents=True, exist_ok=True)
        
        # PDFGIF
        for sample_idx, soft_gates_sequence in self.multi_sample_soft_gates.items():
            if not soft_gates_sequence:
                continue
                
            logger.info(f"{sample_idx}soft gate{len(soft_gates_sequence)}rollout")
            
            # PDF
            pdf_images = []  # GIF
            
            for rollout_step, soft_gates in soft_gates_sequence:
                # PDF
                pdf_output_path = os.path.join(soft_gate_dir, f'sample_{sample_idx}_step_{rollout_step:03d}.pdf')
                
                try:
                    self._plot_soft_gate_heatmap(soft_gates, pdf_output_path)
                    logger.info(f"{sample_idx} rollout step {rollout_step}soft gate: {pdf_output_path}")
                    
                    # GIFPNG
                    png_output_path = os.path.join(soft_gate_dir, f'temp_sample_{sample_idx}_step_{rollout_step:03d}.png')
                    self._plot_soft_gate_heatmap_png(soft_gates, png_output_path)
                    pdf_images.append(png_output_path)
                    
                except Exception as e:
                    logger.error(f"{sample_idx} rollout step {rollout_step}soft gate: {e}")
            
            # GIF
            if pdf_images:
                gif_output_path = os.path.join(soft_gate_dir, f'sample_{sample_idx}_soft_gate_evolution.gif')
                try:
                    self._create_gif_animation(pdf_images, gif_output_path)
                    logger.info(f"{sample_idx}soft gate GIF: {gif_output_path}")
                    
                    # PNG
                    for png_path in pdf_images:
                        if os.path.exists(png_path):
                            os.remove(png_path)
                            
                except Exception as e:
                    logger.error(f"{sample_idx} GIF: {e}")
        
        logger.info(f"soft gate: {soft_gate_dir}")

    def save_delta_heatmaps(self, output_dir: str, step_start: int = 0, step_first: int = 4, step_gap: int = 5) -> None:
        """diffusion stepscompute

        - diffusion steps: [step_start] + [step_first, step_first+step_gap, ...]
        - steps: step_first-step_start, (step_first+gap)-step_first, ...
        - 0 soft_gate/delta
        """
        if not any(self.multi_sample_soft_gates.values()):
            logger.warning("soft gatedelta")
            return

        #  soft_gat/delta
        delta_dir = os.path.join(output_dir, 'soft_gat', 'delta')
        pathlib.Path(delta_dir).mkdir(parents=True, exist_ok=True)

        for sample_idx, soft_gates_sequence in self.multi_sample_soft_gates.items():
            if not soft_gates_sequence:
                continue

            for rollout_step, soft_gates in soft_gates_sequence:
                # soft_gates: [T, B, 2]
                if soft_gates.ndim != 3 or soft_gates.shape[-1] != 2:
                    continue

                T, B, _ = soft_gates.shape
                compute_probs = soft_gates[:, :, 1]  # [T, B]

                # step
                selected_steps = [step_start]
                t = step_first
                while t < T:
                    selected_steps.append(t)
                    t += step_gap

                if len(selected_steps) < 2:
                    continue

                # : [B, num_pairs]
                deltas = []
                for i in range(1, len(selected_steps)):
                    a = selected_steps[i-1]
                    b = selected_steps[i]
                    #  [B]
                    delta_vec = compute_probs[b, :] - compute_probs[a, :]
                    deltas.append(delta_vec[None, :])  # [1, B]

                #  [B, K]
                if not deltas:
                    continue
                import numpy as _np
                # numpynp
                delta_matrix = _np.stack([d.squeeze(0).cpu().numpy() for d in deltas], axis=1)  # [B, K]

                out_path = os.path.join(delta_dir, f'sample_{sample_idx}_rollout_{rollout_step:03d}_delta.pdf')
                try:
                    self._plot_delta_heatmap(delta_matrix, out_path)
                    logger.info(f"{sample_idx} rollout {rollout_step} delta: {out_path}")
                except Exception as e:
                    logger.error(f"delta: {e}")

    def _plot_delta_heatmap(self, delta_matrix: np.ndarray, out_path: str) -> None:
        """delta=block=diffusion step0"""
        apply_paper_style()
        figsize = (8, 6)
        fig = plt.figure(figsize=figsize, dpi=300)

        ax = sns.heatmap(
            delta_matrix,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="vlag",
            square=False,
            cbar=True,
            cbar_kws={'shrink': 0.75, 'aspect': 30},
            linewidths=0,
            linecolor='none',
            xticklabels=False,
            yticklabels=False,
            rasterized=True,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax.collections and getattr(ax.collections[0], 'colorbar', None) is not None:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('')

        plt.tight_layout()
        save_publication_figure(fig, out_path, dpi=300)
        plt.close(fig)
    
    def _plot_soft_gate_heatmap_png(self, soft_gates: np.ndarray, out_path: str) -> None:
        """PNGGIF"""
        apply_paper_style()
        
        # soft_gates: [T, B, 2] - compute
        compute_probs = soft_gates[:, :, 1]  # [T, B] - compute
        
        enhanced_probs = self._enhance_contrast(compute_probs)
        
        # diffusion stepblock
        compute_probs_transposed = enhanced_probs.T  # [B, T]
        
        figsize = (12, 8)
        fig = plt.figure(figsize=figsize, dpi=100)
        
        # vlag
        ax = sns.heatmap(
            compute_probs_transposed,
            vmin=0.0,
            vmax=1.0,
            center=0.5,
            cmap="vlag",
            square=False,
            cbar=True,
            cbar_kws={'shrink': 0.75, 'aspect': 30},
            linewidths=0,
            linecolor='none',
            xticklabels=False,
            yticklabels=False,
            rasterized=True,
        )
        
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # setupcolorbar
        if ax.collections and getattr(ax.collections[0], 'colorbar', None) is not None:
            cbar = ax.collections[0].colorbar        
            cbar.set_label('')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)

    def _enhance_contrast(self, probs: np.ndarray) -> np.ndarray:
        """
        - tanh: 01
        - mid_gamma: 0.5
        """
        if self.contrast_mode == 'tanh':
            x = np.tanh((probs - 0.5) * self.tanh_factor)  # [-1,1]
            return (x + 1.0) / 2.0
        else:
            # mid_gamma 0.5gamma
            # d = |p-0.5| in [0,0.5]; [0,1]d' = d/0.5; gammad'' = (d')**gamma
            # [0,1]p' = 0.5 + sign(p-0.5) * d'' * 0.5
            #  0<gamma<1 0.5>1
            delta = probs - 0.5
            sign = np.sign(delta)
            mag = np.abs(delta) / 0.5  # [0,1]
            mag = np.clip(mag, 0.0, 1.0)
            mag_enhanced = np.power(mag, self.mid_gamma)
            result = 0.5 + sign * mag_enhanced * 0.5
            return np.clip(result, 0.0, 1.0)
    
    def _create_gif_animation(self, image_paths: list, output_path: str, duration: float = 0.5) -> None:
        """PNGGIF"""
        if imageio is None:
            try:
                import imageio.v2 as imageio_local
            except ImportError:
                logger.warning("imageioGIF")
                return
        else:
            imageio_local = imageio
        
        images = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                images.append(imageio_local.imread(image_path))
        
        if images:
            imageio_local.mimsave(output_path, images, duration=duration, loop=0)
        else:
            logger.warning(f"FoundGIF: {output_path}")




def export_gate_data(gate_tracker: Optional[GateTracker], soft_gate_tracker: Optional[SoftGateTracker], output_dir: str) -> None:
    """hard/soft gate//
    - hard gate: GateTracker.multi_sample_gates [R, T, B, 2]
    - soft gate: SoftGateTracker.multi_sample_soft_gates [R, T, B, 2]
    """
    try:
        base_dir = os.path.join(output_dir, 'gate_raw')
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

        if gate_tracker is not None and any(gate_tracker.multi_sample_gates.values()):
            hard_dir = os.path.join(base_dir, 'hard')
            pathlib.Path(hard_dir).mkdir(parents=True, exist_ok=True)
            meta_path = os.path.join(hard_dir, 'meta.json')

            hard_meta = {
                'samples': sorted([idx for idx, seq in gate_tracker.multi_sample_gates.items() if seq]),
                'block_names': gate_tracker.block_names,
            }

            with open(meta_path, 'w') as f:
                json.dump(hard_meta, f, indent=2)

            for sample_idx, seq in gate_tracker.multi_sample_gates.items():
                if not seq:
                    continue
                steps = [int(s) for s, _ in seq]
                #  [R, T, B, 2]
                array_np = np.stack([t.squeeze(0).detach().cpu().numpy() for _, t in seq], axis=0)
                out_path = os.path.join(hard_dir, f'sample_{sample_idx}.npz')
                np.savez_compressed(out_path, steps=np.array(steps, dtype=np.int32), gates=array_np)

        if soft_gate_tracker is not None and any(soft_gate_tracker.multi_sample_soft_gates.values()):
            soft_dir = os.path.join(base_dir, 'soft')
            pathlib.Path(soft_dir).mkdir(parents=True, exist_ok=True)
            meta_path = os.path.join(soft_dir, 'meta.json')

            soft_meta = {
                'samples': sorted([idx for idx, seq in soft_gate_tracker.multi_sample_soft_gates.items() if seq]),
                'block_names': soft_gate_tracker.block_names,
            }

            with open(meta_path, 'w') as f:
                json.dump(soft_meta, f, indent=2)

            for sample_idx, seq in soft_gate_tracker.multi_sample_soft_gates.items():
                if not seq:
                    continue
                steps = [int(s) for s, _ in seq]
                array_np = np.stack([t for _, t in seq], axis=0)  # [R, T, B, 2]
                out_path = os.path.join(soft_dir, f'sample_{sample_idx}.npz')
                np.savez_compressed(out_path, steps=np.array(steps, dtype=np.int32), gates=array_np)
    except Exception as e:
        logger.error(f"gate: {e}")

