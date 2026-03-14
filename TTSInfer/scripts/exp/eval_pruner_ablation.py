"""
 - Cache
3cache24cacherollout_cachecache

:
python TTSInfer/scripts/eval_pruner_ablation.py \
--cache_type 3cache \
--task_name square_ph \
--timestamp 20251106_140229 \
--train_id 0 \
--epoch 17 \
--device cuda:0


python TTSInfer/scripts/eval_pruner_ablation.py \
--cache_type 24cache \
--task_name square_ph \
--timestamp 20251106_140229 \
--train_id 0 \
--epoch 15 \
--device cuda:1

python TTSInfer/scripts/eval_pruner_ablation.py \
--cache_type rollout_cache \
--task_name square_ph \
--timestamp 20251106_140229 \
--train_id 0 \
--epoch 16 \
--device cuda:2

python TTSInfer/scripts/eval_pruner_ablation.py \
--cache_type 3cache \
--task_name can_ph \
--timestamp  \
--train_id 0 \
--epoch 17 \
--device cuda:1
"""

import os
import sys

#  GCC  conda
if 'CONDA_PREFIX' in os.environ:
    system_gcc = '/usr/bin/gcc'
    system_gxx = '/usr/bin/g++'
    if os.path.exists(system_gcc):
        os.environ['CC'] = system_gcc
    if os.path.exists(system_gxx):
        os.environ['CXX'] = system_gxx

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

# TTSInfer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pathlib
import click
import hydra
import torch
import dill
import json
import logging
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from typing import Tuple, Optional, Any, Dict

from TTSInfer.pruner.eval.eval_utils import (
    construct_obs_dict, prepare_env_runner_config, get_actions_per_inference,
    load_workspace, process_runner_log, GateTracker
)
from TTSInfer.pruner.train.train_utils import create_gate_animation
from TTSInfer.pruner.eval.env_runner_setup import instantiate_env_runner
from TTSInfer.acceleration.ablation.pruner_warpper_train_ablation import CachePrunerWrapper
from TTSInfer.pruner.utils import get_task_ckpt_with_train_version


# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_pruner_ablation")
logging.getLogger('absl').setLevel(logging.WARNING)

# Set output buffering
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

@click.command()
@click.option('-c', '--checkpoint', default='auto', help='Policy checkpoint path"auto"task_nametrain_id')
@click.option('-e', '--epoch', required=True, help='Epoch number"final"')
@click.option('-tr', '--train_id', default='0', help='Training ID')
@click.option('-t', '--task_name', required=True, help='Task name')
@click.option('-s', '--seed', default='0', help='Random seed')
@click.option('-d', '--device', default='cuda:0', help='Device')
@click.option('-ti', '--timestamp', required=True, help='Timestamp')
@click.option('-pr', '--pruner_path', default='auto', help='Pruner model path"auto"epoch')
@click.option('--cache_type', type=click.Choice(['3cache', '24cache', 'rollout_cache']), required=True,
              help='Cache: 3cache, 24cache, rollout_cache')
@click.option('--skip_video', is_flag=False, help='Skip video rendering to speed up evaluation')
@click.option('--save_pruning_image', default=True, type=bool, help='Save pruning images')

def main(checkpoint, epoch, train_id, task_name, seed, device, timestamp, pruner_path, 
         cache_type, skip_video, save_pruning_image):
    
    # Set random seed
    seed = int(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    logger.info(f"===  - Cache: {cache_type} ===")
    
    # diffusion policy pretrained ckpt
    if checkpoint == 'auto':
        checkpoint = get_task_ckpt_with_train_version(task_name, int(train_id))
    
    # pruner
    pruner_base_path = os.path.join(
        'ablation_results',
        'single_cache',
        cache_type,
        timestamp,
        str(train_id),
        task_name
    )
    
    if pruner_path == 'auto':
        if epoch == 'final':
            pruner_path = os.path.join(pruner_base_path, 'final_pruner.pth')
            if not os.path.exists(pruner_path):
                raise FileNotFoundError(f"Foundpruner: {pruner_path}")
        else:
            # epoch
            pruner_files = [f for f in os.listdir(pruner_base_path) 
                          if f.startswith(f'pruner_model_{epoch}_') and f.endswith('.pt')]
            if pruner_files:
                best_file = min(pruner_files, key=lambda x: float(x.split('_')[2].replace('.pt', '')))
                pruner_path = os.path.join(pruner_base_path, best_file)
                logger.info(f"Foundpruner: {pruner_path}")
            else:
                raise FileNotFoundError(f"Foundepoch {epoch}pruner: {pruner_base_path}")
    
    output_dir = os.path.join(
        'ablation_results',
        'single_cache',
        cache_type,
        timestamp,
        str(train_id),
        task_name,
        'eval',
        epoch
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f": {output_dir}")
    logger.info(f"checkpoint: {checkpoint}")
    logger.info(f"pruner: {pruner_path}")
    logger.info(f"Cache: {cache_type}")
    
    # workspaceConfig
    workspace, cfg = load_workspace(checkpoint, output_dir)
    
    base_policy = workspace.model
    if getattr(cfg, 'training', None) is not None and getattr(cfg.training, 'use_ema', False):
        ema_model = getattr(workspace, 'ema_model', None)
        if ema_model is not None:
            base_policy = ema_model
    
    torch_device = torch.device(device)
    base_policy = base_policy.to(torch_device)
    base_policy.eval()
    
    # Load pruner model
    logger.info("Load pruner model...")
    pruner_checkpoint = torch.load(pruner_path, map_location=torch_device)
    
    # checkpointprunerConfig
    from TTSInfer.pruner.train.transformer_pruner import TransformerPruner
    from TTSInfer.pruner.train.train_utils import enumerate_decoder_block_keys
    import copy
    
    # layer_names
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]
    
    block_keys = enumerate_decoder_block_keys(layer_names)
    num_steps = getattr(cfg, 'num_inference_steps', 100)
    
    #  obs_dimGet from DP model
    obs_dim = base_policy.model.cond_obs_emb.out_features if hasattr(base_policy.model, 'cond_obs_emb') else 512
    
    # prunercache2
    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_keys,
        hidden_dim=pruner_checkpoint.get('hidden_dim', 256),
        attn_heads=pruner_checkpoint.get('attn_heads', 4),
        dim_feedforward=pruner_checkpoint.get('dim_feedforward', 1024),
        block_encoder_type=pruner_checkpoint.get('block_encoder_type', 'learned'),
        obs_dim=obs_dim,
        head_4=False  # cache2
    ).to(torch_device)
    
    # pruner
    pruner.load_state_dict(pruner_checkpoint['model_state_dict'])
    pruner.eval()
    logger.info("✓ Pruner")
    
    # CachePrunerWrappercache_type
    logger.info(f"CachePrunerWrapper (cache_type={cache_type})...")
    if cache_type == 'rollout_cache':
        CachePrunerWrapper.apply(base_policy, pruner=pruner, cache_type=cache_type, 
                                if_rollout=True, training=False)
    else:
        CachePrunerWrapper.apply(base_policy, pruner=pruner, cache_type=cache_type, 
                                if_rollout=False, training=False)
    
    logger.info("\n===  ===")
    env_runner_cfg = prepare_env_runner_config(cfg, skip_video)
    env_runner = instantiate_env_runner(env_runner_cfg, output_dir)
    
    #  batch_size (n_envs)
    n_envs = len(env_runner.env_fns) if hasattr(env_runner, 'env_fns') else 1
    logger.info(f" n_envs={n_envs} ")
    
    # setupgate
    gate_tracker = None
    if save_pruning_image:
        logger.info("gate...")
        # Configsample
        n_test = getattr(env_runner_cfg, 'n_test', 50)  # 50
        last_sample_idx = n_test - 1
        gate_tracker = GateTracker(sample_indices=[last_sample_idx])
        
        # sample
        def track_last_sample_gate(policy):
            if hasattr(policy, '_cache') and policy._cache.get("gate") is not None:
                gates = policy._cache["gate"]  # [batch, T, B, 2]
                if isinstance(gates, torch.Tensor) and gates.dim() == 4:
                    batch_size = gates.shape[0]
                    # sample
                    if batch_size > 0:
                        last_idx = batch_size - 1
                        sample_gates = gates[last_idx:last_idx+1]  # [1, T, B, 2]
                        
                        # block ()
                        if not gate_tracker.block_names and hasattr(policy, '_cache_block_keys'):
                            gate_tracker.block_names = policy._cache_block_keys
                        
                        # rollout stepgates
                        gate_tracker.gates_sequence.append((gate_tracker.current_rollout_step, sample_gates.clone()))
                        gate_tracker.current_rollout_step += 1
                        
                        logger.debug(f"Tracked gates for rollout step {gate_tracker.current_rollout_step-1}, "
                                   f"last sample (idx={last_idx}), gates shape: {sample_gates.shape}")
        
        # policy
        original_predict_action = base_policy.predict_action
        def tracked_predict_action(obs_dict):
            result = original_predict_action(obs_dict)
            track_last_sample_gate(base_policy)
            return result
        base_policy.predict_action = tracked_predict_action
    
    logger.info("...")
    runner_log = env_runner.run(base_policy)
    
    test_mean_score = runner_log.get('test/mean_score', 0.0)
    
    eval_results = {
        "mean_score": float(test_mean_score),
        "cache_type": cache_type,
        "task_name": task_name,
        "epoch": epoch,
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f" {output_dir}/eval_results.json")
    logger.info(f": {test_mean_score:.4f}")
    
    json_log = process_runner_log(runner_log)
    out_path = os.path.join(output_dir, 'detailed_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f" {out_path}")
    
    # GIF
    if save_pruning_image and gate_tracker is not None and gate_tracker.gates_sequence:
        logger.info("GIF...")
        try:
            # block
            if hasattr(base_policy, '_cache_block_keys'):
                block_names = base_policy._cache_block_keys
            elif gate_tracker.block_names:
                block_names = gate_tracker.block_names
            else:
                # block
                if gate_tracker.gates_sequence:
                    _, first_gates = gate_tracker.gates_sequence[0]
                    if isinstance(first_gates, torch.Tensor) and first_gates.dim() == 4:
                        num_blocks = first_gates.shape[2]
                        block_names = [f"block_{i}" for i in range(num_blocks)]
                    else:
                        block_names = []
                else:
                    block_names = []
            
            if block_names and gate_tracker.gates_sequence:
                # GIF
                gif_path = os.path.join(output_dir, f'gate_evolution_last_sample_{cache_type}.gif')
                
                title_prefix = f"Gate Evolution ({cache_type}) - {task_name.upper()} (Epoch {epoch})"
                
                # cache2
                logger.info(f"cache (cache_type={cache_type}, gate_dim=2)")
                create_gate_animation(
                    gates_sequence=gate_tracker.gates_sequence,
                    block_names=block_names,
                    output_path=gif_path,
                    title_prefix=title_prefix,
                    duration=0.3  # 0.3
                )
                
                logger.info(f"GIF: {gif_path}")
                logger.info(f" {len(gate_tracker.gates_sequence)} rollout")
            else:
                logger.warning("GIFblockgate")
                
        except Exception as e:
            logger.error(f"GIF: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()

