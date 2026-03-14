"""Speed-only evaluation utilities for the released pruner pipeline."""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# Add TTSInfer parent dir to path for diffusion_policy import
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

from TTSInfer.acceleration.rollout.pruner_warpper_test_stream import CachePrunerWrapper as CachePrunerWrapperTest
from pruner.eval.eval_utils import (
    construct_obs_dict, safe_estimate_flops, benchmark_policy, warmup_policy,
    get_actions_per_inference, load_workspace, 
    create_policies, create_policies_offline, test_pruner_time
)
from TTSInfer.pruner.utils import get_task_ckpt_with_train_version

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_speed_only")
logging.getLogger('absl').setLevel(logging.WARNING)

# Set output buffering
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

@click.command()
@click.option('-c', '--checkpoint', default='auto', help='Policy checkpoint path"auto"task_nametrain_id')
@click.option('-e', '--epoch', required=True, help='Epoch number')
@click.option('-tr', '--train_id', default='train0', help='Training ID ( train0, train1)')
@click.option('--train_root', default=None, help=' exp_output/<train_ts>/train/<train_id>/<task_name>')
@click.option('-o', '--output_dir', default='speed_eval_results', help='Base output directory')
@click.option('--summary_file', default=None, help='')
@click.option('-t', '--task_name', required=True, help='Task name')
@click.option('-s', '--seed', default='0', help='Random seed')
@click.option('-d', '--device', default='cuda:0', help='Device')
@click.option('-if', '--if_24cache', default=False, type=bool, help='24cache')
@click.option('-ti', '--timestamp', required=True, help='Timestamp')
@click.option('-pr', '--pruner_path', default='auto', help='Pruner model path"auto"epoch')
@click.option('--mode', type=click.Choice(['origin', 'online', 'offline']), default='online',
              help='Evaluation mode: origin (Original policy), online (Online pruning), offline (Offline pruning)')
@click.option('--num_trials', default=10, type=int, help='Number of benchmark trials')
@click.option('--test_pruner', default=False, type=bool, help='pruner')

def main(checkpoint, epoch, train_id, train_root, output_dir, task_name, seed, device, mode, num_trials, timestamp, pruner_path, if_24cache, summary_file, test_pruner):
    """"""
    
    # Set random seed
    seed = int(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    
    logger.info(f": {task_name}, : {mode}")
    
    # checkpointtask_nametrain_id
    if not checkpoint or checkpoint == 'auto':
        # train_id ( "train0" -> 0)
        if train_id.startswith('train'):
            train_version = int(train_id.replace('train', ''))
        else:
            train_version = int(train_id)
        
        checkpoint = get_task_ckpt_with_train_version(task_name, train_version)
        logger.info(f"checkpoint: {checkpoint}")
    
    # setup
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if summary_file is None:
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
        
    # pruner
    if pruner_path == 'auto' or (not os.path.isabs(pruner_path) and not pruner_path.startswith('exp_output')):
        if train_root is not None and len(str(train_root).strip()) > 0:
            pruner_base_dir = train_root
        else:
            pruner_base_dir = os.path.join('final_result', 'main_result', 'train', train_id, task_name)
        
        if pruner_path == 'auto':
            # epochpruner
            if os.path.exists(pruner_base_dir):
                pruner_files = [f for f in os.listdir(pruner_base_dir) if f.startswith(f'pruner_model_{epoch}_') and f.endswith('.pt')]
                if pruner_files:
                    best_file = min(pruner_files, key=lambda x: float(x.split('_')[2].replace('.pt', '')))
                    pruner_path = os.path.join(pruner_base_dir, best_file)
                    logger.info(f"Foundpruner: {pruner_path}")
                else:
                    if mode == 'online':
                        raise FileNotFoundError(f"Foundepoch {epoch}pruner: {pruner_base_dir}")
                    else:
                        logger.warning(f"FoundprunerOriginal policy")
                        mode = 'origin'
                        pruner_path = None
            else:
                if mode == 'online':
                    raise FileNotFoundError(f": {pruner_base_dir}")
                else:
                    logger.warning(f"Original policy")
                    mode = 'origin'
                    pruner_path = None
        else:
            pruner_path = os.path.join(pruner_base_dir, pruner_path)
    
    logger.info(f": {output_dir}")
    logger.info(f": {summary_file}")
    logger.info(f"checkpoint: {checkpoint}")
    if pruner_path:
        logger.info(f"pruner: {pruner_path}")

    # Load workspace and config
    workspace, cfg = load_workspace(checkpoint, output_dir)
    
    base_policy = workspace.model
    if getattr(cfg, 'training', None) is not None and getattr(cfg.training, 'use_ema', False):
        ema_model = getattr(workspace, 'ema_model', None)
        if ema_model is not None:
            base_policy = ema_model

    torch_device = torch.device(device)
    
    if mode == 'online' and pruner_path:
        original_policy, pruned_policy = create_policies(base_policy, torch_device, pruner_path, cfg, if_24cache)
    else:
        original_policy, pruned_policy = create_policies(base_policy, torch_device, None, cfg)
    
    actions_per_inference = get_actions_per_inference(original_policy, cfg)
    logger.info(f"Actions per inference: {actions_per_inference}")

    # Construct observation dict
    obs_dict = construct_obs_dict(cfg, task_name, torch_device, batch_size=1)
    logger.info(f": {list(obs_dict.keys())}")

    logger.info("...")
    warmup_policy(original_policy, obs_dict)
    if pruned_policy is not None:
        warmup_policy(pruned_policy, obs_dict)
    logger.info("")

    logger.info(f"\n===  (: {actions_per_inference}) ===")
    
    # Original policy
    avg_original, std_original = benchmark_policy(
        original_policy, obs_dict, torch_device, num_trials, "Original policy"
    )
    freq_original = (actions_per_inference / avg_original) if avg_original > 0 else 0.0
    
    # Original policyFLOPs
    logger.info("Original policyFLOPs...")
    flops_original = safe_estimate_flops(original_policy, obs_dict, cfg, torch_device)
    
    # Pruned policy
    avg_pruned, std_pruned = 0.0, 0.0
    freq_pruned = 0.0
    speedup = 0.0
    flops_pruned = 0.0
    
    # Pruner
    avg_pruner_time, std_pruner_time = 0.0, 0.0
    pruner_time_ratio = 0.0
    
    if pruned_policy is not None:
        avg_pruned, std_pruned = benchmark_policy(
            pruned_policy, obs_dict, torch_device, num_trials, "Pruned policy"
        )
        freq_pruned = (actions_per_inference / avg_pruned) if avg_pruned > 0 else 0.0
        speedup = (avg_original / avg_pruned) if (avg_original > 0 and avg_pruned > 0) else 0.0
        logger.info(f": {speedup:.2f}x")
        
        # FLOPs
        logger.info("Pruned policyFLOPs...")
        flops_pruned = safe_estimate_flops(pruned_policy, obs_dict, cfg, torch_device)
        
        # Pruner
        if test_pruner and hasattr(pruned_policy, '_cache') and 'pruner' in pruned_policy._cache:
            logger.info(f"\n=== Pruner  ===")
            avg_pruner_time, std_pruner_time = test_pruner_time(
                pruned_policy, obs_dict, torch_device, num_trials, "Pruner", if_24cache
            )
            
            #  pruner
            if avg_pruned > 0:
                pruner_time_ratio = avg_pruner_time / avg_pruned
                logger.info(f"Pruner : {pruner_time_ratio * 100:.2f}%")
            else:
                logger.warning(" pruner Pruned policy0")

    result_item = {
        "task_name": task_name,
        "train_id": train_id,
        "epoch": int(epoch),
        "speedup": speedup,
        "original_time": avg_original,
        "pruned_time": avg_pruned,
        "flops_reduction": float((flops_original - flops_pruned) / flops_original) if flops_original > 0 else 0.0,
        "original_flops": float(flops_original),
        "pruned_flops": float(flops_pruned) if pruned_policy is not None else 0.0,
        "mode": mode
    }
    
    #  pruner
    if test_pruner:
        result_item.update({
            "pruner_time": avg_pruner_time,
            "pruner_time_std": std_pruner_time,
            "pruner_time_ratio": pruner_time_ratio,
            "test_pruner_enabled": True
        })
    else:
        result_item.update({
            "pruner_time": 0.0,
            "pruner_time_std": 0.0,
            "pruner_time_ratio": 0.0,
            "test_pruner_enabled": False
        })

    summary_data = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
        except:
            summary_data = []
    
    summary_data.append(result_item)
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f": {summary_file}")
    
    print(f"\n{'='*50}")
    print(f": {task_name}")
    print(f"Epoch: {epoch}")
    print(f"Original policy: {avg_original:.4f}s")
    if pruned_policy is not None:
        print(f"Pruned policy: {avg_pruned:.4f}s")
        print(f": {speedup:.2f}x")
        print(f"FLOPs: {((flops_original - flops_pruned) / flops_original * 100):.1f}%")
        
        #  pruner
        if test_pruner and avg_pruner_time > 0:
            print(f"Pruner : {avg_pruner_time:.4f}s ± {std_pruner_time:.4f}s")
            print(f"Pruner : {pruner_time_ratio * 100:.2f}%")
    print(f"{'='*50}")
    
    logger.info(f"Task {task_name} complete!")
    return result_item

if __name__ == '__main__':
    main() 