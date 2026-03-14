"""Simulation evaluation for trained pruners."""


import os
import sys

# Use system GCC instead of conda's old version to avoid glibc compatibility issues
# Conda  GCC 7.5.0  .relr.dyn section
if 'CONDA_PREFIX' in os.environ:
    # Force system gcc/g++ instead of conda's old version
    system_gcc = '/usr/bin/gcc'
    system_gxx = '/usr/bin/g++'
    if os.path.exists(system_gcc):
        os.environ['CC'] = system_gcc
    if os.path.exists(system_gxx):
        os.environ['CXX'] = system_gxx

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import logging
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from typing import Tuple, Optional, Any, Dict
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from TTSInfer.pruner.eval.eval_utils import (
    construct_obs_dict,get_real_obs_dict, get_real_obs_from_dataset, safe_estimate_flops, 
    benchmark_policy, benchmark_policy_with_trajectory, warmup_policy,
    prepare_env_runner_config, get_actions_per_inference,load_workspace, 
    create_policies, process_runner_log,GateTracker, SoftGateTracker, export_gate_data
)
from TTSInfer.pruner.train.train_utils import create_gate_animation, create_gate_animation_trajectory
from TTSInfer.pruner.eval.env_runner_setup import instantiate_env_runner
from TTSInfer.pruner.eval.reuse_module import reuse_block as reuse_block_module

def setup_ddim_scheduler(policy, ddim_steps: int):
    """
    policyschedulerDDIMsetup
    
    Args:
        policy: 
        ddim_steps: DDIM
    """
    original_scheduler = policy.noise_scheduler
    
    logger.info(f"scheduler: {type(original_scheduler).__name__}")
    logger.info(f": {original_scheduler.config.num_train_timesteps}")
    logger.info(f": {policy.num_inference_steps}")
    
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
    
    logger.info(f"✓ DDIM schedulersetup: {ddim_steps}")

from TTSInfer.pruner.utils import get_task_ckpt_with_train_version


# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_pruner")
logging.getLogger('absl').setLevel(logging.WARNING)

# Set output buffering
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

@click.command()
@click.option('-c', '--checkpoint', default='auto', help='Policy checkpoint path"auto"task_nametrain_id')
@click.option('-e', '--epoch', required=True, help='Epoch number')
@click.option('-tr', '--train_id', default='train0', help='Training ID ( train1, train2)')
@click.option('-o', '--output_dir', default='sim_result/pruner_ckpt', help='Root directory containing trained pruner checkpoints')
@click.option('-t', '--task_name', required=True, help='Task name')
@click.option('-s', '--seed', default='0', help='Random seed')
@click.option('-d', '--device', default='cuda:1', help='Device')
@click.option('--one_gate', default=False, type=bool, help='Use observation dict to pre-generate fixed hard gates for all samples and rollout steps')
@click.option('-ti', '--timestamp', required=True, help='Timestamp')
@click.option('-pr', '--pruner_path', default='auto', help='Pruner model path"auto"epoch')
@click.option('--mode', type=click.Choice(['origin', 'online', 'offline']), default='online',
              help='Evaluation mode: origin (Original policy), pruned_online (Online pruning), pruned_offline (Offline pruning)')
# @click.option('--parallel', default=False, type=bool, help='prunerstep')
@click.option('--skip_video', is_flag=False, help='Skip video rendering to speed up evaluation')
@click.option('--num_trials', default=10, type=int, help='Number of benchmark trials')
@click.option('--save_pruning_image', default=True, type=bool, help='Save pruning images')
@click.option('--reuse_block', default=False, type=bool, help=' block embedding  self-attention')
@click.option('--rollout_cache', default=False, type=bool, help='')
@click.option('--ddim', default=None, type=int, help='DDIM scheduler--ddim 4040DDIM')
@click.option('--save_gate', default=False, type=bool, help='hard gatesoft gate')


def main(checkpoint, epoch, train_id, output_dir, task_name, seed, device, mode, skip_video, num_trials, timestamp, pruner_path, save_pruning_image,  one_gate, reuse_block, rollout_cache, ddim, save_gate):

     # Set random seed
    seed = int(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # diffusion policy prtrained ckpt
    if checkpoint == 'auto':
        checkpoint = get_task_ckpt_with_train_version(task_name, train_id)
    else:
        checkpoint = checkpoint

    if rollout_cache:
        pruner_base_path = os.path.join(output_dir, timestamp, train_id, task_name)

    elif output_dir == 'exp_output':
        pruner_base_path = os.path.join(output_dir,timestamp,'train',f'train{train_id}',task_name)
    elif output_dir == 'output' :
        pruner_base_path = os.path.join(output_dir,task_name,timestamp,'train')
    else:
        # output_dirtrain_eval_integrated_multi.py
        if os.path.exists(output_dir) and any(f.startswith('pruner_model_') for f in os.listdir(output_dir)):
            pruner_base_path = output_dir
        else:
            pruner_base_path = os.path.join('exp_output',timestamp,'train',f'train{train_id}',task_name)

    if pruner_path == 'auto':
        pruner_files = [f for f in os.listdir(pruner_base_path) if f.startswith(f'pruner_model_{epoch}_') and f.endswith('.pt')]
        if pruner_files:
            best_file = min(pruner_files, key=lambda x: float(x.split('_')[2].replace('.pt', '')))
            pruner_path = os.path.join(pruner_base_path, best_file)
            logger.info(f"Foundpruner: {pruner_path}")
        else:
            raise FileNotFoundError(f"Foundepoch {epoch}pruner: {pruner_base_path}")

    elif pruner_path.startswith('pruner_model_') and pruner_path.count('_') == 2:
        #  "pruner_model_{epoch}_{score}.pt"
        pruner_path = os.path.join(pruner_base_path, pruner_path)

    # eval
    # output_dir
    original_output_dir = output_dir
    if rollout_cache:
        output_dir = os.path.join('sim_result', 'pruner_eval', timestamp, train_id, task_name, epoch)
    elif output_dir == 'exp_output':
        output_dir = os.path.join(output_dir,timestamp,'eval',train_id,task_name,epoch)
    elif output_dir == 'output' :
        output_dir = os.path.join(output_dir,task_name,timestamp,'eval',epoch)
    else:
        # eval
        if 'train' in output_dir:
            # traineval
            eval_output_dir = output_dir.replace('/train/', '/eval/')
            output_dir = os.path.join(eval_output_dir, epoch)
        else:
            output_dir = os.path.join('exp_output',timestamp,'eval',train_id,task_name,epoch)
   
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f": {output_dir}")
    logger.info(f"checkpoint: {checkpoint}")
    logger.info(f"pruner: {pruner_path}")

    workspace, cfg = load_workspace(checkpoint, output_dir)
    
    base_policy = workspace.model
    if getattr(cfg, 'training', None) is not None and getattr(cfg.training, 'use_ema', False):
        ema_model = getattr(workspace, 'ema_model', None)
        if ema_model is not None:
            base_policy = ema_model

    # DDIMschedulersetup
    if ddim is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"DDIM scheduler: {ddim}")
        logger.info(f"{'='*60}")
        setup_ddim_scheduler(base_policy, ddim)
    else:
        logger.info(f"\nscheduler: {base_policy.num_inference_steps}")

    torch_device = torch.device(device)
    
    # Construct observation dict(Needs to be generated before create_policies, For one_gate mode)
    obs_dict = construct_obs_dict(cfg, task_name,torch_device, batch_size=1)
    
    #  reuse_block tgt_sa
    tgt_sa = None
    if reuse_block:
        tgt_sa = reuse_block_module(pruner_path, cfg, torch_device)
        logger.info(f"[reuse_block] tgt_sa  SA block: {tgt_sa.shape}")
    
    original_policy, pruned_policy = create_policies(base_policy, torch_device, pruner_path, cfg,  one_gate, reuse_block, tgt_sa,rollout_cache)

    actions_per_inference = get_actions_per_inference(original_policy, cfg)
    logger.info(f"Actions per inference: {actions_per_inference}")
    logger.info(f": {list(obs_dict.keys())}")
    for k, v in obs_dict.items():
        logger.info(f"  {k}: ={tuple(v.shape)}")

    logger.info("...")
    warmup_policy(original_policy, obs_dict)
    if pruned_policy is not None:
        warmup_policy(pruned_policy, obs_dict)
    logger.info("")

    logger.info(f"\n===  (: {actions_per_inference}) ===")
    

    # Original policy
    if rollout_cache:
        from TTSInfer.pruner.trajectory.trajectory_dataset import TrajectoryDataset
        trajectory_data_dir = f"PrunerData/pruner_tra_data_max/trajectories/{task_name}"

        # 1episode
        trajectory_dataset = TrajectoryDataset(
            data_dir=trajectory_data_dir,
            device='cpu',
            episode_indices=[0]  # episode
        )
    
        avg_original, std_original = benchmark_policy_with_trajectory(
            original_policy, trajectory_dataset, torch_device, num_trials, 
            "Original policy", use_rollout_cache=False
        )
    else:
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
    
    if pruned_policy is not None:
        if rollout_cache:
            # Pruned policyrollout cache
            avg_pruned, std_pruned = benchmark_policy_with_trajectory(
                pruned_policy, trajectory_dataset, torch_device, num_trials, 
                "Pruned policy", use_rollout_cache=True
            )
        else:
            avg_pruned, std_pruned = benchmark_policy(
                pruned_policy, obs_dict, torch_device, num_trials, "Pruned policy"
            )
        
        #  pruner
        # avg_pruner, std_pruner = test_pruner_time( pruned_policy, obs_dict, torch_device, num_trials, "pruner")
        freq_pruned = (actions_per_inference / avg_pruned) if avg_pruned > 0 else 0.0
        speedup = (avg_original / avg_pruned) if (avg_original > 0 and avg_pruned > 0) else 0.0
        logger.info(f": {speedup:.2f}x")
        
        # FLOPs
        logger.info("Pruned policyFLOPs...")
        flops_pruned = safe_estimate_flops(pruned_policy, obs_dict, cfg, torch_device)

    benchmark_results = {
        "device": str(torch_device),
        "mode": mode,
        "actions_per_inference": int(actions_per_inference),
        "pruner_model_path": pruner_path if mode == 'pruned_online' else None,
        "ddim_steps": ddim if ddim is not None else None,
        "scheduler_type": type(original_policy.noise_scheduler).__name__,
        "num_inference_steps": original_policy.num_inference_steps,
        "original": {
            "avg_time": avg_original,
            "std_time": std_original,
            "frequency": freq_original,
            "flops": float(flops_original)
        },
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    if pruned_policy is not None:
        benchmark_results.update({
            "pruned": {
                "avg_time": avg_pruned,
                "std_time": std_pruned,
                "frequency": freq_pruned,
                "flops": float(flops_pruned)
            },
            "speedup": speedup,
            "flops_reduction": float((flops_original - flops_pruned) / flops_original) if flops_original > 0 else 0.0
        })

    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f" {output_dir}/benchmark_results.json")

    
    logger.info("\n===  ===")
    env_runner_cfg = prepare_env_runner_config(cfg, skip_video)
    env_runner = instantiate_env_runner(env_runner_cfg, output_dir)
    
    #  batch_size (n_envs)
    n_envs = len(env_runner.env_fns) if hasattr(env_runner, 'env_fns') else 1
    logger.info(f" n_envs={n_envs} ")
    
    #  reuse_block  n_envs != 1 n_envs  tgt_sa
    if reuse_block and n_envs != 1:
        logger.info(f"[reuse_block]  n_envs={n_envs}  tgt_sa...")
        tgt_sa_env = reuse_block_module(pruner_path, cfg, torch_device, batch_size=n_envs)
        logger.info(f"[reuse_block] tgt_sa_env : {tgt_sa_env.shape}")
        
        #  pruned_policy  tgt_sa_env
        logger.info("[reuse_block]  pruned_policy  tgt_sa...")
        _, pruned_policy = create_policies(base_policy, torch_device, pruner_path, cfg, one_gate, reuse_block, tgt_sa_env)

    eval_policy = pruned_policy if pruned_policy is not None else original_policy
    
    # Save pruning imagesgatePruned policysetupgate
    gate_tracker = None
    soft_gate_tracker = None
    
    if (save_pruning_image or save_gate) and pruned_policy is not None:
        logger.info("gate...")
        # Configsample
        n_test = getattr(env_runner_cfg, 'n_test', 50)  # 50
        last_sample_idx = n_test - 1
        
        # gate
        if save_gate:
            # gatesample
            gate_tracker = GateTracker(sample_indices=[0])
            soft_gate_tracker = SoftGateTracker(sample_indices=[0])
            logger.info("gatesample")
        else:
            # sample
            gate_tracker = GateTracker(sample_indices=[last_sample_idx])
        
        def track_gates(policy):
            if hasattr(policy, '_cache') and policy._cache.get("gate") is not None:
                gates = policy._cache["gate"]  # [batch, T, B, N]
                if isinstance(gates, torch.Tensor) and gates.dim() == 4:
                    batch_size = gates.shape[0]
                    
                    # block ()
                    if not gate_tracker.block_names and hasattr(policy, '_cache_block_keys'):
                        gate_tracker.block_names = policy._cache_block_keys
                        if soft_gate_tracker:
                            soft_gate_tracker.block_names = policy._cache_block_keys
                    
                    # hard gate
                    if save_gate:
                        # sample
                        if batch_size > 0:
                            sample_gates = gates[0:1]  # [1, T, B, N]
                            gate_tracker.gates_sequence.append((gate_tracker.current_rollout_step, sample_gates.clone()))
                    else:
                        # sample
                        if batch_size > 0:
                            last_idx = batch_size - 1
                            sample_gates = gates[last_idx:last_idx+1]  # [1, T, B, N]
                            gate_tracker.gates_sequence.append((gate_tracker.current_rollout_step, sample_gates.clone()))
                    
                    gate_tracker.current_rollout_step += 1
                    
                    # soft gate
                    if soft_gate_tracker:
                        soft_gate = policy._cache.get("soft_gate", None)
                        if soft_gate is not None and len(soft_gate.shape) == 4:  # [batch, T, B, N]
                            if batch_size > 0:
                                sample_soft_gate = soft_gate[0].detach().cpu().numpy()  # [T, B, N]
                                soft_gate_tracker.multi_sample_soft_gates[0].append(
                                    (soft_gate_tracker.current_rollout_step, sample_soft_gate)
                                )
                        soft_gate_tracker.current_rollout_step += 1
                    
                    logger.debug(f"Tracked gates for rollout step {gate_tracker.current_rollout_step-1}")
        
        # policy
        original_predict_action = eval_policy.predict_action
        def tracked_predict_action(obs_dict):
            result = original_predict_action(obs_dict)
            track_gates(eval_policy)
            return result
        eval_policy.predict_action = tracked_predict_action
    
    logger.info("...")
    runner_log = env_runner.run(eval_policy)

    test_mean_score = runner_log.get('test/mean_score', 0.0)

    eval_results = {
        "mean_score": float(test_mean_score),
        "speedup": float(speedup),
        "flops": float(flops_pruned),
        "mode": mode,
        "num_trials": int(num_trials),
        "ddim_steps": ddim if ddim is not None else None,
        "scheduler_type": type(eval_policy.noise_scheduler).__name__,
        "num_inference_steps": eval_policy.num_inference_steps
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f" {output_dir}/eval_results.json")

    json_log = process_runner_log(runner_log)
    out_path = os.path.join(output_dir, 'detailed_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f" {out_path}")
    
    # gate
    if save_gate and gate_tracker is not None:
        logger.info("gatevisualization/draw/gate...")
        try:
            gate_save_dir = os.path.join('visualization', 'draw', 'gate', task_name, str(epoch))
            os.makedirs(gate_save_dir, exist_ok=True)
            
            # multi_sample_gatesexport_gate_data
            gate_tracker.multi_sample_gates = {0: gate_tracker.gates_sequence}
            if soft_gate_tracker:
                # soft_gate_tracker.multi_sample_soft_gates
                pass
            
            # gate
            export_gate_data(gate_tracker, soft_gate_tracker, gate_save_dir)
            logger.info(f"Gate: {gate_save_dir}")
        except Exception as e:
            logger.error(f"gate: {e}")
            import traceback
            traceback.print_exc()
    
    # GIF
    if save_pruning_image and gate_tracker is not None and gate_tracker.gates_sequence:
        logger.info("GIF...")
        try:
            # block
            if hasattr(eval_policy, '_cache_block_keys'):
                block_names = eval_policy._cache_block_keys
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
                gif_path = os.path.join(output_dir, f'gate_evolution_last_sample_epoch{epoch}.gif')
                
                title_prefix = f"Gate Evolution (Last Sample) - {task_name.upper()} (Epoch {epoch})"
                
                # gate
                _, first_gates = gate_tracker.gates_sequence[0]
                gate_dim = first_gates.shape[-1] if isinstance(first_gates, torch.Tensor) else 3
                
                # rollout-cache trajectory gates
                if rollout_cache and gate_dim == 4:
                    logger.info(f"Trajectory gates enabled (gate_dim={gate_dim})")
                    create_gate_animation_trajectory(
                        gates_sequence=gate_tracker.gates_sequence,
                        block_names=block_names,
                        output_path=gif_path,
                        title_prefix=title_prefix,
                        duration=0.3  # 0.3
                    )
                else:
                    logger.info(f"Standard gates enabled (gate_dim={gate_dim})")
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