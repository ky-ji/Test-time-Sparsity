"""
 Diffusion Policy  DDIM Scheduler  denoising steps 

Usage:
python TTSInfer/scripts/eval_ddim_steps.py --ddim_steps 8 16 32 50 100 --device cuda:0 --output_dir ddim_eval_results

python TTSInfer/scripts/eval_ddim_steps.py --ddim_steps 50 --device cuda:6 --output_dir ddim_eval_results --train_id 1

python TTSInfer/scripts/eval_ddim_steps.py --ddim_steps 50 --device cuda:7 --output_dir ddim_eval_results --train_id 2

python eval_ddim_steps.py --ddim_steps 30 --device cuda:2 --output_dir ddim_eval_results --train_id 2

python eval_ddim_steps.py --ddim_steps 40 --device cuda:3 --output_dir ddim_eval_results

python eval_ddim_steps.py --ddim_steps 50 --device cuda:4 --output_dir ddim_eval_results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import pathlib
import click
import hydra
import torch
import dill
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_ddim_steps")

def get_task_checkpoint(task_name: str, train_id: int = 0) -> str:
    """checkpoint"""
    checkpoint_map = {
        'can_mh': f'checkpoint/can_mh/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'can_ph': f'checkpoint/can_ph/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'lift_mh': f'checkpoint/lift_mh/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'lift_ph': f'checkpoint/lift_ph/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'square_mh': f'checkpoint/square_mh/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'square_ph': f'checkpoint/square_ph/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'tool_hang_ph': f'checkpoint/tool_hang_ph/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'transport_mh': f'checkpoint/transport_mh/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'transport_ph': f'checkpoint/transport_ph/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
        'kitchen': f'checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_{train_id}/checkpoints/latest.ckpt',
    }
    
    if task_name not in checkpoint_map:
        raise ValueError(f"Task name: {task_name}")
    
    return checkpoint_map[task_name]


def load_policy_with_ddim(checkpoint_path: str, device: str, ddim_steps: int, train_id: int = 0):
    """
    policyDDIM schedulersetup
    
    Args:
        checkpoint_path: checkpoint
        device: Device
        ddim_steps: DDIM
    
    Returns:
        policy: Configpolicy
        cfg: Config object
    """
    logger.info(f"checkpoint: {checkpoint_path}")
    
    # checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # DDIM scheduler
    original_scheduler = policy.noise_scheduler
    logger.info(f"scheduler: {type(original_scheduler).__name__}")
    logger.info(f": {original_scheduler.config.num_train_timesteps}")
    
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
    
    # setup
    policy.num_inference_steps = ddim_steps
    
    logger.info(f"DDIM schedulersetup: {ddim_steps}")
    
    # Device
    torch_device = torch.device(device)
    policy.to(torch_device)
    policy.eval()
    
    return policy, cfg


def evaluate_task(task_name: str, checkpoint_path: str, device: str, 
                  ddim_steps: int, output_dir: str, skip_video: bool = True, train_id: int = 0) -> Dict[str, Any]:
    """
    DDIM
    
    Args:
        task_name: Task name
        checkpoint_path: checkpoint
        device: Device
        ddim_steps: DDIM
        output_dir: 
        skip_video: 
    
    Returns:
        
    """
    logger.info(f"\n{'='*60}")
    logger.info(f": {task_name}, DDIM: {ddim_steps}")
    logger.info(f"{'='*60}")
    
    try:
        # policy
        policy, cfg = load_policy_with_ddim(checkpoint_path, device, ddim_steps, train_id)
        
        task_output_dir = os.path.join(output_dir,f"{ddim_steps}ddim", f"train_{train_id}", task_name, f"ddim_{ddim_steps}")
        pathlib.Path(task_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configenv_runner
        env_runner_cfg = cfg.task.env_runner
        if skip_video:
            if hasattr(env_runner_cfg, 'video_dir'):
                env_runner_cfg.video_dir = None
            if hasattr(env_runner_cfg, 'n_video'):
                env_runner_cfg.n_video = 0
        
        # env_runner
        env_runner = hydra.utils.instantiate(
            env_runner_cfg,
            output_dir=task_output_dir
        )
        
        logger.info(f"...")
        runner_log = env_runner.run(policy)
        
        test_mean_score = runner_log.get('test/mean_score', 0.0)
        test_std_score = runner_log.get('test/std_score', 0.0)
        
        logger.info(f" {task_name} (DDIM {ddim_steps}) - : {test_mean_score:.4f} ± {test_std_score:.4f}")
        
        result = {
            'task_name': task_name,
            'ddim_steps': ddim_steps,
            'mean_score': float(test_mean_score),
            'std_score': float(test_std_score),
            'success': True,
            'checkpoint': checkpoint_path
        }
        
        detailed_log_path = os.path.join(task_output_dir, 'eval_log.json')
        json_log = {}
        for key, value in runner_log.items():
            if hasattr(value, '_path'):  # wandb video
                json_log[key] = value._path
            else:
                json_log[key] = value
        
        with open(detailed_log_path, 'w') as f:
            json.dump(json_log, f, indent=2, sort_keys=True)
        
        return result
        
    except Exception as e:
        logger.error(f" {task_name} (DDIM {ddim_steps}) : {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'task_name': task_name,
            'ddim_steps': ddim_steps,
            'mean_score': 0.0,
            'std_score': 0.0,
            'success': False,
            'error': str(e),
            'checkpoint': checkpoint_path
        }


@click.command()
@click.option('--ddim_steps', '-s', multiple=True, type=int, default=[8, 16, 32, 50, 100],
              help='DDIM: -s 8 -s 16 -s 32')
@click.option('--device', '-d', default='cuda:0', help='Device')
@click.option('--output_dir', '-o', default='ddim_eval_results', help='')
@click.option('--train_id', '-t', default=0, type=int, help='Training ID')
@click.option('--skip_video', is_flag=True, default=True, help='Skip video rendering to speed up evaluation')
@click.option('--tasks', multiple=True, type=str, 
              default=['can_mh', 'can_ph', 'lift_mh', 'lift_ph', 'square_mh', 
                      'square_ph', 'tool_hang_ph', 'kitchen', 'transport_mh', 'transport_ph'],
              help='')
def main(ddim_steps: tuple, device: str, output_dir: str, train_id: int, 
         skip_video: bool, tasks: tuple):
    """
    Diffusion PolicyDDIM
    """
    ddim_steps_list = list(ddim_steps)
    tasks_list = list(tasks)
    
    logger.info("="*80)
    logger.info("Diffusion Policy DDIM Steps ")
    logger.info("="*80)
    logger.info(f": {tasks_list}")
    logger.info(f"DDIM: {ddim_steps_list}")
    logger.info(f"Training ID: {train_id}")
    logger.info(f"Device: {device}")
    logger.info(f": {output_dir}")
    logger.info("="*80)
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # DDIM
    for task_name in tasks_list:
        # checkpoint
        try:
            checkpoint_path = get_task_checkpoint(task_name, train_id)
            
            # checkpoint
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint: {checkpoint_path} {task_name}")
                continue
            
            # DDIM
            for steps in ddim_steps_list:
                result = evaluate_task(
                    task_name=task_name,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    ddim_steps=steps,
                    output_dir=output_dir,
                    skip_video=skip_video,
                    train_id=train_id
                )
                all_results.append(result)
                
        except Exception as e:
            logger.error(f" {task_name} : {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("Starting evaluation...")
    logger.info("="*80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'train_id': train_id,
        'device': device,
        'ddim_steps_evaluated': ddim_steps_list,
        'tasks_evaluated': tasks_list,
        'results_by_task': {},
        'results_by_steps': {},
        'all_results': all_results
    }
    
    for result in all_results:
        task = result['task_name']
        if task not in summary['results_by_task']:
            summary['results_by_task'][task] = []
        summary['results_by_task'][task].append({
            'ddim_steps': result['ddim_steps'],
            'mean_score': result['mean_score'],
            'std_score': result['std_score'],
            'success': result['success']
        })
    
    for result in all_results:
        steps = result['ddim_steps']
        steps_key = f"ddim_{steps}"
        if steps_key not in summary['results_by_steps']:
            summary['results_by_steps'][steps_key] = []
        summary['results_by_steps'][steps_key].append({
            'task_name': result['task_name'],
            'mean_score': result['mean_score'],
            'std_score': result['std_score'],
            'success': result['success']
        })
    
    summary['average_by_steps'] = {}
    for steps_key, results in summary['results_by_steps'].items():
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_score = sum(r['mean_score'] for r in successful_results) / len(successful_results)
            summary['average_by_steps'][steps_key] = {
                'average_score': avg_score,
                'num_tasks': len(successful_results),
                'total_tasks': len(results)
            }
    
    # JSON
    summary_path = os.path.join(output_dir, 'ddim_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    logger.info(f"\n: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("")
    logger.info("="*80)
    
    logger.info("\n:")
    for task, results in summary['results_by_task'].items():
        logger.info(f"\n: {task}")
        for r in sorted(results, key=lambda x: x['ddim_steps']):
            status = "✓" if r['success'] else "✗"
            logger.info(f"  DDIM {r['ddim_steps']:3d}: {r['mean_score']:.4f} ± {r['std_score']:.4f} {status}")
    
    logger.info("\nDDIM ():")
    for steps_key in sorted(summary['average_by_steps'].keys(), 
                           key=lambda x: int(x.split('_')[1])):
        avg_info = summary['average_by_steps'][steps_key]
        steps = int(steps_key.split('_')[1])
        logger.info(f"  DDIM {steps:3d}: {avg_info['average_score']:.4f} "
                   f"({avg_info['num_tasks']}/{avg_info['total_tasks']} )")
    
    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

