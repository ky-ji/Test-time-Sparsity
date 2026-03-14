"""
setup
"""

import inspect
import logging
import hydra
from typing import Any

logger = logging.getLogger(__name__)


def instantiate_env_runner(env_runner_cfg: Any, output_dir: str) -> Any:
    """
    
    
    Args:
        env_runner_cfg: Env runner config
        output_dir: 
        
    Returns:
        
    """
    try:
        return hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    except Exception as e:
        logger.error(f": {e}")
        return _fallback_instantiate_env_runner(env_runner_cfg, output_dir, e)


def _fallback_instantiate_env_runner(env_runner_cfg: Any, output_dir: str, original_error: Exception) -> Any:
    """
    
    
    Args:
        env_runner_cfg: Env runner config
        output_dir: 
        original_error: 
        
    Returns:
        
        
    Raises:
        Exception: 
    """
    try:
        from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner
        from diffusion_policy.env_runner.blockpush_lowdim_runner import BlockPushLowdimRunner

        runner_class = None
        cfg_str = str(env_runner_cfg)
        
        if "KitchenLowdimRunner" in cfg_str:
            runner_class = KitchenLowdimRunner
            logger.info("ConfigKitchenLowdimRunner")
        elif "BlockPushLowdimRunner" in cfg_str:
            runner_class = BlockPushLowdimRunner
            logger.info("ConfigBlockPushLowdimRunner")

        if runner_class:
            # Config
            cleaned_cfg = _clean_runner_config(runner_class, env_runner_cfg, output_dir)
            return hydra.utils.instantiate(cleaned_cfg)
        else:
            raise original_error
            
    except Exception as inner_e:
        logger.error(f": {inner_e}")
        raise original_error


def _clean_runner_config(runner_class: type, env_runner_cfg: Any, output_dir: str) -> Any:
    """
    Config
    
    Args:
        runner_class: 
        env_runner_cfg: Env runner config
        output_dir: 
        
    Returns:
        Config
    """
    if not isinstance(env_runner_cfg, dict):
        env_runner_cfg = dict(env_runner_cfg)
    
    sig = inspect.signature(runner_class.__init__)
    required_params = set(sig.parameters.keys()) - {'self'}
    
    for param in list(env_runner_cfg.keys()):
        if param not in required_params and param != '_target_':
            logger.info(f": {param}")
            del env_runner_cfg[param]
    
    env_runner_cfg['output_dir'] = output_dir
    
    return env_runner_cfg 