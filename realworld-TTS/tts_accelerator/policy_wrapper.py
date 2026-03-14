"""
TTS Policy Wrapper - 


1.  policy 
2. 
3. /
"""
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class TTSPolicyWrapper:
    """
    TTS  Policy 
    
    
    -  policy 
    - 4compute, 3cache, 24cache, rollout_cache
    -  encoder  +  pruner
    - /
    
    
        policy = load_dp_policy(checkpoint_path)
        pruner = load_pruner(pruner_path)
        accelerated_policy = TTSPolicyWrapper(policy, pruner)
        
        #  policy
        action = accelerated_policy.predict_action(obs_dict)
    """
    
    def __init__(
        self,
        policy: Any,
        pruner: Optional[torch.nn.Module] = None,
        enable_sag: bool = True,
        training: bool = False,
        if_rollout_cache: bool = True,
    ):
        """
         TTS Policy Wrapper
        
        Args:
            policy:  Diffusion Policy
            pruner:  Pruner TTS 
            enable_sag:  TTS 
            training: 
            if_rollout_cache:  rollout cache4
        """
        self.original_policy = policy
        self.pruner = pruner
        self.enable_sag = enable_sag and (pruner is not None)
        self.training = training
        self.if_rollout_cache = if_rollout_cache
        self._stats = {}
        
        if self.enable_sag:
            self._apply_tts_acceleration()
            self.policy = self.original_policy
        else:
            self.policy = self.original_policy
            if enable_sag:
                logger.warning("TTS acceleration disabled: no pruner model provided")
    
    def _apply_tts_acceleration(self):
        """Apply TTS acceleration (using CachePrunerWrapper)"""
        from .cache_pruner_wrapper import CachePrunerWrapper
        
        logger.info("Applying TTS acceleration to policy...")
        logger.info("  - Using batch encoder computation")
        
        cache_mode = "3cache + 24cache + rollout_cache (4)" if self.if_rollout_cache else "3cache + 24cache (3)"
        logger.info(f"  - Cache mode: {cache_mode}")
        
        CachePrunerWrapper.apply(
            policy=self.original_policy,
            pruner=self.pruner,
            training=self.training,
            if_rollout_cache=self.if_rollout_cache,
            one_gate=False,
        )
        
        logger.info("✓ TTS acceleration applied (batch computation + multi-level caching)")
    
    def predict_action(self, obs_dict):
        """
        
        
        Args:
            obs_dict: 
            
        Returns:
            
        """
        return self.policy.predict_action(obs_dict)
    
    def reset(self):
        """ policy """
        self.policy.reset()
        
        if self.enable_sag and hasattr(self.policy, '_cache'):
            logger.debug("TTS cache will be reset by wrapper")
        
        return self
    
    def eval(self):
        """setupEvaluation mode"""
        self.policy.eval()
        return self
    
    def train(self, mode: bool = True):
        """setup"""
        self.policy.train(mode)
        return self
    
    def to(self, device):
        """Device"""
        self.policy.to(device)
        
        if self.enable_sag and self.pruner is not None:
            self.pruner.to(device)
        
        return self
    
    def __getattr__(self, name):
        """
         policy
        
         policy 
        """
        if name in ('original_policy', 'policy', 'pruner', 'enable_sag', 
                    'training', 'if_rollout_cache', '_stats'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.policy, name)
    
    # ==========  ==========
    
    @property
    def n_obs_steps(self):
        return self.policy.n_obs_steps
    
    @property
    def n_action_steps(self):
        return self.policy.n_action_steps
    
    @property
    def num_inference_steps(self):
        return self.policy.num_inference_steps
    
    @num_inference_steps.setter
    def num_inference_steps(self, value):
        self.policy.num_inference_steps = value
    
    @property
    def model(self):
        """ pruner """
        return self.policy.model
    
    @property
    def noise_scheduler(self):
        return self.policy.noise_scheduler
    
    # ========== TTS  ==========
    
    def get_stats(self):
        """ TTS """
        if not self.enable_sag or not hasattr(self.policy, '_cache'):
            return None
        
        cache_ctx = self.policy._cache
        timing = cache_ctx.get('timing', {})
        total_steps = timing.get('total_steps', 0)
        
        cache_type = '3cache + 24cache + rollout_cache' if self.if_rollout_cache else '3cache + 24cache'
        
        stats = {
            'sag_enabled': True,
            'mode': 'TTS',
            'rollout_cache': self.if_rollout_cache,
            'cache_type': cache_type,
        }
        
        if total_steps > 0:
            stats.update({
                'encoder_time': timing.get('encoder_step0', 0),
                'pruner_submit_time': timing.get('submit_pruner', 0),
                'decoder_time': timing.get('decoder', 0),
                'total_steps': total_steps,
            })
        else:
            stats['expected_cache_ratio'] = 0.92
        
        return stats
    
    def disable_acceleration(self):
        """/"""
        self.enable_sag = False
        logger.info("TTS acceleration disabled at runtime")
    
    def enable_acceleration(self):
        """"""
        if self.pruner is not None:
            self.enable_sag = True
            logger.info("TTS acceleration re-enabled")
        else:
            logger.warning("Cannot enable TTS: no pruner model")
