"""
TTS Accelerator -  Diffusion Policy 


1. pip install -e /path/to/realworld-TTS
2.  DP ConfigACCELERATOR = {'enabled': True, 'type': 'tts', ...}
3.  DP 


- TTSPolicyWrapper:  policy TTS 
- load_pruner:  pruner 
- CachePrunerWrapper: Low-level cache pruning mechanism
"""

from .policy_wrapper import TTSPolicyWrapper
from .pruner_loader import load_pruner
from .config import TTSConfig

__version__ = "0.1.0"
__all__ = ["TTSPolicyWrapper", "load_pruner", "TTSConfig"]
