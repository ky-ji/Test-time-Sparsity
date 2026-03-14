"""
TTS Config

 TTS ConfigConfig
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class TTSConfig:
    """TTS Config"""
    
    enabled: bool = True
    
    # Pruner Config
    pruner_checkpoint: Optional[str] = None  # pruner 
    target_prune_ratio: float = 0.93
    
    # Pruner
    pruner_hidden_dim: int = 512
    pruner_decoder_layers: int = 1
    pruner_block_encoder: str = "SA"
    pruner_attn_heads: int = 8
    pruner_dim_feedforward: int = 1024
    pruner_dropout: float = 0.1
    pruner_reuse_dp_encoder: bool = False
    pruner_reuse_block: bool = False
    
    if_rollout_cache: bool = True  # 4
    cache_strategy: str = "Omini"
    
    async_pruner: bool = True
    use_threading: bool = True
    
    # /
    enable_timing_stats: bool = True
    log_stats_interval: int = 100
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "enabled": self.enabled,
            "type": "tts",
            "pruner_checkpoint": self.pruner_checkpoint,
            "target_prune_ratio": self.target_prune_ratio,
            "wrapper_kwargs": {
                "enable_sag": self.enabled,
                "training": False,
                "if_rollout_cache": self.if_rollout_cache,
            },
            "pruner_config": {
                "hidden_dim": self.pruner_hidden_dim,
                "decoder_layers": self.pruner_decoder_layers,
                "block_encoder": self.pruner_block_encoder,
                "attn_heads": self.pruner_attn_heads,
                "dim_feedforward": self.pruner_dim_feedforward,
                "dropout": self.pruner_dropout,
                "reuse_dp_encoder": self.pruner_reuse_dp_encoder,
                "reuse_block": self.pruner_reuse_block,
                "if_rollout_cache": self.if_rollout_cache,
            },
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TTSConfig":
        """Config"""
        pruner_config = d.get("pruner_config", {})
        wrapper_kwargs = d.get("wrapper_kwargs", {})
        
        return cls(
            enabled=d.get("enabled", True),
            pruner_checkpoint=d.get("pruner_checkpoint"),
            target_prune_ratio=d.get("target_prune_ratio", 0.93),
            pruner_hidden_dim=pruner_config.get("hidden_dim", 512),
            pruner_decoder_layers=pruner_config.get("decoder_layers", 1),
            pruner_block_encoder=pruner_config.get("block_encoder", "SA"),
            pruner_attn_heads=pruner_config.get("attn_heads", 8),
            pruner_dim_feedforward=pruner_config.get("dim_feedforward", 1024),
            pruner_dropout=pruner_config.get("dropout", 0.1),
            pruner_reuse_dp_encoder=pruner_config.get("reuse_dp_encoder", False),
            pruner_reuse_block=pruner_config.get("reuse_block", False),
            if_rollout_cache=pruner_config.get("if_rollout_cache", True),
        )
    
    def validate(self) -> None:
        """Config"""
        if self.enabled and not self.pruner_checkpoint:
            raise ValueError("TTS enabled but pruner_checkpoint not specified")
        
        if self.pruner_checkpoint:
            path = Path(self.pruner_checkpoint).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Pruner checkpoint not found: {path}")


# Config
PRESET_CONFIGS = {
    "assembly_chocolate": TTSConfig(
        pruner_hidden_dim=512,
        pruner_decoder_layers=1,
        target_prune_ratio=0.93,
        if_rollout_cache=True,
    ),
    "pick": TTSConfig(
        pruner_hidden_dim=512,
        pruner_decoder_layers=1,
        target_prune_ratio=0.92,
        if_rollout_cache=True,
    ),
}


def get_preset_config(task_name: str) -> TTSConfig:
    """Config"""
    if task_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[task_name]
