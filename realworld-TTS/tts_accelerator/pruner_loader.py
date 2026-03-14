"""
Pruner 

 pruner  checkpoint 
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_pruner(
    policy: Any,
    config: Union[Dict[str, Any], "TTSConfig"],
    device: str = "cuda",
) -> Optional[nn.Module]:
    """
     Pruner 
    
    Args:
        policy: Diffusion Policy encoder 
        config: TTS Config TTSConfig 
        device: Device
        
    Returns:
         Pruner  None
    """
    # Config
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    
    pruner_checkpoint = config.get('pruner_checkpoint')
    if not pruner_checkpoint:
        logger.warning("No pruner checkpoint specified")
        return None
    
    pruner_config = config.get('pruner_config', {})
    
    try:
        #  checkpoint
        pruner_path = _find_checkpoint(pruner_checkpoint)
        if pruner_path is None:
            logger.error(f"Pruner checkpoint not found: {pruner_checkpoint}")
            return None
        
        logger.info(f"Loading TTS pruner: {pruner_path}")
        
        #  policy
        model = policy.model
        block_names, num_layers = _extract_block_names(model)
        
        if len(block_names) == 0:
            logger.error("No TransformerDecoderLayer found in model!")
            return None
        
        logger.info(f"  - Detected {num_layers} TransformerDecoderLayers")
        logger.info(f"  - Block keys: {len(block_names)} ({num_layers} layers × 3 sub-blocks)")
        
        #  encoder
        components = _extract_encoder_components(policy)
        
        num_steps = getattr(policy, 'num_inference_steps', 100)
        n_obs_steps = getattr(policy, 'n_obs_steps', 2)
        
        #  Pruner
        pruner = _create_pruner(
            num_steps=num_steps,
            block_names=block_names,
            pruner_config=pruner_config,
            n_obs_steps=n_obs_steps,
            **components
        )
        
        #  checkpoint
        _load_checkpoint(pruner, pruner_path)
        
        # Device
        pruner.to(device)
        pruner.eval()
        
        logger.info("✓ TTS Pruner loaded successfully")
        return pruner
        
    except Exception as e:
        logger.error(f"Failed to load TTS pruner: {e}")
        import traceback
        traceback.print_exc()
        return None


def _find_checkpoint(path: str) -> Optional[str]:
    """ checkpoint """
    p = Path(path).expanduser()
    if p.exists():
        return str(p)
    return None


def _extract_block_names(model: nn.Module) -> Tuple[List[str], int]:
    """ block """
    cacheable_layers = []
    
    #  decoder.layers
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        for i, layer in enumerate(model.decoder.layers):
            if isinstance(layer, nn.TransformerDecoderLayer):
                name = f"decoder.layers.{i}"
                cacheable_layers.append((name, layer))
    
    # Fallback: full scan
    if not cacheable_layers:
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerDecoderLayer):
                cacheable_layers.append((name, module))
    
    #  block keys
    block_names = []
    for layer_name, _ in cacheable_layers:
        block_names.append(f"{layer_name}_sa_block")
        block_names.append(f"{layer_name}_mha_block")
        block_names.append(f"{layer_name}_ff_block")
    
    return block_names, len(cacheable_layers)


def _extract_encoder_components(policy: Any) -> Dict[str, Any]:
    """ policy  encoder """
    components = {
        'obs_encoder': None,
        'obs_emb': None,
        'time_emb': None,
        'position_embeddings': None,
        'encoder': None,
        'obs_dim': 0,
    }
    
    model = policy.model
    
    if hasattr(policy, 'obs_encoder'):
        components['obs_encoder'] = copy.deepcopy(policy.obs_encoder)
    
    if hasattr(model, 'cond_obs_emb'):
        components['obs_emb'] = copy.deepcopy(model.cond_obs_emb)
        components['obs_dim'] = model.cond_obs_emb.out_features
    
    if hasattr(model, 'time_emb'):
        components['time_emb'] = copy.deepcopy(model.time_emb)
    
    if hasattr(model, 'cond_pos_emb'):
        components['position_embeddings'] = copy.deepcopy(model.cond_pos_emb)
    
    if hasattr(model, 'encoder'):
        components['encoder'] = copy.deepcopy(model.encoder)
    
    return components


def _create_pruner(
    num_steps: int,
    block_names: List[str],
    pruner_config: Dict[str, Any],
    n_obs_steps: int,
    obs_encoder: Optional[nn.Module],
    obs_emb: Optional[nn.Module],
    time_emb: Optional[nn.Module],
    position_embeddings: Optional[nn.Module],
    encoder: Optional[nn.Module],
    obs_dim: int,
) -> nn.Module:
    """ Pruner """
    #  TransformerPruner
    TransformerPruner = None
    import_errors = []
    
    for module_path in [
        "TTSInfer.pruner.train.transformer_pruner",
        "pruner.train.transformer_pruner",
        "tts_accelerator.pruner.transformer_pruner",
    ]:
        try:
            import importlib
            module = importlib.import_module(module_path)
            TransformerPruner = module.TransformerPruner
            break
        except ImportError as e:
            import_errors.append(f"{module_path}: {e}")
    
    if TransformerPruner is None:
        raise ImportError(
            f"Could not import TransformerPruner from any known location.\n"
            f"Tried: {import_errors}"
        )
    
    pruner = TransformerPruner(
        max_steps=num_steps,
        block_names=block_names,
        hidden_dim=pruner_config.get('hidden_dim', 512),
        decoder_layers=pruner_config.get('decoder_layers', 1),
        block_encoder_type=pruner_config.get('block_encoder', 'SA'),
        time_emb=time_emb,
        position_embeddings=position_embeddings,
        attn_heads=pruner_config.get('attn_heads', 8),
        dim_feedforward=pruner_config.get('dim_feedforward', 1024),
        obs_dim=obs_dim,
        n_obs_steps=n_obs_steps,
        encoder=encoder,
        obs_encoder=obs_encoder,
        obs_emb=obs_emb,
        training=False,
        dropout=pruner_config.get('dropout', 0.1),
        reuse_dp_encoder=pruner_config.get('reuse_dp_encoder', False),
        reuse_block=pruner_config.get('reuse_block', False),
        tgt_sa=None,
        head_4=True,  # 4
        head_2=False,
    )
    
    return pruner


def _load_checkpoint(pruner: nn.Module, path: str) -> None:
    """ checkpoint  pruner"""
    checkpoint = torch.load(path, map_location='cpu')
    
    #  checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pruner.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            logger.info(f"  - Loaded from epoch {epoch}")
        elif 'state_dict' in checkpoint:
            pruner.load_state_dict(checkpoint['state_dict'])
        else:
            pruner.load_state_dict(checkpoint)
    else:
        pruner.load_state_dict(checkpoint)
