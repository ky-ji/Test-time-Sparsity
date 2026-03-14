import os
import torch
from datetime import datetime
from TTSInfer.pruner.train.transformer_pruner import TransformerPruner
from TTSInfer.pruner.train.train_utils import enumerate_decoder_block_keys


def get_stage1_ckpt(train_id, task_name, epoch):
    """ stage1 checkpoint """
    pruner_base_path = os.path.join('stage1result', 'train', f'train{train_id}', task_name)

    pruner_files = [f for f in os.listdir(pruner_base_path) if f.startswith(f'pruner_model_{epoch}_') and f.endswith('.pt')]
    if pruner_files:
        best_file = min(pruner_files, key=lambda x: float(x.split('_')[2].replace('.pt', '')))
        stage1_pruner_path = os.path.join(pruner_base_path, best_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage2_output_path = os.path.join('stage1result', 'stage2ckpt', "2stage", timestamp, str(train_id), task_name, str(epoch))
    os.makedirs(stage2_output_path, exist_ok=True)

    return stage1_pruner_path, stage2_output_path


def load_stage1_pruner(stage1_pruner_path, cfg, policy, device):
    """Load trained pruner model"""
    # Get model parameters
    num_steps = getattr(cfg, 'num_inference_steps', 100)

    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]

    block_keys = enumerate_decoder_block_keys(layer_names)
    
    checkpoint = torch.load(stage1_pruner_path, map_location='cpu')
    
    #  checkpoint
    hidden_dim = checkpoint['hidden_dim']
    attn_heads = checkpoint['attn_heads']
    dim_feedforward = checkpoint['dim_feedforward']
    block_encoder_type = checkpoint['block_encoder_type']

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
    ).to(device)
    
    if os.path.exists(stage1_pruner_path):
        pruner.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pruner model: {stage1_pruner_path}")
        print(f"  - epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  - valid_loss: {checkpoint.get('valid_loss', 'unknown')}")
        print(f"  - pruning_ratio: {checkpoint.get('valid_pruning_ratio', 'unknown')}")
    else:
        print(f"Pruner model file not found: {stage1_pruner_path}")
        return None
    
    pruner.to(device)
    pruner.eval()
    return pruner

def add_new_head_dim(pruner, init_bias_stage2):
    """
    pruner34
    
    Args:
        pruner: TransformerPruner
        
    Returns:
        pruner: pruner
    """
    device = next(pruner.parameters()).device
    old_head = pruner.head
    
    # head
    old_weight = old_head.weight.data  # shape: [3, hidden_dim]
    old_bias = old_head.bias.data 
    
    hidden_dim = old_weight.shape[1]
    
    # 4
    new_head = torch.nn.Linear(hidden_dim, 4).to(device)
    
    # 3
    with torch.no_grad():
        # 3
        new_head.weight.data[:3, :] = old_weight
        if old_bias is not None:
            new_head.bias.data[:3] = old_bias
        
        # 4Xavier
        torch.nn.init.xavier_uniform_(new_head.weight.data[3:4, :])
        if new_head.bias is not None:
            torch.nn.init.zeros_(new_head.bias.data[3:4])
            # rollout cache
            new_head.bias.data[3:4] += init_bias_stage2

    # prunerhead
    pruner.head = new_head
    
    print(f"pruner34")
     
    return pruner