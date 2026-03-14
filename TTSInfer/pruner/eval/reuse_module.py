from TTSInfer.pruner.train.transformer_pruner import enumerate_decoder_block_keys
import copy
from TTSInfer.pruner.train.transformer_pruner import TransformerPruner, SinusoidalPosEmb
import torch.nn as nn
import torch
        


def reuse_block(pruner_path, cfg, torch_device, batch_size=1):
    """
     SA block  tgt_sa
    
    Args:
        pruner_path: pruner 
        cfg: Config object
        torch_device: Device
        batch_size: 1
        
    Returns:
        tgt_sa:  self-attention block  [batch_size, B, hidden_dim]
    """
    checkpoint = torch.load(pruner_path, map_location='cpu')
    hidden_dim = checkpoint['hidden_dim']
    attn_heads = checkpoint['attn_heads']
    dim_feedforward = checkpoint['dim_feedforward']
    dropout = checkpoint.get('dropout', 0.1)
    decoder_layers = checkpoint.get('decoder_layers', 1)
    state_dict = checkpoint['model_state_dict']
    
    #  layer_names  block_keys
    if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'layer_names'):
        layer_names = cfg.policy.model.layer_names
    else:
        layer_names = [f"decoder.layers.{i}" for i in range(8)]

    block_keys = enumerate_decoder_block_keys(layer_names)
    block_emb_module = SinusoidalPosEmb(hidden_dim).to(torch_device)
    
    #  block IDs  embeddings
    block_to_id = {name: i for i, name in enumerate(block_keys)}
    block_ids = torch.tensor([block_to_id[b] for b in block_keys], dtype=torch.long, device=torch_device)
    block_emb = block_emb_module(block_ids)  # [B, hidden_dim]
    
    #  batch
    tgt = block_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, B, hidden_dim]
    
    #  transformer decoder  checkpoint
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=attn_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    temp_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
    
    #  checkpoint  transformer_decoder
    decoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('transformer_decoder.'):
            #  'transformer_decoder.'
            new_key = key.replace('transformer_decoder.', '')
            decoder_state_dict[new_key] = value
    
    if decoder_state_dict:
        temp_decoder.load_state_dict(decoder_state_dict)
        print(f"[reuse_block]  checkpoint  transformer_decoder ")
    else:
        print(f"[reuse_block] : checkpoint Found transformer_decoder ")
    
    temp_decoder = temp_decoder.to(torch_device)
    
    #  self-attention block  tgt_sa
    with torch.no_grad():
        temp_decoder.eval()
        #  self-attention
        tgt_sa = tgt.clone()
        for mod in temp_decoder.layers:
            #  self-attention block
            if mod.norm_first:
                tgt_sa = tgt_sa + mod._sa_block(mod.norm1(tgt_sa), None, None)
            else:
                tgt_sa = mod.norm1(tgt_sa + mod._sa_block(tgt_sa, None, None))


    return tgt_sa