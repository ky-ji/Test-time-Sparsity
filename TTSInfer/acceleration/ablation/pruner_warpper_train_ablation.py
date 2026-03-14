# Cache - Wrapper
# 3cache24cacherollout_cachecache

from __future__ import annotations

import types
import logging
from typing import Any, Dict, List, Tuple, Optional
from TTSInfer.pruner.train.gate_scheduler import apply_scheduler_single, apply_scheduler_batch_ablation

import re
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CachePrunerWrapper:
    """
    CacheTransformer Cache Pruner Wrapper
    
    Design:
    - cache3cache24cacherollout_cache
    - 2computereuse
    - 3cache/24cache: diffusion stepForce full compute
    - rollout_cache: First rollout step: force full compute
    """

    @staticmethod
    def apply(
        policy: Any,
        pruner: Optional[nn.Module] = None,
        cache_type: str = '3cache',
        if_rollout: bool = False,
        training: bool = True,
    ) -> Any:
        """cache
        
        Args:
            policy: 
            pruner: pruner2
            cache_type: cache'3cache', '24cache', 'rollout_cache'
            if_rollout: rollout cache
            training: 
        """
        assert hasattr(policy, "model"), "policy must have a 'model' attribute"
        assert cache_type in ['3cache', '24cache', 'rollout_cache'], f"cache_type: {cache_type}"
        
        model = policy.model
        model.training = training

        # Initialize cache context
        cache: Dict[str, Any] = {
            "current_step": -1,
            "block_cache_3": {},
            "block_cache_24": {},
            "block_cache_rollout": {},
            "gate": None,
            "pruner": pruner,
            "training": training,
            "cache_type": cache_type,
            "encoder_buffer_batch": None,
            "is_first_predict_action_in_chunk": None,  # rollout cache
        }
        policy._cache = cache

        # Find decoder layers
        cacheable_layers: List[Tuple[str, nn.Module]] = CachePrunerWrapper._find_decoder_layers(model)
        policy._cacheable_layers = cacheable_layers
        policy._cache_block_keys = CachePrunerWrapper._enumerate_block_keys(cacheable_layers)
        logger.info(f"Found {len(cacheable_layers)}  TransformerDecoderLayercache: {cache_type}")

        # Inject per-layer forward
        for layer_name, layer in cacheable_layers:
            CachePrunerWrapper._inject_layer_forward_single_cache(layer, layer_name, policy, cache_type)

        def integrated_forward(self, sample, timestep, cond=None, **kwargs):
            """forward: encoderpruner"""
            cache_ctx = getattr(policy, "_cache", {})
            cache_ctx["current_step"] = cache_ctx.get("current_step", -1) + 1
            cur_step = cache_ctx["current_step"]

            # Step 0: encoderpruner
            if cur_step == 0 and cond is not None:
                cache_ctx["cond_for_workers"] = cond
                CachePrunerWrapper._batch_compute_all_encoders(policy)
                
                encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
                memory = encoder_buffer_batch[:, 0, :, :]

                CachePrunerWrapper._batch_compute_pruner(policy)
            
            # Step 1-99: Use precomputed results
            else:
                encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
                memory = encoder_buffer_batch[:, cur_step, :, :]
                
            # Decoder execution
            input_emb = self.input_emb(sample)
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)
            x = self.ln_f(x)
            x = self.head(x)

            return x

        model.forward = types.MethodType(integrated_forward, model)

        #  predict_action
        original_predict_action = policy.predict_action

        def predict_action_with_reset(self, *args, **kwargs):
            cache_ctx = getattr(policy, "_cache", {})
            cache_type = cache_ctx.get("cache_type", "None")
            
            # rollout cacherollout step
            if cache_type == 'rollout_cache':
                is_first_predict = cache_ctx.get("is_first_predict_action_in_chunk", False)
                CachePrunerWrapper.reset_cache(policy, clear_rollout_cache=is_first_predict)
                if is_first_predict:
                    cache_ctx["is_first_predict_action_in_chunk"] = False
            else:
                # 3cache24cacheresetdiffusion
                CachePrunerWrapper.reset_cache(policy, clear_rollout_cache=False)

            return original_predict_action(*args, **kwargs)

        policy.predict_action = types.MethodType(predict_action_with_reset, policy)

        #  resetrollout cache
        if cache_type == 'rollout_cache':
            original_reset = policy.reset
            
            def reset_with_chunk_flag(self):
                cache_ctx = getattr(policy, "_cache", {})
                cache_ctx["is_first_predict_action_in_chunk"] = True
                print("[Chunk] predict_actionrollout step")
                return original_reset()
            
            policy.reset = types.MethodType(reset_with_chunk_flag, policy)

        return policy

    @staticmethod
    def _batch_compute_all_encoders(policy: Any) -> None:
        """Batch compute all-step encoders (synchronous)"""
        cache_ctx = policy._cache
        model = policy.model
        num_steps = cache_ctx.get("num_steps", 100)
        cond = cache_ctx.get("cond_for_workers")
        device = next(model.parameters()).device
        batch_size = cond.shape[0]
        
        with torch.no_grad():
            timesteps_batch = torch.arange(num_steps-1, -1, -1, dtype=torch.long, device=device)
            time_emb_batch = model.time_emb(timesteps_batch)
            time_emb_batch = time_emb_batch.unsqueeze(0).expand(batch_size, -1, -1)
            time_emb_batch = time_emb_batch.unsqueeze(2)
            
            if model.obs_as_cond:
                cond_obs_emb = model.cond_obs_emb(cond)
                cond_obs_emb_batch = cond_obs_emb.unsqueeze(1).expand(-1, num_steps, -1, -1)
                cond_embeddings_batch = torch.cat([time_emb_batch, cond_obs_emb_batch], dim=2)
            else:
                cond_embeddings_batch = time_emb_batch
            
            bs, ns, seq_len, hidden_dim = cond_embeddings_batch.shape
            cond_embeddings_flat = cond_embeddings_batch.view(bs * ns, seq_len, hidden_dim)
            tc = cond_embeddings_flat.shape[1]
            position_embeddings = model.cond_pos_emb[:, :tc, :]
            x_batch = model.drop(cond_embeddings_flat + position_embeddings)
            
            memory_batch = model.encoder(x_batch)
            memory_batch_reshaped = memory_batch.view(batch_size, num_steps, -1, memory_batch.shape[-1])
            cache_ctx["encoder_buffer_batch"] = memory_batch_reshaped

    @staticmethod
    def _batch_compute_pruner(policy: Any) -> None:
        """pruner"""
        cache_ctx = policy._cache
        training = cache_ctx.get("training", True)
        pruner = cache_ctx.get("pruner")
        num_steps = cache_ctx.get("num_steps", 100)
        encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")

        if training:
            pruner.train()
        else:
            pruner.eval()

        if encoder_buffer_batch is None or pruner is None:
            return
        
        batch_size = encoder_buffer_batch.shape[0]
        
        memory_batch = encoder_buffer_batch.view(batch_size * num_steps, -1, encoder_buffer_batch.shape[-1])
        
        block_keys = policy._cache_block_keys
        memory_proj_batch = pruner.memory_proj(memory_batch)
        block_ids = pruner._get_block_ids(block_keys)
        block_emb = pruner.block_emb(block_ids)
        tgt_batch = block_emb.unsqueeze(0).expand(batch_size * num_steps, -1, -1)
        decoder_output_batch = pruner.transformer_decoder(tgt_batch, memory_proj_batch)
        logits_batch = pruner.head(decoder_output_batch)
        
        # cache2compute, reuse
        logits_batch = logits_batch.view(batch_size, num_steps, -1, 2)
        
        # logits 2 hardgate
        # apply_scheduler_batchsofthard gates
        soft_gate, hard_gate = apply_scheduler_batch_ablation(
            logits=logits_batch,
            num_steps=num_steps,
            cache_type= cache_ctx.get("cache_type", None),
        )         
        cache_ctx["soft_gate"] = soft_gate
        cache_ctx["gate"] = hard_gate
                    
    @staticmethod
    def reset_cache(policy: Any, obs_dict: Any = None, clear_rollout_cache: bool = False) -> None:
        """Reset cache
        
        Args:
            clear_rollout_cache: rollout cacherollout_cache
        """
        cache_ctx = getattr(policy, "_cache", None)
        if cache_ctx is None:
            return
        
        cache_ctx["current_step"] = -1
        cache_ctx["num_steps"] = getattr(policy, "num_inference_steps", 100)
        cache_ctx["gate"] = None
        cache_ctx["encoder_buffer_batch"] = None
        cache_ctx["cond_for_workers"] = None

        cache_type = cache_ctx.get("cache_type", None)
        
        # cache
        if cache_type == '3cache':
            cache_ctx["block_cache_3"] = {"sa_block": None, "mha_block": None, "ff_block": None}
        elif cache_type == '24cache':
            block_cache_24 = {}
            cacheable_layers = getattr(policy, "_cacheable_layers", [])
            num_layers = len(cacheable_layers)
            for layer_idx in range(num_layers):
                block_cache_24[f"layer_{layer_idx}_sa_block"] = None
                block_cache_24[f"layer_{layer_idx}_mha_block"] = None
                block_cache_24[f"layer_{layer_idx}_ff_block"] = None
            cache_ctx["block_cache_24"] = block_cache_24
        elif cache_type == 'rollout_cache':
            if clear_rollout_cache:
                cache_ctx["block_cache_rollout"] = {}
                cache_ctx["is_first_rollout_step"] = True
                print("[First Rollout Step] Clearing rollout cache for full-compute initialization")
            else:
                cache_ctx["is_first_rollout_step"] = False

    # ---------------- internals ----------------

    @staticmethod
    def _find_decoder_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        layers: List[Tuple[str, nn.Module]] = []
        decoder = getattr(model, "decoder", None)
        decoder_layers = getattr(decoder, "layers", None) if decoder is not None else None
        if isinstance(decoder_layers, (list, tuple)):
            for i, layer in enumerate(decoder_layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    name = f"decoder.layers.{i}"
                    layers.append((name, layer))
        if not layers:
            for name, module in model.named_modules():
                if isinstance(module, nn.TransformerDecoderLayer):
                    layers.append((name, module))
        return layers

    @staticmethod
    def _enumerate_block_keys(layers: List[Tuple[str, nn.Module]]) -> List[str]:
        keys: List[str] = []
        for layer_name, _ in layers:
            keys.append(f"{layer_name}_sa_block")
            keys.append(f"{layer_name}_mha_block")
            keys.append(f"{layer_name}_ff_block")
        return keys

    @staticmethod
    def _inject_layer_forward_single_cache(layer: nn.Module, layer_name: str, policy: Any, cache_type: str) -> None:
        """cacheforward
        
        Args:
            layer: decoder
            layer_name: 
            policy: 
            cache_type: cache3cache, 24cache, rollout_cache
        """
        cache_ctx = policy._cache
        match = re.search(r'\.(\d+)$', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                               tgt_key_padding_mask=None, memory_key_padding_mask=None,
                               tgt_is_causal=None, memory_is_causal=None):
            
            cur_step = cache_ctx.get("current_step", 0)
            hard_gate = cache_ctx.get("gate", None)
            
            # Block
            local_sa_block_idx = layer_idx * 3 + 0
            local_mha_block_idx = layer_idx * 3 + 1  
            local_ff_block_idx = layer_idx * 3 + 2
            
            is_batch = tgt.shape[0] > 1
            
            # cache_typecache
            if cache_type == '3cache':
                cache_dict = cache_ctx["block_cache_3"]
                sa_key = "sa_block"
                mha_key = "mha_block"
                ff_key = "ff_block"
                force_compute = (cur_step == 0)  # diffusion stepForce full compute
            elif cache_type == '24cache':
                cache_dict = cache_ctx["block_cache_24"]
                sa_key = f"layer_{layer_idx}_sa_block"
                mha_key = f"layer_{layer_idx}_mha_block"
                ff_key = f"layer_{layer_idx}_ff_block"
                force_compute = (cur_step == 0)  # diffusion stepForce full compute
            else:  # rollout_cache
                cache_dict = cache_ctx["block_cache_rollout"]
                sa_key = f"step_{cur_step}_layer_{layer_idx}_sa_block"
                mha_key = f"step_{cur_step}_layer_{layer_idx}_mha_block"
                ff_key = f"step_{cur_step}_layer_{layer_idx}_ff_block"
                force_compute = cache_ctx.get("is_first_rollout_step", False)  # First rollout step: force full compute
            
            x = tgt
            
            # ----- SA Block -----
            if not is_batch:
                # hard gate
                if force_compute or hard_gate is None:
                    strategy = 0  # 
                else:
                    # gate0=compute, 1=reuse
                    gate_value = hard_gate[0, cur_step, local_sa_block_idx, :]
                    strategy = torch.argmax(gate_value).item()
                
                if strategy == 0:  # compute
                    sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    x = x + sa_cur
                    # rollout_cachedetach3cache/24cachediffusion
                    if cache_type == 'rollout_cache':
                        cache_dict[sa_key] = sa_cur.detach()
                    else:
                        cache_dict[sa_key] = sa_cur
                else:  # reuse
                    cached = cache_dict.get(sa_key, None)
                    if cached is not None:
                        x = x + cached
                    else:
                        # fallback: cache
                        sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                        x = x + sa_cur
                        if cache_type == 'rollout_cache':
                            cache_dict[sa_key] = sa_cur.detach()
                        else:
                            cache_dict[sa_key] = sa_cur
            else:
                # Batchsoft gate
                if force_compute or hard_gate is None:
                    # Force full compute
                    sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    x = x + sa_cur
                    if cache_type == 'rollout_cache':
                        cache_dict[sa_key] = sa_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_sa_block_idx, :]
                    g0, g1 = w[:, 0:1], w[:, 1:2]  # compute, reuse
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    
                    sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    cached = cache_dict.get(sa_key, None)
                    
                    if cached is None:
                        x = x + sa_cur
                        cache_dict[sa_key] = sa_cur.detach()
                    else:
                        mix = sa_cur * g0 + cached * g1
                        x = x + mix
                        # cachecompute
                        condition = g0.squeeze(-1).squeeze(-1).bool()
                        cache_dict[sa_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached)
            
            # ----- MHA Block -----
            if not is_batch:
                if force_compute or hard_gate is None:
                    strategy = 0
                else:
                    gate_value = hard_gate[0, cur_step, local_mha_block_idx, :]
                    strategy = torch.argmax(gate_value).item()
                
                if strategy == 0:  # compute
                    mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    x = x + mha_cur
                    if cache_type == 'rollout_cache':
                        cache_dict[mha_key] = mha_cur.detach()
                    else:
                        cache_dict[mha_key] = mha_cur
                else:  # reuse
                    cached = cache_dict.get(mha_key, None)
                    if cached is not None:
                        x = x + cached
                    else:
                        mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                        x = x + mha_cur
                        if cache_type == 'rollout_cache':
                            cache_dict[mha_key] = mha_cur.detach()
                        else:
                            cache_dict[mha_key] = mha_cur
            else:
                if force_compute or hard_gate is None:
                    mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    x = x + mha_cur
                    if cache_type == 'rollout_cache':
                        cache_dict[mha_key] = mha_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_mha_block_idx, :]
                    g0, g1 = w[:, 0:1], w[:, 1:2]
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    
                    mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    cached = cache_dict.get(mha_key, None)
                    
                    if cached is None:
                        x = x + mha_cur
                        cache_dict[mha_key] = mha_cur.detach()
                    else:
                        mix = mha_cur * g0 + cached * g1
                        x = x + mix
                        condition = g0.squeeze(-1).squeeze(-1).bool()
                        cache_dict[mha_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached)
            
            # ----- FF Block -----
            if not is_batch:
                if force_compute or hard_gate is None:
                    strategy = 0
                else:
                    gate_value = hard_gate[0, cur_step, local_ff_block_idx, :]
                    strategy = torch.argmax(gate_value).item()
                
                if strategy == 0:  # compute
                    ff_cur = self._ff_block(self.norm3(x))
                    x = x + ff_cur
                    if cache_type == 'rollout_cache':
                        cache_dict[ff_key] = ff_cur.detach()
                    else:
                        cache_dict[ff_key] = ff_cur
                else:  # reuse
                    cached = cache_dict.get(ff_key, None)
                    if cached is not None:
                        x = x + cached
                    else:
                        ff_cur = self._ff_block(self.norm3(x))
                        x = x + ff_cur
                        if cache_type == 'rollout_cache':
                            cache_dict[ff_key] = ff_cur.detach()
                        else:
                            cache_dict[ff_key] = ff_cur
            else:
                if force_compute or hard_gate is None:
                    ff_cur = self._ff_block(self.norm3(x))
                    x = x + ff_cur
                    if cache_type == 'rollout_cache':
                        cache_dict[ff_key] = ff_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_ff_block_idx, :]
                    g0, g1 = w[:, 0:1], w[:, 1:2]
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    
                    ff_cur = self._ff_block(self.norm3(x))
                    cached = cache_dict.get(ff_key, None)
                    
                    if cached is None:
                        x = x + ff_cur
                        cache_dict[ff_key] = ff_cur.detach()
                    else:
                        mix = ff_cur * g0 + cached * g1
                        x = x + mix
                        condition = g0.squeeze(-1).squeeze(-1).bool()
                        cache_dict[ff_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached)
            
            return x

        layer.forward = types.MethodType(forward_with_cache, layer)

