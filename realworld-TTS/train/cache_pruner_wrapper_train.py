# Transformer Cache Pruner Wrapper - Training Version
# Cache pruning mechanism for Diffusion Policy Transformer models
# Based on TTSInfer/acceleration/rollout/pruner_warpper_train.py

from __future__ import annotations

import sys
from pathlib import Path
#  TTSInfer
# - this_file: realworld-TTS/train/cache_pruner_wrapper_train.py
# - repo_root : realworld-TTS
# - realworld_sag_root: RealWorld-SAG TTSInfer/
this_file = Path(__file__).resolve()
repo_root = this_file.parent.parent
realworld_sag_root = repo_root.parent
ttsinfer_root = realworld_sag_root / "TTSInfer"
if ttsinfer_root.exists() and str(ttsinfer_root) not in sys.path:
    #  TTSInfer  sys.path `import pruner.*`
    sys.path.insert(0, str(ttsinfer_root))

import types
import logging
from typing import Any, Dict, List, Tuple, Optional
try:
    # sys.path  TTSInfer  import pruner.*
    from pruner.train.gate_scheduler import apply_scheduler_batch
except ImportError:
    #  RealWorld-SAG  sys.path
    from TTSInfer.pruner.train.gate_scheduler import apply_scheduler_batch

import re
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CachePrunerWrapper:
    """
    Diffusion PolicyTransformerstep×block
    
    Design:
    - Gating: layer-specific, pruner generates per-layer per-step gate decisions
    - Cache3cache+ 24cacheper-layer+ rollout cacheframe
    - Training: soft-gated mixed output with gradient support
    """

    @staticmethod
    def apply(
        policy: Any,
        pruner: Optional[nn.Module] = None,
        if_rollout_cache: bool = False,
        training: bool = None,
        one_gate: bool = False,
    ) -> Any:
        """Apply cache + pruning wrapper to policy"""
        assert hasattr(policy, "model"), "policy must have a 'model' attribute"
        model = policy.model
        model.training = training
        #  wrapper cache  pruner/flags
        if getattr(policy, "_cpw_wrapped", False):
            cache_existing = getattr(policy, "_cache", None)
            if cache_existing is None:
                cache_existing = {}
            cache_existing["pruner"] = pruner
            cache_existing["training"] = training
            cache_existing["one_gate"] = one_gate
            cache_existing["if_rollout_cache"] = if_rollout_cache
            policy._cache = cache_existing
            logger.info("CachePrunerWrapper.apply: already applied — updated cache.pruner/training/flags and returned")
            return policy

        # Initialize cache context
        cache: Dict[str, Any] = {
            "current_step": -1,
            "block_cache_3": {},
            "block_cache_24": {},
            "block_cache_rollout": {},
            "gate": None,
            "pruner": pruner,
            "training": training,
            "one_gate": one_gate,
            "encoder_buffer_batch": None,
            "is_first_predict_action_in_chunk": None,  # Marks the first predict_action in a chunk (first rollout step)
        }
        policy._cache = cache
        #  predict_action/reset
        policy._cpw_wrapped = True

        # Find decoder layers
        cacheable_layers: List[Tuple[str, nn.Module]] = CachePrunerWrapper._find_decoder_layers(model)
        policy._cacheable_layers = cacheable_layers
        policy._cache_block_keys = CachePrunerWrapper._enumerate_block_keys(cacheable_layers)
        #  diffusion step  block = decoder_layers * 3
        #  24 decoder  != 8 assembly_bun  12 =36 blocks
        # /
        cache["num_blocks_per_step"] = len(policy._cache_block_keys)
        logger.info(f"Found {len(cacheable_layers)} TransformerDecoderLayer(s) to inject with cache gating")

        # Inject per-layer forward
        if if_rollout_cache:
            for layer_name, layer in cacheable_layers:
                CachePrunerWrapper._inject_layer_forward_rollout(layer, layer_name, policy, training)
        else:
            for layer_name, layer in cacheable_layers:
                CachePrunerWrapper._inject_layer_forward(layer, layer_name, policy, training)

        def integrated_forward(self, sample, timestep, cond=None, **kwargs):
            """Async forward: Step0 batch-computes encoders + async pruner; Step1-99 uses precomputed results"""
            cache_ctx = getattr(policy, "_cache", {})
            cache_ctx["current_step"] = cache_ctx.get("current_step", -1) + 1
            cur_step = cache_ctx["current_step"]

            # Step 0: Batch compute encoders, launch pruner async
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

        # Wrap predict_action: reset cache before each call
        original_predict_action = policy.predict_action

        def predict_action_with_reset(self, *args, **kwargs):
            # Reset cache before each predict_action (diffusion)
            cache_ctx = getattr(policy, "_cache", {})
            
            # Check if this is the first predict_action in the chunk (first rollout step)
            is_first_predict = cache_ctx.get("is_first_predict_action_in_chunk", False)
            
            # Decide whether to clear rollout cache based on flag
            CachePrunerWrapper.reset_cache(policy, None, clear_rollout_cache=is_first_predict)
            
            # If first predict_action in chunk, clear rollout cache to force full computation
            if is_first_predict:
                cache_ctx["is_first_predict_action_in_chunk"] = False

            return original_predict_action(*args, **kwargs)

        policy.predict_action = types.MethodType(predict_action_with_reset, policy)

        # Wrap reset: mark rollout cache for re-initialization at chunk start
        original_reset = policy.reset
        
        def reset_with_chunk_flag(self):
            cache_ctx = getattr(policy, "_cache", {})
            # Mark next predict_action as first in chunk, requiring rollout cache re-init
            cache_ctx["is_first_predict_action_in_chunk"] = True
            logger.info("[Chunk start] Marking next predict_action as first rollout step (full compute)")
            return original_reset()
        
        policy.reset = types.MethodType(reset_with_chunk_flag, policy)

        return policy

    @staticmethod
    def _batch_compute_all_encoders(policy: Any) -> None:
        """Batch compute all-step encoders (synchronous)"""
        cache_ctx = policy._cache
        model = policy.model
        cond = cache_ctx.get("cond_for_workers")
        device = next(model.parameters()).device
        batch_size = cond.shape[0]
        
        with torch.no_grad():
            # Use scheduler.timesteps instead of simple arange to support DDIM skip-step sampling
            scheduler = policy.noise_scheduler
            timesteps_batch = scheduler.timesteps.to(device)
            num_steps = len(timesteps_batch)
            cache_ctx["num_steps"] = num_steps
            
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
    def _unwrap_ddp(module):
        """ DDP """
        if hasattr(module, "module"):
            return module.module
        return module

    @staticmethod
    def _batch_compute_pruner(policy: Any) -> None:
        """pruner"""
        cache_ctx = policy._cache
        training = cache_ctx.get("training", None)
        pruner_raw = cache_ctx.get("pruner")
        num_steps = cache_ctx.get("num_steps", 100)
        encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")

        if pruner_raw is None or encoder_buffer_batch is None:
            return

        #  DDP
        pruner = CachePrunerWrapper._unwrap_ddp(pruner_raw)

        if training:
            pruner_raw.train()
        else:
            pruner_raw.eval()
        
        batch_size = encoder_buffer_batch.shape[0]
        
        memory_batch = encoder_buffer_batch.view(batch_size * num_steps, -1, encoder_buffer_batch.shape[-1])
        
        block_keys = policy._cache_block_keys
        memory_proj_batch = pruner.memory_proj(memory_batch)
        block_ids = pruner._get_block_ids(block_keys)
        block_emb = pruner.block_emb(block_ids)
        tgt_batch = block_emb.unsqueeze(0).expand(batch_size * num_steps, -1, -1)
        decoder_output_batch = pruner.transformer_decoder(tgt_batch, memory_proj_batch)
        logits_batch = pruner.head(decoder_output_batch)
        logits_batch = logits_batch.view(batch_size, num_steps, -1, 4)
        logits_batch = torch.flip(logits_batch, dims=[1])
        
        # apply_scheduler_batch
        soft_gate, hard_gate = apply_scheduler_batch(
            logits=logits_batch,
            num_steps=num_steps,
        )
        cache_ctx["soft_gate"] = soft_gate
        cache_ctx["gate"] = hard_gate


    @staticmethod
    def reset_cache_train(policy: Any, obs_dict: Any = None) -> None:
        cache_ctx = getattr(policy, "_cache", None)
        if cache_ctx is None:
            return
        cache_ctx["current_step"] = -1
        cache_ctx["block_cache_3"] = {"sa_block": None, "mha_block": None, "ff_block": None}
        
        block_cache_24 = {}
        cacheable_layers = getattr(policy, "_cacheable_layers", [])
        num_layers = len(cacheable_layers)
        for layer_idx in range(num_layers):
            block_cache_24[f"layer_{layer_idx}_sa_block"] = None
            block_cache_24[f"layer_{layer_idx}_mha_block"] = None
            block_cache_24[f"layer_{layer_idx}_ff_block"] = None
        cache_ctx["block_cache_24"] = block_cache_24

    @staticmethod
    def reset_cache(policy: Any, obs_dict: Any = None, clear_rollout_cache: bool = False) -> None:
        """Reset cache"""
        cache_ctx = getattr(policy, "_cache", None)
        if cache_ctx is None:
            return
        cache_ctx["current_step"] = -1
        cache_ctx["num_steps"] = getattr(policy, "num_inference_steps", 100)

        if cache_ctx.get("one_gate", False) and cache_ctx.get("first_rollout_gate", None) is not None:
            cache_ctx["gate"] = cache_ctx["first_rollout_gate"].clone()
        else:
            cache_ctx["gate"] = None

        cache_ctx["strategy_dict"] = {}
        cache_ctx["encoder_buffer_batch"] = None
        cache_ctx["cond_for_workers"] = None

        cache_ctx["block_cache_3"] = {"sa_block": None, "mha_block": None, "ff_block": None}
        block_cache_24 = {}
        cacheable_layers = getattr(policy, "_cacheable_layers", [])
        num_layers = len(cacheable_layers)
        for layer_idx in range(num_layers):
            block_cache_24[f"layer_{layer_idx}_sa_block"] = None
            block_cache_24[f"layer_{layer_idx}_mha_block"] = None
            block_cache_24[f"layer_{layer_idx}_ff_block"] = None
        cache_ctx["block_cache_24"] = block_cache_24
        
        # Clear rollout cache based on parameter
        if clear_rollout_cache:
            cache_ctx["block_cache_rollout"] = {}
            cache_ctx["is_first_rollout_step"] = True
            logger.info("[First Rollout Step] Clearing rollout cache for full-compute initialization")
        else:
            cache_ctx["is_first_rollout_step"] = False

    @staticmethod
    def reset_rollout_cache(policy: Any) -> None:
        """rollout cache"""
        cache_ctx = getattr(policy, "_cache", None)
        if cache_ctx is None:
            return
        cache_ctx["block_cache_rollout"] = {}

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
    def _inject_layer_forward(layer: nn.Module, layer_name: str, policy: Any, training: bool) -> None:
        """forward3"""
        cache_ctx = policy._cache
        match = re.search(r'\.(\d+)$', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                               tgt_key_padding_mask=None, memory_key_padding_mask=None,
                               tgt_is_causal=None, memory_is_causal=None):
            
            cur_step = cache_ctx.get("current_step", 0)
            block_cache_3 = cache_ctx["block_cache_3"]
            block_cache_24 = cache_ctx["block_cache_24"]

            cache_3_sa_key = "sa_block"
            cache_3_mha_key = "mha_block" 
            cache_3_ff_key = "ff_block"
            
            cache_24_sa_key = f"layer_{layer_idx}_sa_block"
            cache_24_mha_key = f"layer_{layer_idx}_mha_block"
            cache_24_ff_key = f"layer_{layer_idx}_ff_block"

            local_sa_block_idx = layer_idx * 3 + 0
            local_mha_block_idx = layer_idx * 3 + 1  
            local_ff_block_idx = layer_idx * 3 + 2

            num_blocks_per_step = cache_ctx.get("num_blocks_per_step", 24)
            sa_block_idx = cur_step * num_blocks_per_step + local_sa_block_idx
            mha_block_idx = cur_step * num_blocks_per_step + local_mha_block_idx
            ff_block_idx = cur_step * num_blocks_per_step + local_ff_block_idx
            
            is_not_batch = True if tgt.shape[0] == 1 else False
            
            # Step 0Force full compute
            if cur_step == 0:
                sa_strategy = 0
                mha_strategy = 0
                ff_strategy = 0
                hard_gate = cache_ctx.get("gate", None)
            else:
                strategy_dict = cache_ctx.get("strategy_dict", {})
                hard_gate = cache_ctx.get("gate", None)
                sa_strategy = strategy_dict.get(sa_block_idx, 0)
                mha_strategy = strategy_dict.get(mha_block_idx, 0)
                ff_strategy = strategy_dict.get(ff_block_idx, 0)
            
            x = tgt

            # SA Block
            if is_not_batch:
                with torch.no_grad():
                    if sa_strategy == 0:
                        sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                        x = x + sa_cur
                        block_cache_3[cache_3_sa_key] = sa_cur
                        block_cache_24[cache_24_sa_key] = sa_cur
                    elif sa_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_sa_key, None)
                        if cached_3 is not None:
                            x = x + cached_3
                        else:
                            sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                            x = x + sa_cur
                            block_cache_3[cache_3_sa_key] = sa_cur
                            block_cache_24[cache_24_sa_key] = sa_cur
                    elif sa_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_sa_key, None)
                        if cached_24 is not None:
                            x = x + cached_24
                        else:
                            sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                            x = x + sa_cur
                            block_cache_3[cache_3_sa_key] = sa_cur
                            block_cache_24[cache_24_sa_key] = sa_cur
            else:
                w = hard_gate[:, cur_step, local_sa_block_idx, :]
                g0, g1, g2 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
                g0 = g0.unsqueeze(-1)
                g1 = g1.unsqueeze(-1)
                g2 = g2.unsqueeze(-1)
                
                sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                cached_3 = block_cache_3.get(cache_3_sa_key, None)
                cached_24 = block_cache_24.get(cache_24_sa_key, None)
                
                if cached_3 is None or cached_24 is None:
                    x = x + sa_cur
                    block_cache_3[cache_3_sa_key] = sa_cur.detach()
                    block_cache_24[cache_24_sa_key] = sa_cur.detach()
                else:
                    mix = sa_cur * g0 + cached_3 * g1 + cached_24 * g2
                    x = x + mix
                    condition = g0.squeeze(-1).squeeze(-1).bool()
                    block_cache_3[cache_3_sa_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached_3)
                    block_cache_24[cache_24_sa_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached_24)

            # MHA Block
            if is_not_batch:
                with torch.no_grad():
                    if mha_strategy == 0:
                        mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                        x = x + mha_cur
                        block_cache_3[cache_3_mha_key] = mha_cur
                        block_cache_24[cache_24_mha_key] = mha_cur
                    elif mha_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_mha_key, None)
                        if cached_3 is not None:
                            x = x + cached_3
                        else:
                            mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                            x = x + mha_cur
                            block_cache_3[cache_3_mha_key] = mha_cur
                            block_cache_24[cache_24_mha_key] = mha_cur
                    elif mha_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_mha_key, None)
                        if cached_24 is not None:
                            x = x + cached_24
                        else:
                            mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                            x = x + mha_cur
                            block_cache_3[cache_3_mha_key] = mha_cur
                            block_cache_24[cache_24_mha_key] = mha_cur
            else:
                w = hard_gate[:, cur_step, local_mha_block_idx, :]
                g0, g1, g2 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
                g0 = g0.unsqueeze(-1)
                g1 = g1.unsqueeze(-1)
                g2 = g2.unsqueeze(-1)
                
                mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                cached_3 = block_cache_3.get(cache_3_mha_key, None)
                cached_24 = block_cache_24.get(cache_24_mha_key, None)
                
                if cached_3 is None or cached_24 is None:
                    x = x + mha_cur
                    block_cache_3[cache_3_mha_key] = mha_cur.detach()
                    block_cache_24[cache_24_mha_key] = mha_cur.detach()
                else:
                    mix = mha_cur * g0 + cached_3 * g1 + cached_24 * g2
                    x = x + mix
                    condition = g0.squeeze(-1).squeeze(-1).bool()
                    block_cache_3[cache_3_mha_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached_3)
                    block_cache_24[cache_24_mha_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached_24)

            # FF Block
            if is_not_batch:
                with torch.no_grad():
                    if ff_strategy == 0:
                        ff_cur = self._ff_block(self.norm3(x))
                        x = x + ff_cur
                        block_cache_3[cache_3_ff_key] = ff_cur
                        block_cache_24[cache_24_ff_key] = ff_cur
                    elif ff_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_ff_key, None)
                        if cached_3 is not None:
                            x = x + cached_3
                        else:
                            ff_cur = self._ff_block(self.norm3(x))
                            x = x + ff_cur
                            block_cache_3[cache_3_ff_key] = ff_cur
                            block_cache_24[cache_24_ff_key] = ff_cur
                    elif ff_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_ff_key, None)
                        if cached_24 is not None:
                            x = x + cached_24
                        else:
                            ff_cur = self._ff_block(self.norm3(x))
                            x = x + ff_cur
                            block_cache_3[cache_3_ff_key] = ff_cur
                            block_cache_24[cache_24_ff_key] = ff_cur
            else:
                w = hard_gate[:, cur_step, local_ff_block_idx, :]
                g0, g1, g2 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
                g0 = g0.unsqueeze(-1)
                g1 = g1.unsqueeze(-1)
                g2 = g2.unsqueeze(-1)
                
                ff_cur = self._ff_block(self.norm3(x))
                cached_3 = block_cache_3.get(cache_3_ff_key, None)
                cached_24 = block_cache_24.get(cache_24_ff_key, None)
                
                if cached_3 is None or cached_24 is None:
                    x = x + ff_cur
                    block_cache_3[cache_3_ff_key] = ff_cur.detach()
                    block_cache_24[cache_24_ff_key] = ff_cur.detach()
                else:
                    mix = ff_cur * g0 + cached_3 * g1 + cached_24 * g2
                    x = x + mix
                    condition = g0.squeeze(-1).squeeze(-1).bool()
                    cached_3 = block_cache_3[cache_3_ff_key]
                    cached_24 = block_cache_24[cache_24_ff_key]
                    block_cache_3[cache_3_ff_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached_3)
                    block_cache_24[cache_24_ff_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached_24)

            return x

        layer.forward = types.MethodType(forward_with_cache, layer)


    @staticmethod
    def _inject_layer_forward_rollout(layer: nn.Module, layer_name: str, policy: Any, training: bool) -> None:
        """forward4rollout cache- TTSInfer"""
        cache_ctx = policy._cache
        match = re.search(r'\.(\d+)$', layer_name)
        layer_idx = int(match.group(1)) if match else 0

        def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                               tgt_key_padding_mask=None, memory_key_padding_mask=None,
                               tgt_is_causal=None, memory_is_causal=None):
            
            cur_step = cache_ctx.get("current_step", 0)
            block_cache_3 = cache_ctx["block_cache_3"]   
            block_cache_24 = cache_ctx["block_cache_24"] 
            block_cache_rollout = cache_ctx["block_cache_rollout"]  

            cache_3_sa_key = "sa_block"
            cache_3_mha_key = "mha_block" 
            cache_3_ff_key = "ff_block"
            
            cache_24_sa_key = f"layer_{layer_idx}_sa_block"
            cache_24_mha_key = f"layer_{layer_idx}_mha_block"
            cache_24_ff_key = f"layer_{layer_idx}_ff_block"

            local_sa_block_idx = layer_idx * 3 + 0
            local_mha_block_idx = layer_idx * 3 + 1  
            local_ff_block_idx = layer_idx * 3 + 2

            num_blocks_per_step = cache_ctx.get("num_blocks_per_step", 24)
            sa_block_idx = cur_step * num_blocks_per_step + local_sa_block_idx
            mha_block_idx = cur_step * num_blocks_per_step + local_mha_block_idx
            ff_block_idx = cur_step * num_blocks_per_step + local_ff_block_idx

            is_not_batch = True if tgt.shape[0] == 1 else False

            is_first_rollout = cache_ctx.get("is_first_rollout_step", False)
            
            # Debug:
            if layer_idx == 0 and cur_step == 0:
                has_rollout_cache = len(cache_ctx.get("block_cache_rollout", {})) > 0
                logger.debug(f"[Layer0-Step0] is_first_rollout={is_first_rollout}, has_rollout_cache={has_rollout_cache}")
            
            if is_first_rollout:
                sa_strategy = 0
                mha_strategy = 0
                ff_strategy = 0
                hard_gate = None
            else:
                strategy_dict = cache_ctx.get("strategy_dict", {})
                hard_gate = cache_ctx.get("gate", None)
                sa_strategy = strategy_dict.get(sa_block_idx, 0)
                mha_strategy = strategy_dict.get(mha_block_idx, 0)
                ff_strategy = strategy_dict.get(ff_block_idx, 0)

            cache_rollout_sa_key = f"step_{cur_step}_layer_{layer_idx}_sa_block"
            cache_rollout_mha_key = f"step_{cur_step}_layer_{layer_idx}_mha_block"
            cache_rollout_ff_key = f"step_{cur_step}_layer_{layer_idx}_ff_block"

            x = tgt
            
            # ----- SA Block -----
            if is_not_batch:
                with torch.no_grad():  # 
                    if sa_strategy == 0:
                        sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                        x = x + sa_cur
                        block_cache_3[cache_3_sa_key] = sa_cur
                        block_cache_24[cache_24_sa_key] = sa_cur
                        block_cache_rollout[cache_rollout_sa_key] = sa_cur

                    elif sa_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_sa_key, None)
                        x = x + cached_3

                    elif sa_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_sa_key, None)
                        x = x + cached_24
    
                    elif sa_strategy == 3:
                        cached_rollout = block_cache_rollout.get(cache_rollout_sa_key, None)
                        x = x + cached_rollout
        
            else:
                if is_first_rollout:          
                    sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    x = x + sa_cur
                    block_cache_rollout[cache_rollout_sa_key] = sa_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_sa_block_idx, :]
                    g0, g1, g2, g3 = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
                    
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    g2 = g2.unsqueeze(-1)
                    g3 = g3.unsqueeze(-1)
                    
                    sa_cur = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                    
                    cached_3 = block_cache_3.get(cache_3_sa_key, None)
                    cached_24 = block_cache_24.get(cache_24_sa_key, None)
                    cached_rollout = block_cache_rollout.get(cache_rollout_sa_key, None)
                    
                    if cached_3 is None or cached_24 is None or cached_rollout is None:
                        # diffusion rollout cache cache
                        x = x + sa_cur
                        block_cache_3[cache_3_sa_key] = sa_cur.detach()
                        block_cache_24[cache_24_sa_key] = sa_cur.detach()
                        block_cache_rollout[cache_rollout_sa_key] = sa_cur.detach()
                    else:
                        mix = sa_cur * g0 + cached_3 * g1 + cached_24 * g2 + cached_rollout * g3
                        x = x + mix
                        
                        condition = g0.squeeze(-1).squeeze(-1).bool()  
                        
                        block_cache_3[cache_3_sa_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached_3)
                        block_cache_24[cache_24_sa_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached_24)
                        block_cache_rollout[cache_rollout_sa_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), sa_cur.detach(), cached_rollout)

            # MHA Block (SA Block)
            if is_not_batch:
                with torch.no_grad():  # 
                    if mha_strategy == 0:
                        mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                        x = x + mha_cur
                        block_cache_3[cache_3_mha_key] = mha_cur
                        block_cache_24[cache_24_mha_key] = mha_cur
                        block_cache_rollout[cache_rollout_mha_key] = mha_cur
                    elif mha_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_mha_key, None)
                        x = x + cached_3
                    elif mha_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_mha_key, None)
                        x = x + cached_24
                    elif mha_strategy == 3:
                        cached_rollout = block_cache_rollout.get(cache_rollout_mha_key, None)
                        x = x + cached_rollout
            else:
                if is_first_rollout:
                    mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    x = x + mha_cur
                    block_cache_rollout[cache_rollout_mha_key] = mha_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_mha_block_idx, :]
                    g0, g1, g2, g3 = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    g2 = g2.unsqueeze(-1)
                    g3 = g3.unsqueeze(-1)
                    
                    mha_cur = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
                    cached_3 = block_cache_3.get(cache_3_mha_key, None)
                    cached_24 = block_cache_24.get(cache_24_mha_key, None)
                    cached_rollout = block_cache_rollout.get(cache_rollout_mha_key, None)
                    
                    if cached_3 is None or cached_24 is None or cached_rollout is None:
                        # diffusion rollout cache cache
                        x = x + mha_cur
                        block_cache_3[cache_3_mha_key] = mha_cur.detach()
                        block_cache_24[cache_24_mha_key] = mha_cur.detach()
                        block_cache_rollout[cache_rollout_mha_key] = mha_cur.detach()
                    else:
                        mix = mha_cur * g0 + cached_3 * g1 + cached_24 * g2 + cached_rollout * g3
                        x = x + mix
                        condition = g0.squeeze(-1).squeeze(-1).bool()
                        block_cache_3[cache_3_mha_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached_3)
                        block_cache_24[cache_24_mha_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached_24)
                        block_cache_rollout[cache_rollout_mha_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), mha_cur.detach(), cached_rollout)

            # FF Block (SA Block)
            if is_not_batch:
                with torch.no_grad():  # 
                    if ff_strategy == 0:
                        ff_cur = self._ff_block(self.norm3(x))
                        x = x + ff_cur
                        block_cache_3[cache_3_ff_key] = ff_cur
                        block_cache_24[cache_24_ff_key] = ff_cur
                        block_cache_rollout[cache_rollout_ff_key] = ff_cur
                    elif ff_strategy == 1:
                        cached_3 = block_cache_3.get(cache_3_ff_key, None)
                        x = x + cached_3
                    elif ff_strategy == 2:
                        cached_24 = block_cache_24.get(cache_24_ff_key, None)
                        x = x + cached_24
                    elif ff_strategy == 3:
                        cached_rollout = block_cache_rollout.get(cache_rollout_ff_key, None)
                        x = x + cached_rollout
            else:
                if is_first_rollout:
                    ff_cur = self._ff_block(self.norm3(x))
                    x = x + ff_cur
                    block_cache_rollout[cache_rollout_ff_key] = ff_cur.detach()
                else:
                    w = hard_gate[:, cur_step, local_ff_block_idx, :]
                    g0, g1, g2, g3 = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
                    g0 = g0.unsqueeze(-1)
                    g1 = g1.unsqueeze(-1)
                    g2 = g2.unsqueeze(-1)
                    g3 = g3.unsqueeze(-1)
                    
                    ff_cur = self._ff_block(self.norm3(x))
                    cached_3 = block_cache_3.get(cache_3_ff_key, None)
                    cached_24 = block_cache_24.get(cache_24_ff_key, None)
                    cached_rollout = block_cache_rollout.get(cache_rollout_ff_key, None)
                    
                    if cached_3 is None or cached_24 is None or cached_rollout is None:
                        # diffusion rollout cache cache
                        x = x + ff_cur
                        block_cache_3[cache_3_ff_key] = ff_cur.detach()
                        block_cache_24[cache_24_ff_key] = ff_cur.detach()
                        block_cache_rollout[cache_rollout_ff_key] = ff_cur.detach()
                    else:
                        mix = ff_cur * g0 + cached_3 * g1 + cached_24 * g2 + cached_rollout * g3
                        x = x + mix
                        
                        # Update cache only when compute branch is selected
                        condition = g0.squeeze(-1).squeeze(-1).bool()
                        
                        block_cache_3[cache_3_ff_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached_3)
                        block_cache_24[cache_24_ff_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached_24)
                        block_cache_rollout[cache_rollout_ff_key] = torch.where(
                            condition.unsqueeze(-1).unsqueeze(-1), ff_cur.detach(), cached_rollout)

            return x

        layer.forward = types.MethodType(forward_with_cache, layer)
