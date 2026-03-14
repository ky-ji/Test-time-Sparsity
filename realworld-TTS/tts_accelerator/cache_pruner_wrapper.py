"""
Cache Pruner Wrapper - 

 TTS 
- Step×Block 
- 3cache + 24cache + rollout_cache 
-  encoder  +  pruner 

 acceleration/cache_pruner_wrapper_test.py 
      
"""

from __future__ import annotations

import logging
import re
import threading
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CachePrunerWrapper:
    """
     Diffusion Policy  Transformer  step×block 
    
    Design:
    - pruner 
    - Cache 3cache+ 24cacheper-layer+ rollout cacheframe
    - Parallelism:Step 0  encoder pruner
    """

    @staticmethod
    def apply(
        policy: Any,
        pruner: Optional[nn.Module] = None,
        if_rollout_cache: bool = True,
        training: bool = False,
        one_gate: bool = False,
    ) -> Any:
        """Apply cache + pruning wrapper to policy"""
        assert hasattr(policy, "model"), "policy must have a 'model' attribute"
        model = policy.model
        model.training = training

        #  cache
        cache: Dict[str, Any] = {
            "current_step": -1,
            "block_cache_3": {},
            "block_cache_24": {},
            "block_cache_rollout": {},
            "gate": None,
            "pruner": pruner,
            "training": training,
            "one_gate": one_gate,
            "if_rollout_cache": if_rollout_cache,
            "pruner_thread": None,
            "encoder_buffer_batch": None,
            "is_first_predict_action_in_chunk": True,
        }
        policy._cache = cache

        # Find decoder layers
        cacheable_layers: List[Tuple[str, nn.Module]] = CachePrunerWrapper._find_decoder_layers(model)
        policy._cacheable_layers = cacheable_layers
        policy._cache_block_keys = CachePrunerWrapper._enumerate_block_keys(cacheable_layers)
        #  diffusion step  block = decoder_layers * 3
        #  24 decoder  != 8
        cache["num_blocks_per_step"] = len(policy._cache_block_keys)
        logger.info(f"Found {len(cacheable_layers)} TransformerDecoderLayers for cache injection")

        #  forward
        for layer_name, layer in cacheable_layers:
            CachePrunerWrapper._inject_layer_forward_rollout(layer, layer_name, policy, training)

        #  model.forward
        CachePrunerWrapper._wrap_model_forward(model, policy)

        #  predict_action
        CachePrunerWrapper._wrap_predict_action(policy, training)

        #  reset
        CachePrunerWrapper._wrap_reset(policy)

        return policy

    @staticmethod
    def _wrap_model_forward(model: nn.Module, policy: Any) -> None:
        """ model.forward """
        
        def integrated_forward(self, sample, timestep, cond=None, **kwargs):
            cache_ctx = getattr(policy, "_cache", {})
            cache_ctx["current_step"] = cache_ctx.get("current_step", -1) + 1
            cur_step = cache_ctx["current_step"]

            if "timing" not in cache_ctx:
                cache_ctx["timing"] = {
                    "encoder_step0": 0.0,
                    "submit_pruner": 0.0,
                    "decoder": 0.0,
                    "total_steps": 0
                }

            # Step 0:  encoder pruner
            if cur_step == 0 and cond is not None:
                t0_enc = time.time()
                cache_ctx["cond_for_workers"] = cond
                CachePrunerWrapper._batch_compute_all_encoders(policy)
                cache_ctx["timing"]["encoder_step0"] = time.time() - t0_enc
                
                encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
                memory = encoder_buffer_batch[:, 0, :, :]
                
                #  pruner
                t0_submit = time.time()
                
                def pruner_worker():
                    try:
                        cache_ctx["_pruner_result"] = CachePrunerWrapper._batch_compute_pruner(
                            policy, write_result=False
                        )
                    except Exception as e:
                        logger.error(f"Pruner thread error: {e}")
                        cache_ctx["_pruner_result"] = None
                
                pruner_thread = threading.Thread(target=pruner_worker, daemon=True)
                pruner_thread.start()
                cache_ctx["pruner_thread"] = pruner_thread
                cache_ctx["timing"]["submit_pruner"] = time.time() - t0_submit
            
            # Step 1:  pruner
            elif cur_step == 1:
                encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
                memory = encoder_buffer_batch[:, cur_step, :, :]
                CachePrunerWrapper._sync_pruner_result(cache_ctx)
            
            # Step 2-99: Use precomputed results
            else:
                encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
                memory = encoder_buffer_batch[:, cur_step, :, :]

            # Decoder
            t0_dec = time.time()
            input_emb = self.input_emb(sample)
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)
            x = self.ln_f(x)
            x = self.head(x)
            
            # Step 0  pruner
            if cur_step == 0:
                CachePrunerWrapper._sync_pruner_result(cache_ctx)

            cache_ctx["timing"]["decoder"] += time.time() - t0_dec
            cache_ctx["timing"]["total_steps"] += 1

            return x

        model.forward = types.MethodType(integrated_forward, model)

    @staticmethod
    def _wrap_predict_action(policy: Any, training: bool) -> None:
        """ predict_action"""
        original_predict_action = policy.predict_action

        def predict_action_with_reset(self, *args, **kwargs):
            cache_ctx = getattr(policy, "_cache", {})
            
            if training:
                CachePrunerWrapper.reset_cache_train(policy, None)
            else:
                is_first_predict = cache_ctx.get("is_first_predict_action_in_chunk", False)
                CachePrunerWrapper.reset_cache(policy, None, clear_rollout_cache=is_first_predict)
                
                if is_first_predict:
                    cache_ctx["is_first_predict_action_in_chunk"] = False

            return original_predict_action(*args, **kwargs)

        policy.predict_action = types.MethodType(predict_action_with_reset, policy)

    @staticmethod
    def _wrap_reset(policy: Any) -> None:
        """ reset"""
        original_reset = policy.reset
        
        def reset_with_chunk_flag(self):
            cache_ctx = getattr(policy, "_cache", {})
            cache_ctx["is_first_predict_action_in_chunk"] = True
            logger.debug("[Chunk start] Marking next predict_action as first rollout step")
            return original_reset()
        
        policy.reset = types.MethodType(reset_with_chunk_flag, policy)

    @staticmethod
    def _sync_pruner_result(cache_ctx: Dict[str, Any]) -> None:
        """ pruner """
        pruner_thread = cache_ctx.get("pruner_thread")
        if pruner_thread is not None and pruner_thread.is_alive():
            pruner_thread.join()
        
        result = cache_ctx.get("_pruner_result")
        if result is not None:
            if isinstance(result, dict):
                cache_ctx["strategy_dict"] = result
            else:
                soft_gate, hard_gate = result
                cache_ctx["soft_gate"] = soft_gate
                cache_ctx["gate"] = hard_gate
        
        cache_ctx["pruner_thread"] = None
        cache_ctx["_pruner_result"] = None

    @staticmethod
    def _batch_compute_all_encoders(policy: Any) -> None:
        """ encoder"""
        cache_ctx = policy._cache
        model = policy.model
        num_steps = cache_ctx.get("num_steps", 100)
        cond = cache_ctx.get("cond_for_workers")
        device = next(model.parameters()).device
        batch_size = cond.shape[0]
        
        with torch.no_grad():
            scheduler = policy.noise_scheduler
            timesteps_batch = scheduler.timesteps.to(device)
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
    def _batch_compute_pruner(policy: Any, write_result: bool = True) -> Optional[Dict]:
        """ pruner """
        cache_ctx = policy._cache
        pruner = cache_ctx.get("pruner")
        num_steps = cache_ctx.get("num_steps", 100)
        encoder_buffer_batch = cache_ctx.get("encoder_buffer_batch")
        if_rollout_cache = cache_ctx.get("if_rollout_cache", False)

        if encoder_buffer_batch is None or pruner is None:
            logger.warning("Pruner computation skipped: encoder_buffer or pruner is None")
            return None

        pruner_unwrapped = pruner.module if hasattr(pruner, "module") else pruner

        batch_size = encoder_buffer_batch.shape[0]
        logger.debug(f"Pruner computation: batch_size={batch_size}, num_steps={num_steps}")

        with torch.no_grad():
            memory_batch = encoder_buffer_batch.view(batch_size * num_steps, -1, encoder_buffer_batch.shape[-1])

            block_keys = policy._cache_block_keys
            pruner.eval()
            memory_proj_batch = pruner_unwrapped.memory_proj(memory_batch)
            block_ids = pruner_unwrapped._get_block_ids(block_keys)
            block_emb = pruner_unwrapped.block_emb(block_ids)
            tgt_batch = block_emb.unsqueeze(0).expand(batch_size * num_steps, -1, -1)
            decoder_output_batch = pruner_unwrapped.transformer_decoder(tgt_batch, memory_proj_batch)
            logits_batch = pruner_unwrapped.head(decoder_output_batch)

            num_strategies = 4
            logits_batch = logits_batch.view(batch_size, num_steps, -1, num_strategies)
            logits_batch = torch.flip(logits_batch, dims=[1])

            #  gate_scheduler
            try:
                #  `scripts/run_dp_with_tts.py`  `RealWorld-SAG/TTSInfer`  sys.path
                #  `import pruner.train.*` `TTSInfer.pruner.train.*`
                from pruner.train.gate_scheduler import apply_scheduler_single, apply_scheduler_batch
            except ImportError:
                try:
                    #  RealWorld-SAG  sys.path
                    from TTSInfer.pruner.train.gate_scheduler import apply_scheduler_single, apply_scheduler_batch
                except ImportError:
                    #  scheduler step0  compute
                    from .gate_scheduler import apply_scheduler_single, apply_scheduler_batch

            if batch_size == 1:
                strategy_dict = apply_scheduler_single(
                    logits=logits_batch,
                    num_steps=num_steps,
                    batch_idx=0,
                )
                
                strategy_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                for strategy in strategy_dict.values():
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                total = sum(strategy_counts.values())
                logger.info(f"✓ Pruner strategy: compute={strategy_counts[0]}/{total} ({strategy_counts[0]/total*100:.1f}%), "
                          f"3cache={strategy_counts[1]}/{total} ({strategy_counts[1]/total*100:.1f}%), "
                          f"24cache={strategy_counts[2]}/{total} ({strategy_counts[2]/total*100:.1f}%), "
                          f"rollout={strategy_counts[3]}/{total} ({strategy_counts[3]/total*100:.1f}%)")

                if write_result:
                    cache_ctx["strategy_dict"] = strategy_dict
                return strategy_dict
            else:
                soft_gate, hard_gate = apply_scheduler_batch(
                    logits=logits_batch,
                    num_steps=num_steps,
                )
                if write_result:
                    cache_ctx["soft_gate"] = soft_gate
                    cache_ctx["gate"] = hard_gate
                return (soft_gate, hard_gate)

    @staticmethod
    def reset_cache_train(policy: Any, obs_dict: Any = None) -> None:
        """ cache"""
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
        """ cache"""
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
        cache_ctx["pruner_thread"] = None
        cache_ctx["_pruner_result"] = None
        cache_ctx["timing"] = {
            "encoder_step0": 0.0,
            "submit_pruner": 0.0,
            "decoder": 0.0,
            "total_steps": 0
        }

        cache_ctx["block_cache_3"] = {"sa_block": None, "mha_block": None, "ff_block": None}
        block_cache_24 = {}
        cacheable_layers = getattr(policy, "_cacheable_layers", [])
        num_layers = len(cacheable_layers)
        for layer_idx in range(num_layers):
            block_cache_24[f"layer_{layer_idx}_sa_block"] = None
            block_cache_24[f"layer_{layer_idx}_mha_block"] = None
            block_cache_24[f"layer_{layer_idx}_ff_block"] = None
        cache_ctx["block_cache_24"] = block_cache_24
        
        if clear_rollout_cache:
            cache_ctx["block_cache_rollout"] = {}
            cache_ctx["is_first_rollout_step"] = True
            logger.debug("[First rollout step] Clearing rollout cache")
        else:
            cache_ctx["is_first_rollout_step"] = False

    # ========== Internal helpers ==========

    @staticmethod
    def _find_decoder_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """ TransformerDecoderLayer"""
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
        """ block keys"""
        keys: List[str] = []
        for layer_name, _ in layers:
            keys.append(f"{layer_name}_sa_block")
            keys.append(f"{layer_name}_mha_block")
            keys.append(f"{layer_name}_ff_block")
        return keys

    @staticmethod
    def _inject_layer_forward_rollout(layer: nn.Module, layer_name: str, policy: Any, training: bool) -> None:
        """ forward 4  rollout cache"""
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
            
            # ===== SA Block =====
            x = CachePrunerWrapper._process_block(
                self, x, "sa", sa_strategy, is_not_batch, is_first_rollout, hard_gate,
                cur_step, local_sa_block_idx, memory, tgt_mask, tgt_key_padding_mask,
                memory_mask, memory_key_padding_mask,
                block_cache_3, cache_3_sa_key,
                block_cache_24, cache_24_sa_key,
                block_cache_rollout, cache_rollout_sa_key,
                layer_idx
            )
            
            # ===== MHA Block =====
            x = CachePrunerWrapper._process_block(
                self, x, "mha", mha_strategy, is_not_batch, is_first_rollout, hard_gate,
                cur_step, local_mha_block_idx, memory, tgt_mask, tgt_key_padding_mask,
                memory_mask, memory_key_padding_mask,
                block_cache_3, cache_3_mha_key,
                block_cache_24, cache_24_mha_key,
                block_cache_rollout, cache_rollout_mha_key,
                layer_idx
            )
            
            # ===== FF Block =====
            x = CachePrunerWrapper._process_block(
                self, x, "ff", ff_strategy, is_not_batch, is_first_rollout, hard_gate,
                cur_step, local_ff_block_idx, memory, tgt_mask, tgt_key_padding_mask,
                memory_mask, memory_key_padding_mask,
                block_cache_3, cache_3_ff_key,
                block_cache_24, cache_24_ff_key,
                block_cache_rollout, cache_rollout_ff_key,
                layer_idx
            )

            return x

        layer.forward = types.MethodType(forward_with_cache, layer)

    @staticmethod
    def _process_block(
        layer, x, block_type, strategy, is_not_batch, is_first_rollout, hard_gate,
        cur_step, local_block_idx, memory, tgt_mask, tgt_key_padding_mask,
        memory_mask, memory_key_padding_mask,
        block_cache_3, cache_3_key,
        block_cache_24, cache_24_key,
        block_cache_rollout, cache_rollout_key,
        layer_idx
    ):
        """ blockSA/MHA/FF"""
        
        #  block  norm
        if block_type == "sa":
            compute_fn = lambda: layer._sa_block(layer.norm1(x), tgt_mask, tgt_key_padding_mask)
        elif block_type == "mha":
            compute_fn = lambda: layer._mha_block(layer.norm2(x), memory, memory_mask, memory_key_padding_mask)
        elif block_type == "ff":
            compute_fn = lambda: layer._ff_block(layer.norm3(x))
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        if is_not_batch:
            #  batch  strategy_dict
            if strategy == 0:
                # Compute
                output = compute_fn()
                x = x + output
                block_cache_3[cache_3_key] = output
                block_cache_24[cache_24_key] = output
                block_cache_rollout[cache_rollout_key] = output
            elif strategy == 1:
                # 3cache
                cached = block_cache_3.get(cache_3_key)
                if cached is None:
                    output = compute_fn()
                    x = x + output
                    block_cache_3[cache_3_key] = output
                else:
                    x = x + cached
            elif strategy == 2:
                # 24cache
                cached = block_cache_24.get(cache_24_key)
                if cached is None:
                    output = compute_fn()
                    x = x + output
                    block_cache_24[cache_24_key] = output
                else:
                    x = x + cached
            elif strategy == 3:
                # rollout cache
                cached = block_cache_rollout.get(cache_rollout_key)
                if cached is None:
                    output = compute_fn()
                    x = x + output
                    block_cache_rollout[cache_rollout_key] = output
                else:
                    x = x + cached
        else:
            # Batch  hard_gate
            if is_first_rollout:
                output = compute_fn()
                x = x + output
                block_cache_rollout[cache_rollout_key] = output.detach()
            else:
                w = hard_gate[:, cur_step, local_block_idx, :]
                g0, g1, g2, g3 = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
                g0 = g0.unsqueeze(-1)
                g1 = g1.unsqueeze(-1)
                g2 = g2.unsqueeze(-1)
                g3 = g3.unsqueeze(-1)
                
                output = compute_fn()
                cached_3 = block_cache_3.get(cache_3_key)
                cached_24 = block_cache_24.get(cache_24_key)
                cached_rollout = block_cache_rollout.get(cache_rollout_key)
                
                if cached_3 is None or cached_24 is None:
                    x = x + output
                    block_cache_3[cache_3_key] = output.detach()
                    block_cache_24[cache_24_key] = output.detach()
                else:
                    mix = output * g0 + cached_3 * g1 + cached_24 * g2 + cached_rollout * g3
                    x = x + mix
                    condition = g0.squeeze(-1).squeeze(-1).bool()
                    block_cache_3[cache_3_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), output.detach(), cached_3)
                    block_cache_24[cache_24_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), output.detach(), cached_24)
                    block_cache_rollout[cache_rollout_key] = torch.where(
                        condition.unsqueeze(-1).unsqueeze(-1), output.detach(), cached_rollout)
        
        return x
