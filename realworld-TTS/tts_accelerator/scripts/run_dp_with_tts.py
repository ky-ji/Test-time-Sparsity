#!/usr/bin/env python3
"""
 DP  TTS 


    python tts_accelerator/scripts/run_dp_with_tts.py --config tts_accelerator/configs/assembly_bun.yaml
    
    
    python tts_accelerator/scripts/run_dp_with_tts.py \
        --checkpoint /path/to/dp_policy.ckpt \
        --pruner /path/to/pruner.pt \
        --port 8007


1.  DP 
2.  TTS  pruner
3. 

 realworld-SAG/draw  session 
    python tts_accelerator/scripts/run_dp_with_tts.py ... --vis

 --vis 
- phaseBGR+  diffusion step×block
- realworld-TTS/output/sessions/session_YYYYMMDD_HHMMSS_data.npz
-  realworld-SAG/draw/visualize_session.py  mp4/gif
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# - this_file: realworld-TTS/tts_accelerator/scripts/run_dp_with_tts.py
# - pkg_root : realworld-TTS/tts_accelerator
# - repo_root: realworld-TTS
this_file = Path(__file__).resolve()
pkg_root = this_file.parent.parent
repo_root = pkg_root.parent

#  `import tts_accelerator`  cwd
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# TTSInfer  pruner/TransformerPruner  sys.path
realworld_sag_root = repo_root.parent
for maybe in (realworld_sag_root / "TTSInfer",):
    if maybe.exists() and str(maybe) not in sys.path:
        sys.path.insert(0, str(maybe))

def _configure_logging(verbose: bool = True) -> None:
    """
    Config logging tts_accelerator  logger.info 
    "//decoder"
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    #  basicConfig  root level
    logging.getLogger().setLevel(level)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DP Inference Server with TTS Acceleration")
    
    # Config
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, help="Path to DP policy checkpoint")
    parser.add_argument("--pruner", type=str, default=None, help="Path to TTS pruner checkpoint")

    #  +
    parser.add_argument("--vis", action="store_true", help="Enable pruning visualization (save session .npz and render mp4/gif on disconnect)")
    parser.add_argument("--vis-output-dir", type=str, default=None, help="Directory to save session data/visualizations (default: realworld-TTS/output/sessions)")
    parser.add_argument("--vis-save-only", action="store_true", help="Only save session .npz, do not render mp4/gif")
    
    # Config
    # default=None  YAML  YAML
    parser.add_argument("--ip", type=str, default=None, help="Server IP")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    # TTS Config
    parser.add_argument("--disable-tts", action="store_true", help="Disable TTS acceleration")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Target prune ratio")
    
    # Config
    parser.add_argument("--num-steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--scheduler", type=str, default=None, choices=["DDPM", "DDIM"])
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """ YAML Config"""
    try:
        import yaml
    except Exception as e:
        raise ImportError(
            " PyYAMLpip install pyyaml\n"
            f"{e}"
        )
    p = Path(config_path).expanduser()
    if not p.is_absolute() and not p.exists():
        #  configs/xxx.yaml
        # -  repo_root
        # -  pkg_root
        # -  pkg_root/configs/<basename>
        candidates = [
            (repo_root / p),
            (pkg_root / p),
            (pkg_root / "configs" / p.name),
        ]
        for c in candidates:
            if c.exists():
                p = c
                break

    with open(p, 'r') as f:
        return yaml.safe_load(f)

def _pick(config: dict, key: str, cli_value):
    """ CLI YAML None"""
    return cli_value if cli_value is not None else config.get(key)

def _find_dp_root(checkpoint_path: Optional[str], config: dict) -> Path:
    """
    Find project root (containing diffusion_policy submodule and realworld_deploy).

    Search order:
    - config['dp_root']
    - Walk up from checkpoint_path
    - Default location (project root)
    """
    candidates: List[Path] = []
    dp_root_cfg = config.get("dp_root")
    if dp_root_cfg:
        candidates.append(Path(dp_root_cfg).expanduser())

    if checkpoint_path:
        p = Path(checkpoint_path).expanduser().resolve()
        for parent in [p] + list(p.parents):
            if parent.is_file():
                continue
            candidates.append(parent)

    # Project root: realworld-TTS/../  i.e. Test-time-Sparsity/
    project_root = Path(__file__).parent.parent.parent.parent
    candidates.append(project_root)

    seen = set()
    for root in candidates:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen:
            continue
        seen.add(root)
        probe = root / "realworld_deploy" / "server" / "inference_server.py"
        if probe.exists():
            return root

    raise FileNotFoundError(
        "Cannot find realworld_deploy/server/inference_server.py\n"
        "Please specify dp_root: /abs/path/to/project_root in YAML"
    )

def _import_dp_inference_server(dp_root: Path):
    """
     `import realworld_deploy.server...`
    realworld_deploy/server/__init__.py  dp_inference_server.py
     inference_server.py __init__.py
    """
    import importlib.util

    dp_root_str = str(dp_root)
    if dp_root_str not in sys.path:
        sys.path.insert(0, dp_root_str)

    inference_server_path = dp_root / "realworld_deploy" / "server" / "inference_server.py"
    if not inference_server_path.exists():
        raise FileNotFoundError(f"Cannot find {inference_server_path}")

    spec = importlib.util.spec_from_file_location("dp_inference_server_module", inference_server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {inference_server_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DPInferenceServerSSH


def main():
    args = parse_args()
    
    # Config
    config = {}
    if args.config:
        config = load_config_from_yaml(args.config)

    # logging TTS decoder//
    _configure_logging(verbose=bool(config.get("verbose", True)))
    
    # Override config file
    checkpoint = args.checkpoint or config.get('checkpoint_path')
    pruner_path = args.pruner or config.get('pruner_checkpoint')
    enable_tts = bool(config.get("enable_tts", True)) and (not args.disable_tts)
    enable_vis = bool(config.get("vis", False)) or bool(args.vis)
    vis_save_only = bool(config.get("vis_save_only", False)) or bool(args.vis_save_only)
    vis_output_dir = _pick(config, "vis_output_dir", args.vis_output_dir)

    server_ip = _pick(config, "server_ip", args.ip) or "0.0.0.0"
    server_port = _pick(config, "server_port", args.port) or 8007
    device = _pick(config, "device", args.device) or "cuda:0"
    scheduler = (_pick(config, "scheduler", args.scheduler) or "DDPM").upper()
    num_steps = _pick(config, "num_inference_steps", args.num_steps) or 100
    prune_ratio = _pick(config, "target_prune_ratio", args.prune_ratio)
    
    if not checkpoint:
        print("Error: --checkpoint is required")
        sys.exit(1)
    
    print("=" * 60)
    print("Diffusion Policy Inference Server with TTS Acceleration")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    if enable_tts and pruner_path:
        print(f"  Pruner: {pruner_path}")
    elif pruner_path and not enable_tts:
        print(f"  Pruner: {pruner_path} (disabled)")
    else:
        print("  Pruner: None (no acceleration)")
    print(f"  Server: {server_ip}:{server_port}")
    print(f"  Device: {device}")
    print(f"  Scheduler: {scheduler}")
    print(f"  Inference steps: {num_steps}")
    if enable_vis:
        print(f"  Visualization: enabled (save_only={vis_save_only})")
    print("=" * 60)
    
    #  server_config
    import types
    server_config = types.ModuleType('server_config')
    
    server_config.SERVER_IP = server_ip
    server_config.SERVER_PORT = server_port
    server_config.CHECKPOINT_PATH = checkpoint
    server_config.USE_EMA = config.get('use_ema', True)
    server_config.DEVICE = device
    server_config.SCHEDULER_TYPE = scheduler
    server_config.NUM_INFERENCE_STEPS = int(num_steps)
    server_config.INFERENCE_FREQ = config.get('inference_freq', 10.0)
    server_config.SOCKET_TIMEOUT = 5.0
    server_config.BUFFER_SIZE = 4096
    server_config.ENCODING = "utf-8"
    server_config.MAX_CLIENTS = 1
    server_config.VERBOSE = True
    
    # Config
    # / client /IK
    server_config.ACTION_SCALE = config.get('action_scale', 1.0)
    server_config.ACTION_SMOOTHING_ALPHA = config.get('smoothing_alpha', 0.0)
    server_config.MAX_DELTA_POSITION = config.get('max_delta_position', 0.04)
    server_config.MAX_DELTA_ROTATION = config.get('max_delta_rotation', 0.1)
    server_config.ENABLE_ACTION_LIMIT = config.get('enable_action_limit', False)
    
    # TTS Config
    if enable_tts and pruner_path:
        server_config.ACCELERATOR = {
            'enabled': True,
            'type': 'tts',
            'pruner_checkpoint': pruner_path,
            'target_prune_ratio': prune_ratio if prune_ratio is not None else 0.93,
            'pruner_config': config.get('pruner_config', {
                'hidden_dim': 512,
                'decoder_layers': 1,
                'block_encoder': 'SA',
                'attn_heads': 8,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'reuse_dp_encoder': False,
                'reuse_block': False,
                'if_rollout_cache': True,
            }),
            'wrapper_kwargs': {
                'enable_sag': True,
                'training': False,
                'if_rollout_cache': True,
            }
        }
    else:
        server_config.ACCELERATOR = None
    
    # Config
    sys.modules['server_config'] = server_config
    
    #  realworld_deploy/server/__init__.py
    dp_root = _find_dp_root(checkpoint_path=checkpoint, config=config)
    DPInferenceServerSSH = _import_dp_inference_server(dp_root)
    
    class AcceleratedDPServer(DPInferenceServerSSH):
        """ TTS """
        
        def _load_model(self):
            """"""
            super()._load_model()
            
            accelerator_config = getattr(server_config, 'ACCELERATOR', None)
            if accelerator_config and accelerator_config.get('enabled'):
                self.policy = self._apply_accelerator(self.policy, accelerator_config)
        
        def _apply_accelerator(self, policy, config):
            """"""
            accelerator_type = config.get('type')
            
            if accelerator_type == 'tts':
                try:
                    from tts_accelerator import TTSPolicyWrapper, load_pruner
                    
                    print("\n" + "=" * 40)
                    print("Applying TTS Acceleration")
                    print("=" * 40)
                    
                    pruner = load_pruner(policy, config, device=self.device)
                    
                    if pruner is not None:
                        wrapper_kwargs = config.get('wrapper_kwargs', {})
                        accelerated = TTSPolicyWrapper(
                            policy=policy,
                            pruner=pruner,
                            **wrapper_kwargs
                        )
                        print("✓ TTS acceleration enabled")
                        print(f"  Target prune ratio: {config.get('target_prune_ratio', 0.93)}")
                        print("=" * 40 + "\n")
                        return accelerated
                    else:
                        print("✗ TTS pruner load failed, running without acceleration")
                        
                except ImportError as e:
                    print(f"Warning: TTS accelerator not available: {e}")
                    print("Running without acceleration")
            
            return policy

    class AcceleratedDPServerViz(AcceleratedDPServer):
        """
        AcceleratedDPServer + 

        
        - 
        -  client  session  mp4/gif
        - gate  [N, T, B, 4] 4
            - [:,:,:,0] compute
            - [:,:,:,1] 3cache
            - [:,:,:,2] 24cache
            - [:,:,:,3] rollout_cache
           realworld-SAG/draw/visualize_session.py 
        """

        def __init__(self, *a, **k):
            self._vis_enabled = True
            self._vis_save_only = vis_save_only
            self._vis_output_dir = Path(vis_output_dir) if vis_output_dir else (repo_root / "output" / "sessions")

            self._vis_inference_count = 0
            self._vis_total_infer_time = 0.0
            self._vis_session_start_time = None

            self._vis_frames: List[Any] = []
            self._vis_gates_4: List[Any] = []
            self._vis_block_names: List[str] = []

            super().__init__(*a, **k)

        # ---------- hook: client lifecycle ----------
        def _handle_client(self, client_socket, client_addr):
            from datetime import datetime
            self._vis_inference_count = 0
            self._vis_total_infer_time = 0.0
            self._vis_frames = []
            self._vis_gates_4 = []
            self._vis_block_names = []
            self._vis_session_start_time = datetime.now()

            try:
                return super()._handle_client(client_socket, client_addr)
            finally:
                # client /
                try:
                    data_path = self._save_session_data()
                    if (not self._vis_save_only) and data_path is not None:
                        self._render_session(data_path)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[vis] / session {e}")

        # ---------- hook: message processing ----------
        def _process_message(self, client_socket, message: str, recv_timestamp: float):
            """
            "BGR"

             RGB  resize 
            """
            import json
            import base64
            import numpy as np
            import cv2
            import time

            data = json.loads(message)

            if data.get("type") == "reset":
                #  reset policy / episode timers
                return super()._process_message(client_socket, message, recv_timestamp)

            if data.get("type") != "observation":
                return super()._process_message(client_socket, message, recv_timestamp)

            # Parse phase: BGR image
            process_start_time = time.time()

            images_b64 = data.get("images", [])
            poses_list = data.get("poses", [])
            grippers_list = data.get("grippers", [])
            client_timestamps = data.get("timestamps", [])
            client_send_timestamp = data.get("send_timestamp")

            message_interval = None
            if getattr(self, "last_recv_timestamp", None) is not None:
                message_interval = recv_timestamp - self.last_recv_timestamp
            self.last_recv_timestamp = recv_timestamp

            if client_send_timestamp is not None:
                transport_latency = recv_timestamp - client_send_timestamp
            else:
                transport_latency = message_interval if message_interval is not None else 0.0

            #  BGR RGB + resize
            images_rgb = []
            last_bgr = None
            for img_b64 in images_b64:
                img_data = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                last_bgr = bgr
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if self.expected_image_shape is not None:
                    expected_h, expected_w = self.expected_image_shape
                    if rgb.shape[:2] != (expected_h, expected_w):
                        rgb = cv2.resize(rgb, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                images_rgb.append(rgb)

            if last_bgr is not None:
                self._vis_frames.append(last_bgr)

            #  env_obs
            poses = np.array(poses_list, dtype=np.float32)
            grippers = np.array(grippers_list, dtype=np.float32)

            env_obs = {}
            if self.obs_keys and self.obs_keys.get("rgb"):
                env_obs[self.obs_keys["rgb"]] = np.stack(images_rgb, axis=0).astype(np.uint8)
            if self.obs_keys and self.obs_keys.get("lowdim"):
                for lowdim_key in self.obs_keys["lowdim"]:
                    if "pose" in lowdim_key.lower():
                        env_obs[lowdim_key] = poses
                    elif "gripper" in lowdim_key.lower():
                        env_obs[lowdim_key] = grippers

            last_image_b64 = images_b64[-1] if images_b64 else None

            action = self._infer_action(
                env_obs,
                np.array(client_timestamps, dtype=np.float32),
                recv_timestamp,
                client_send_timestamp,
                float(transport_latency),
                process_start_time,
                message_interval,
                last_image_b64,
            )

            response = {"type": "action", "action": action.tolist()}
            msg = json.dumps(response) + "\n"
            enc = getattr(server_config, "ENCODING", "utf-8")
            client_socket.sendall(msg.encode(enc))

        def _infer_action(
            self,
            env_obs: dict,
            timestamps,
            recv_timestamp: float,
            client_send_timestamp: Optional[float],
            transport_latency: float,
            process_start_time: float,
            message_interval: Optional[float] = None,
            raw_image_b64: Optional[str] = None,
        ):
            import time
            t0 = time.time()
            action = super()._infer_action(
                env_obs,
                timestamps,
                recv_timestamp,
                client_send_timestamp,
                transport_latency,
                process_start_time,
                message_interval,
                raw_image_b64,
            )
            infer_time = time.time() - t0
            self._vis_inference_count += 1
            self._vis_total_infer_time += float(infer_time)
            self._collect_gate_data()
            return action

        # ---------- vis helpers ----------
        def _unwrap_policy_for_cache(self):
            """
            
            - self.policy  TTSPolicyWrapper
            - self.policy Already the original policy (injected by CachePrunerWrapper)
            """
            p = getattr(self, "policy", None)
            if p is None:
                return None
            # TTSPolicyWrapper:  .policy  .original_policy
            inner = getattr(p, "policy", None) or getattr(p, "original_policy", None)
            return inner if inner is not None else p

        def _gate_from_cache(self, cache_ctx: Dict[str, Any], num_steps: int, num_blocks: int) -> Optional[Tuple[Any, Any]]:
            """
             cache_ctx  gate
            - gate4: [T, B, 4](compute, 3cache, 24cache, rollout) /one-hot
            - block_names: List[str]
            """
            import numpy as np

            def _is_tensor_like(x: Any) -> bool:
                #  torch torch stub
                return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "dim")

            block_names = []

            # block keys
            p = self._unwrap_policy_for_cache()
            if p is not None and hasattr(p, "_cache_block_keys"):
                block_names = list(getattr(p, "_cache_block_keys", []))

            # 1) soft_gate: [B, T, Bk, 4]
            soft_gate = cache_ctx.get("soft_gate", None)
            if soft_gate is not None and _is_tensor_like(soft_gate):
                sg = soft_gate.detach().cpu()
                if sg.dim() == 4:
                    sg0 = sg[0].numpy().astype(np.float32)  # [T, num_blocks, 4]
                    return sg0, block_names

            # 2) hard gate one-hot: [B, T, Bk, 4]
            hard_gate = cache_ctx.get("gate", None)
            if hard_gate is not None and _is_tensor_like(hard_gate):
                hg = hard_gate.detach().cpu()
                if hg.dim() == 4:
                    hg0 = hg[0].numpy().astype(np.float32)  # [T, num_blocks, 4]
                    return hg0, block_names

            # 3) strategy_dict: {global_block_idx: strategy_id}
            strategy_dict = cache_ctx.get("strategy_dict", None)
            if isinstance(strategy_dict, dict) and len(strategy_dict) > 0:
                gate4 = np.zeros((num_steps, num_blocks, 4), dtype=np.float32)
                #  compute
                gate4[:, :, 0] = 1.0
                for global_idx, sid in strategy_dict.items():
                    step = int(global_idx) // int(num_blocks)
                    blk = int(global_idx) % int(num_blocks)
                    if 0 <= step < num_steps and 0 <= blk < num_blocks:
                        s = int(sid)
                        if 0 <= s < 4:
                            gate4[step, blk, :] = 0.0
                            gate4[step, blk, s] = 1.0
                return gate4, block_names

            return None

        def _collect_gate_data(self) -> None:
            """ TTS cache  gate4 compute/3cache/24cache/rollout"""
            p = self._unwrap_policy_for_cache()
            if p is None or not hasattr(p, "_cache"):
                return
            cache_ctx = getattr(p, "_cache", None)
            if not isinstance(cache_ctx, dict):
                return

            num_steps = int(getattr(p, "num_inference_steps", 100))
            num_blocks = int(cache_ctx.get("num_blocks_per_step", 0) or 0)
            if num_blocks <= 0 and hasattr(p, "_cache_block_keys"):
                num_blocks = len(getattr(p, "_cache_block_keys", []))
            if num_blocks <= 0:
                return

            out = self._gate_from_cache(cache_ctx, num_steps=num_steps, num_blocks=num_blocks)
            if out is None:
                return
            gate4, block_names = out

            if (not self._vis_block_names) and block_names:
                self._vis_block_names = block_names

            self._vis_gates_4.append(gate4)

        def _save_session_data(self) -> Optional[Path]:
            import numpy as np
            if (not self._vis_frames) and (not self._vis_gates_4):
                return None

            self._vis_output_dir.mkdir(parents=True, exist_ok=True)
            ts = self._vis_session_start_time.strftime("%Y%m%d_%H%M%S") if self._vis_session_start_time else "unknown"
            session_name = f"session_{ts}"
            data_path = self._vis_output_dir / f"{session_name}_data.npz"

            frames_arr = np.array(self._vis_frames) if self._vis_frames else np.array([], dtype=np.uint8)
            gates_arr = np.array(self._vis_gates_4) if self._vis_gates_4 else None
            avg_time = (self._vis_total_infer_time / self._vis_inference_count) if self._vis_inference_count > 0 else 0.0

            np.savez(
                str(data_path),
                frames=frames_arr,
                gates=gates_arr,
                block_names=np.array(self._vis_block_names, dtype=object),
                timestamp=str(ts),
                num_inferences=int(self._vis_inference_count),
                avg_inference_time=float(avg_time),
            )

            logging.getLogger(__name__).info(f"[vis] session data saved: {data_path}")
            logging.getLogger(__name__).info(f"[vis]   frames={len(self._vis_frames)} gates={len(self._vis_gates_4)} blocks={len(self._vis_block_names)}")
            return data_path

        def _render_session(self, data_path: Path) -> None:
            """
             realworld-SAG/draw/visualize_session.py /GIF
             .npz
            """
            import importlib.util
            from types import SimpleNamespace

            draw_candidates = [
                realworld_sag_root / "realworld-SAG" / "draw" / "visualize_session.py",
                realworld_sag_root / "draw" / "visualize_session.py",
            ]
            draw_viz = next((p for p in draw_candidates if p.exists()), None)
            if draw_viz is None:
                logging.getLogger(__name__).warning(
                    "[vis]  draw "
                )
                return

            spec = importlib.util.spec_from_file_location("sag_draw_visualize_session", draw_viz.resolve())
            if spec is None or spec.loader is None:
                logging.getLogger(__name__).warning("[vis]  draw/visualize_session.py")
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            #  visualize_session.process_single_file  args
            args = SimpleNamespace(
                output_dir=str(self._vis_output_dir),
                all=False,
                combined_only=False,
                fps=10,
                gif_duration=0.2,
                style="vertical",
                figsize="12,8",
                dpi=200,
                tanh_factor=60.0,
            )

            if hasattr(mod, "process_single_file"):
                mod.process_single_file(Path(data_path), args)
                logging.getLogger(__name__).info(f"[vis] render complete: {self._vis_output_dir}")
            else:
                logging.getLogger(__name__).warning("[vis] visualize_session.py  process_single_file")
    
    print("\nStarting server...")
    server = AcceleratedDPServerViz() if enable_vis else AcceleratedDPServer()
    server.start()


if __name__ == "__main__":
    main()
