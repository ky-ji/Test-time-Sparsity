"""Local TTS server config template."""

# Server
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8007

# Model
CHECKPOINT_PATH = ""  # Set to /abs/path/to/policy.ckpt
USE_EMA = True

# Inference
DEVICE = "cuda:0"
SCHEDULER_TYPE = "DDPM"  # "DDIM" or "DDPM"
NUM_INFERENCE_STEPS = 100
INFERENCE_FREQ = 10.0

# Communication
SOCKET_TIMEOUT = 5.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 1

# Logging
VERBOSE = True
LOG_LEVEL = "INFO"

# Action limit / smoothing (ref: Diffusion Policy original implementation)
# Client-side slerp interpolation and IK clamping
# Diffusion Policy _limit_and_smooth_action applies 3-step smoothing/clamping
#  7D pose (xyz+quat) /
#  action
ACTION_SCALE = 1.0
ACTION_SMOOTHING_ALPHA = 0.0
MAX_DELTA_POSITION = 0.04
MAX_DELTA_ROTATION = 0.1
ENABLE_ACTION_LIMIT = False

# TTS acceleration
ENABLE_TTS = True
PRUNER_CHECKPOINT = ""  # Set to /abs/path/to/pruner.pt
TARGET_PRUNE_RATIO = 0.93
PRUNER_CONFIG = {
    "hidden_dim": 512,
    "decoder_layers": 1,
    "block_encoder": "SA",
    "attn_heads": 8,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "reuse_dp_encoder": False,
    "reuse_block": False,
    "if_rollout_cache": True,
}

TTS_ASYNC_PRUNER = True
TTS_USE_THREADING = True
TTS_CACHE_STRATEGY = "Omini"
TTS_ENABLE_TIMING_STATS = True
TTS_LOG_STATS_INTERVAL = 100

