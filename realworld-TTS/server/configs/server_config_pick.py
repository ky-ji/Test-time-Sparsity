"""
TTS Configpick / cogact_robot_7d
"""

# Server
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8006

# Model
CHECKPOINT_PATH = ""  # TODO: /abs/path/to/policy.ckpt
USE_EMA = True

# Inference
DEVICE = "cuda:2"
SCHEDULER_TYPE = "DDPM"
NUM_INFERENCE_STEPS = 100
INFERENCE_FREQ = 10.0

# Image
IMAGE_QUALITY = 85
IMAGE_RESIZE = True
MAX_IMAGE_SIZE = (1920, 1080)

# Communication
SOCKET_TIMEOUT = 5.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 1

# Logging
VERBOSE = True
LOG_LEVEL = "INFO"

# Action limit / smoothing
# / client /IK
ACTION_SCALE = 1.0
ACTION_SMOOTHING_ALPHA = 0.0
MAX_DELTA_POSITION = 0.04
MAX_DELTA_ROTATION = 0.1
ENABLE_ACTION_LIMIT = False

# TTS acceleration
ENABLE_TTS = True
PRUNER_CHECKPOINT = ""  # TODO: /abs/path/to/pruner.pt
TARGET_PRUNE_RATIO = 0.95
PRUNER_CONFIG = {
    "hidden_dim": 512,
    "encoder_type": "MLP",
    "encoder_layers": 1,
    "step_encoder": "sin",
    "block_encoder": "sin",
    "time_encoder_type": "SA",
    "use_obs_encoder": True,
    "train_obs_encoder": True,
    "init_obs_encoder": False,
    "use_obs_emb": True,
    "mlp_mode": "model_wise",
    "ca_ff_dim": 512,
    "sa_ratio": 1.0,
    "ca_ratio": 1.0,
    "ffn_ratio": 1.0,
}

TTS_ASYNC_PRUNER = True
TTS_USE_THREADING = True
TTS_CACHE_STRATEGY = "3cache+24cache"
TTS_ENABLE_TIMING_STATS = True
TTS_LOG_STATS_INTERVAL = 100
