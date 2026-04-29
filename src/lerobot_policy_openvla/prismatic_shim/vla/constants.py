"""
Minimal shim for prismatic.vla.constants.
Only the constants used by modeling_prismatic.py are provided.
Source: https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py

We fix the platform to LIBERO since that is what the downloaded model
(openvla-7b-oft-finetuned-libero-10) was trained on. Change ROBOT_PLATFORM
if you are using a different model checkpoint.
"""
from enum import Enum


class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


# Platform constants
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Llama-2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'

# Fixed to LIBERO for openvla-7b-oft-finetuned-libero-10.
# Change this to ALOHA_CONSTANTS or BRIDGE_CONSTANTS for other checkpoints.
_constants = LIBERO_CONSTANTS

NUM_ACTIONS_CHUNK = _constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = _constants["ACTION_DIM"]
PROPRIO_DIM = _constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = _constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]
