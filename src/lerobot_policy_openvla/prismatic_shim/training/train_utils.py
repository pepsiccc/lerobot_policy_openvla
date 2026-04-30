"""
Minimal shim for prismatic.training.train_utils.
Only the two mask functions used by modeling_prismatic.py are implemented.
Source: https://github.com/moojink/openvla-oft/blob/main/prismatic/training/train_utils.py
"""
import torch
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    """Mask for the current action chunk tokens in the sequence."""
    newline_positions = (token_ids != IGNORE_INDEX).long()  # bool → int64
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask


def get_next_actions_mask(token_ids):
    """Mask for the next action chunk tokens in the sequence."""
    newline_positions = (token_ids != IGNORE_INDEX).long()  # bool → int64
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask
