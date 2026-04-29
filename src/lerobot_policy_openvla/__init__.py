"""
lerobot_policy_openvla — OpenVLA-OFT Policy Plugin for LeRobot.

将 OpenVLA-OFT（Fine-Tuning Vision-Language-Action Models: Optimizing Speed
and Success）集成为 LeRobot 的标准 policy 插件。

主要特性：
- 连续动作输出（MLP action head，L1 回归），无需离散化
- Action Chunking：单次推理预测多步动作，大幅降低延迟
- 多相机支持：第三人称 + 腕部相机（最多 3 路）
- 本体感知（proprioception）输入支持
- LoRA 微调支持（通过 peft）
- 4-bit/8-bit 量化推理支持（通过 bitsandbytes）

快速开始（推理）：
    from lerobot_policy_openvla import OpenVLAPolicy, OpenVLAConfig

    policy = OpenVLAPolicy.from_pretrained(
        "your-hf-repo/openvla-oft-finetuned"
    )
    action = policy.select_action(observation)

快速开始（训练）：
    lerobot-train \\
        --policy.type openvla \\
        --policy.pretrained_backbone openvla/openvla-7b \\
        --policy.use_lora true \\
        --policy.lora_rank 32 \\
        --policy.action_dim 7 \\
        --policy.action_chunk_size 8 \\
        --dataset.repo_id your/lerobot-dataset

参考：
- 论文：https://openvla-oft.github.io/
- 原始代码：https://github.com/moojink/openvla-oft
- LeRobot 插件文档：https://huggingface.co/docs/lerobot/en/bring_your_own_policies
"""

# 检查 lerobot 是否已安装
try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot 未安装。请先安装 lerobot：\n"
        "  pip install lerobot\n"
        "然后再安装本插件：\n"
        "  pip install lerobot_policy_openvla"
    )

from .configuration_openvla import OpenVLAConfig
from .modeling_openvla import MLPActionHead, OpenVLAPolicy, ProprioProjector
from .processor_openvla import (
    OpenVLAPostProcessor,
    OpenVLAPreProcessor,
    make_openvla_pre_post_processors,
)

__all__ = [
    # 配置
    "OpenVLAConfig",
    # Policy（主类）
    "OpenVLAPolicy",
    # 子模块（供高级用户直接使用）
    "MLPActionHead",
    "ProprioProjector",
    # 处理器
    "OpenVLAPreProcessor",
    "OpenVLAPostProcessor",
    "make_openvla_pre_post_processors",
]

__version__ = "0.1.0"
