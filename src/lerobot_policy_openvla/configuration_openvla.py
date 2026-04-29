"""
OpenVLA-OFT Policy Configuration for LeRobot.

OpenVLA-OFT (Optimized Fine-Tuning) 相比原版 OpenVLA 的关键改进：
  - 并行解码 (parallel decoding) + Action Chunking → 推理速度提升 25-50x
  - 连续动作输出（MLP action head，L1 regression）而非离散 token
  - 支持多图像输入（第三人称 + 腕部相机）
  - 支持本体感知 (proprioception) 输入
  - 支持 FiLM conditioning（用于 ALOHA 等双臂任务）

参考：https://openvla-oft.github.io/
代码：https://github.com/moojink/openvla-oft
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    """OpenVLA-OFT Policy 配置。

    Args:
        # ── 模型基础 ──────────────────────────────────────────────────────────
        pretrained_backbone: HuggingFace Hub 上的 OpenVLA-OFT 基础模型 ID。
            默认使用官方 7B 基础模型（未针对特定任务微调）。
            微调后的检查点示例：
              - "moojink/openvla-7b-oft-finetuned-libero-spatial"
              - "moojink/openvla-7b-oft-finetuned-libero-goal"

        # ── Action Head（动作头）──────────────────────────────────────────────
        action_dim: 动作空间维度（如 7 表示 6-DOF + gripper，14 表示双臂）。
        action_chunk_size: 单次前向推理输出的动作步数（action chunking）。
            OpenVLA-OFT 默认为 8（LIBERO）或 25（ALOHA）。
        action_head_hidden_dim: MLP action head 的隐藏层维度。
        action_head_num_hidden_layers: MLP action head 的隐藏层数。
        use_l1_regression: 使用 L1 回归损失训练 action head（推荐，默认 True）。
            若为 False 则使用 MSE 损失。
        use_diffusion: 使用扩散模型作为 action head（实验性，通常用 L1）。

        # ── 输入配置 ──────────────────────────────────────────────────────────
        num_images_in_input: 输入图像数量（1=仅第三人称，2=第三人称+腕部，3=双臂3相机）。
        image_size: 图像 resize 目标尺寸（高, 宽），OpenVLA 训练用 224x224。
        center_crop: 推理时是否使用中心裁剪（90% 面积）。
            训练时使用随机裁剪，推理时用中心裁剪以对齐分布，默认 True。
        use_proprio: 是否使用本体感知 (proprioception) 作为额外输入。
        proprio_dim: 本体感知状态维度（开启 use_proprio 时有效）。
        proprio_projector_hidden_dim: proprio projector MLP 的隐藏维度。

        # ── FiLM Conditioning ─────────────────────────────────────────────────
        use_film: 是否使用 FiLM（Feature-wise Linear Modulation）条件化。
            适用于双臂任务（如 ALOHA），可以更好地利用语言指令。
            单臂任务（如 LIBERO）通常不需要。

        # ── 量化 & 精度 ───────────────────────────────────────────────────────
        load_in_8bit: 使用 bitsandbytes 8-bit 量化加载 LLM 骨干。
        load_in_4bit: 使用 bitsandbytes 4-bit 量化加载 LLM 骨干（更激进的压缩）。
        torch_dtype: 模型权重精度，推荐 "bfloat16"。

        # ── LoRA 微调 ─────────────────────────────────────────────────────────
        use_lora: 是否启用 LoRA 适配器（训练时使用）。
        lora_rank: LoRA 秩（rank），通常 32~128。
        lora_alpha: LoRA 缩放因子，通常设为 lora_rank 的 2 倍。
        lora_dropout: LoRA dropout 概率。
        lora_target_modules: 需要注入 LoRA 的模块名（None 则自动选择注意力层）。

        # ── 归一化 ────────────────────────────────────────────────────────────
        unnorm_key: 动作反归一化所用的数据集统计键名（来自 openvla-oft 原始代码）。
            如果使用 LeRobot 的 dataset_stats 归一化则可忽略此字段。
        normalization_mapping: LeRobot 标准归一化模式映射。

        # ── 推理优化 ──────────────────────────────────────────────────────────
        num_open_loop_steps: 每次推理后在执行阶段复用相同 action chunk 的步数。
            通常等于 action_chunk_size（全开环）或更小（半开环）。
        n_action_steps: LeRobot 推理循环中每次调用 select_action 后执行的步数。
    """

    # ── 模型基础 ──────────────────────────────────────────────────────────────
    pretrained_backbone: str = "openvla/openvla-7b"
    """HuggingFace Hub 上的 OpenVLA-OFT 基础/微调模型 ID。"""

    # ── Action Head ───────────────────────────────────────────────────────────
    action_dim: int = 6                  # SO-101：6 个关节（含夹爪），改成你机器人的实际值
    action_chunk_size: int = 8
    action_head_hidden_dim: int = 1024
    action_head_num_hidden_layers: int = 2
    use_l1_regression: bool = True
    use_diffusion: bool = False

    # ── 输入配置 ──────────────────────────────────────────────────────────────
    num_images_in_input: int = 2         # SO-101 默认：front + wrist 两路相机
    image_size: tuple[int, int] = (224, 224)
    center_crop: bool = True
    use_proprio: bool = True             # SO-101 默认启用，有 observation.state
    proprio_dim: int = 6                 # SO-101：与 action_dim 相同，6 维
    proprio_projector_hidden_dim: int = 512

    # ── 相机键名（与 LeRobotDataset 的 feature 键名对应）─────────────────────
    # SO-101 数据集的实际键名，顺序决定多图像拼接顺序
    image_keys: tuple[str, ...] = ("observation.images.front", "observation.images.wrist")

    # ── 语言指令键名 ──────────────────────────────────────────────────────────
    # LeRobot 0.4.3 数据集中任务描述的键名（batch 里直接有 "task" 字段）
    task_key: str = "task"

    # ── FiLM Conditioning ─────────────────────────────────────────────────────
    use_film: bool = False

    # ── 量化 & 精度 ───────────────────────────────────────────────────────────
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # ── LoRA 微调 ─────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] | None = None

    # ── 归一化 ────────────────────────────────────────────────────────────────
    unnorm_key: str | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "observation.images.*": NormalizationMode.IDENTITY,  # 图像不额外归一化（processor 内部处理）
            "observation.state": NormalizationMode.MEAN_STD,
            "action": NormalizationMode.MEAN_STD,
        }
    )

    # ── 推理优化 ──────────────────────────────────────────────────────────────
    num_open_loop_steps: int = 8
    n_action_steps: int = 8

    def __post_init__(self):
        super().__post_init__()

        # n_action_steps 不能超过 action_chunk_size
        if self.n_action_steps > self.action_chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) 不能超过 "
                f"action_chunk_size ({self.action_chunk_size})。"
            )

        # 量化选项互斥
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit 和 load_in_4bit 不能同时为 True。")

        # 量化时不支持 LoRA 训练（仅推理）
        if self.use_lora and (self.load_in_8bit or self.load_in_4bit):
            raise ValueError(
                "量化模式（8bit/4bit）下不支持 LoRA 训练，仅用于推理。"
                "如需 LoRA 微调，请关闭量化选项。"
            )

        # diffusion 和 l1 互斥
        if self.use_diffusion and self.use_l1_regression:
            raise ValueError("use_diffusion 和 use_l1_regression 不能同时为 True。")

    def validate_features(self) -> None:
        """校验输入/输出 feature 的兼容性。

        LeRobot 训练启动时自动调用，提前发现配置与数据集不匹配的问题。
        基于 SO-101 数据集的实际键名进行校验：
          - observation.images.front  (3, H, W)
          - observation.images.wrist  (3, H, W)
          - observation.state         (6,)
          - action                    (6,)
          - task                      str（语言指令）
        """
        # ── 1. 校验 image_keys 与 num_images_in_input 一致 ────────────────
        if len(self.image_keys) != self.num_images_in_input:
            raise ValueError(
                f"image_keys 有 {len(self.image_keys)} 个键 {self.image_keys}，"
                f"但 num_images_in_input={self.num_images_in_input}，两者必须相等。"
            )

        # ── 2. 校验 image_keys 在 input_features 中都存在 ─────────────────
        if self.input_features:
            for key in self.image_keys:
                if key not in self.input_features:
                    raise ValueError(
                        f"配置的图像键 '{key}' 不在数据集 input_features 中。"
                        f"数据集图像键：{[k for k in self.input_features if 'image' in k]}"
                    )

        # ── 3. 校验 proprio_dim 与 observation.state 实际维度一致 ──────────
        if self.use_proprio and self.input_features:
            state_feature = self.input_features.get("observation.state")
            if state_feature is not None and hasattr(state_feature, "shape"):
                actual_dim = state_feature.shape[-1]
                if actual_dim != self.proprio_dim:
                    raise ValueError(
                        f"proprio_dim={self.proprio_dim} 与数据集 observation.state "
                        f"实际维度 {actual_dim} 不符。请将 proprio_dim 设为 {actual_dim}。"
                    )

        # ── 4. 校验 action_dim 与数据集 action 实际维度一致 ───────────────
        if self.output_features:
            action_feature = self.output_features.get("action")
            if action_feature is not None and hasattr(action_feature, "shape"):
                # LeRobot 单步 action shape 是 (action_dim,)
                # action chunking 时 DataLoader 会按 delta_timestamps 堆叠成 (chunk, dim)
                actual_dim = action_feature.shape[-1]
                if actual_dim != self.action_dim:
                    raise ValueError(
                        f"action_dim={self.action_dim} 与数据集 action "
                        f"实际维度 {actual_dim} 不符。请将 action_dim 设为 {actual_dim}。"
                    )
