"""
OpenVLA-OFT Policy Configuration for LeRobot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig, LRSchedulerConfig


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    """OpenVLA-OFT Policy 配置。"""

    # ── 模型基础 ──────────────────────────────────────────────────────────────
    pretrained_backbone: str = "openvla/openvla-7b"

    # ── Action Head ───────────────────────────────────────────────────────────
    action_dim: int = 6
    action_chunk_size: int = 8
    action_head_hidden_dim: int = 1024
    action_head_num_hidden_layers: int = 2
    use_l1_regression: bool = True
    use_diffusion: bool = False

    # ── 输入配置 ──────────────────────────────────────────────────────────────
    num_images_in_input: int = 2
    image_size: tuple[int, int] = (224, 224)
    center_crop: bool = True
    use_proprio: bool = True
    proprio_dim: int = 6
    proprio_projector_hidden_dim: int = 512

    # ── 相机键名 ──────────────────────────────────────────────────────────────
    image_keys: tuple[str, ...] = ("observation.images.front", "observation.images.wrist")

    # ── 语言指令键名 ──────────────────────────────────────────────────────────
    task_key: str = "task"

    # ── FiLM Conditioning ─────────────────────────────────────────────────────
    use_film: bool = False

    # ── 量化 & 精度 ───────────────────────────────────────────────────────────
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"

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
            "observation.images.*": NormalizationMode.IDENTITY,
            "observation.state": NormalizationMode.MEAN_STD,
            "action": NormalizationMode.MEAN_STD,
        }
    )

    # ── 推理优化 ──────────────────────────────────────────────────────────────
    num_open_loop_steps: int = 8
    n_action_steps: int = 8

    # ── Optimizer 超参数 ──────────────────────────────────────────────────────
    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8

    # ── Scheduler 超参数 ──────────────────────────────────────────────────────
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.action_chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) 不能超过 "
                f"action_chunk_size ({self.action_chunk_size})。"
            )
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit 和 load_in_4bit 不能同时为 True。")
        if self.use_lora and (self.load_in_8bit or self.load_in_4bit):
            raise ValueError("量化模式下不支持 LoRA 训练，仅用于推理。")
        if self.use_diffusion and self.use_l1_regression:
            raise ValueError("use_diffusion 和 use_l1_regression 不能同时为 True。")

    # ── 抽象属性实现 ──────────────────────────────────────────────────────────

    @property
    def observation_delta_indices(self) -> list | None:
        """当前帧观测，VLA 单帧即可决策。"""
        return [0]

    @property
    def action_delta_indices(self) -> list | None:
        """预测 action_chunk_size 步未来动作。
        chunk_size=8 → [0,1,2,3,4,5,6,7]
        """
        return list(range(self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        """纯模仿学习，不使用 reward。"""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        """AdamW，小学习率防止遗忘 VLA 预训练知识。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Cosine 衰减 + warmup。"""
        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """校验 config 与数据集 feature 的兼容性。"""
        if len(self.image_keys) != self.num_images_in_input:
            raise ValueError(
                f"image_keys 有 {len(self.image_keys)} 个键 {self.image_keys}，"
                f"但 num_images_in_input={self.num_images_in_input}，两者必须相等。"
            )
        if self.input_features:
            for key in self.image_keys:
                if key not in self.input_features:
                    raise ValueError(
                        f"配置的图像键 '{key}' 不在数据集 input_features 中。"
                        f"数据集图像键：{[k for k in self.input_features if 'image' in k]}"
                    )
        if self.use_proprio and self.input_features:
            state_feature = self.input_features.get("observation.state")
            if state_feature is not None and hasattr(state_feature, "shape"):
                actual_dim = state_feature.shape[-1]
                if actual_dim != self.proprio_dim:
                    raise ValueError(
                        f"proprio_dim={self.proprio_dim} 与数据集 observation.state "
                        f"实际维度 {actual_dim} 不符。请将 proprio_dim 设为 {actual_dim}。"
                    )
        if self.output_features:
            action_feature = self.output_features.get("action")
            if action_feature is not None and hasattr(action_feature, "shape"):
                actual_dim = action_feature.shape[-1]
                if actual_dim != self.action_dim:
                    raise ValueError(
                        f"action_dim={self.action_dim} 与数据集 action "
                        f"实际维度 {actual_dim} 不符。请将 action_dim 设为 {actual_dim}。"
                    )
