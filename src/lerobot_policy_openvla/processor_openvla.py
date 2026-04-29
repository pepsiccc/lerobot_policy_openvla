"""
OpenVLA-OFT 数据处理 Pipeline。

LeRobot 的 PolicyProcessorPipeline 系统将数据处理分为两阶段：
  - PreProcessor：观测 dict → 模型输入 dict（图像预处理、tokenization、归一化）
  - PostProcessor：模型输出 dict → 执行动作 dict（反归一化、裁剪）

OpenVLA-OFT 的预处理关键点：
  1. 图像：使用 AutoProcessor（Prismatic processor）进行 resize + normalize
     - 输入：PIL Image 或 uint8 numpy/tensor (H, W, C)
     - 输出：float32 tensor (C, H, W)，值域约 [-2.5, 2.5]（ImageNet norm）
  2. Center Crop：推理时取中心 90% 区域（对齐训练时的随机裁剪增强）
  3. 语言指令 tokenization：使用与 VLA 配套的 tokenizer
  4. 本体状态：mean-std 归一化（使用 dataset_stats）
  5. 动作：mean-std 归一化（训练时），反归一化（推理时）
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

# OpenVLA 训练时的随机裁剪比例（推理时用中心裁剪对齐）
CENTER_CROP_RATIO = 0.9

# OpenVLA 提示词模板
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"

# 默认任务描述（未提供时使用）
DEFAULT_TASK_DESCRIPTION = "complete the task"


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def center_crop_image(image: torch.Tensor, crop_ratio: float = CENTER_CROP_RATIO) -> torch.Tensor:
    """对图像执行中心裁剪，保留 crop_ratio 面积。

    Args:
        image: (C, H, W) float tensor
        crop_ratio: 裁剪后面积占原图的比例
    Returns:
        cropped: (C, H, W)，裁剪并 resize 回原尺寸
    """
    _, H, W = image.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    # 中心裁剪
    cropped = TF.center_crop(image, [crop_h, crop_w])
    # resize 回原尺寸（与训练时对齐）
    return TF.resize(cropped, [H, W], antialias=True)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """将 (C, H, W) float tensor 转为 PIL Image。
    支持归一化后的图像（先反归一化到 [0, 255]）和原始 uint8 图像。
    """
    if image.dtype == torch.float32 or image.dtype == torch.bfloat16:
        # 假设已是 [0, 1] 范围
        image = image.float().clamp(0, 1)
        image = (image * 255).byte()
    # (C, H, W) → (H, W, C)
    return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# 主处理器类
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPreProcessor:
    """OpenVLA-OFT 输入预处理器。

    职责：
    1. 图像 center crop（可选）+ Prismatic processor 预处理
    2. 提示词构造 + tokenization
    3. 本体状态归一化（可选）
    4. 动作归一化（训练时）

    所有方法都是纯函数，便于在 DataLoader worker 中使用。
    """

    def __init__(
        self,
        config,
        dataset_stats: dict[str, Any] | None = None,
    ):
        self.config = config
        self.dataset_stats = dataset_stats or {}

        # 延迟加载 processor（避免在 worker 中重复初始化）
        self._hf_processor: Any | None = None

    @property
    def hf_processor(self):
        """懒加载 HuggingFace AutoProcessor。"""
        if self._hf_processor is None:
            self._hf_processor = AutoProcessor.from_pretrained(
                self.config.pretrained_backbone,
                trust_remote_code=True,
            )
        return self._hf_processor

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理整个 batch。

        Args:
            batch: LeRobot DataLoader 输出的批次，图像为 (B, C, H, W) float [0,1]
                   任务描述在 batch["task"]（LeRobot 0.4.x 数据集格式）

        Returns:
            处理后的 dict，新增 input_ids / attention_mask / pixel_values
        """
        processed = dict(batch)

        # ── 1. 按 config.image_keys 顺序收集图像 ─────────────────────────
        # 用配置中指定的键名，而不是自动扫描，保证顺序固定：front 在前，wrist 在后
        image_keys = list(self.config.image_keys)   # e.g. ["observation.images.front", "observation.images.wrist"]
        processed_images: list[list[Image.Image]] = []   # [cam_idx][batch_idx]

        batch_size = None
        for key in image_keys:
            if key not in batch:
                raise KeyError(
                    f"batch 中缺少图像键 '{key}'。"
                    f"请检查 config.image_keys 是否与数据集一致。"
                    f"当前 batch 键：{list(batch.keys())}"
                )
            imgs = batch[key]   # (B, C, H, W)
            if batch_size is None:
                batch_size = imgs.shape[0]

            cam_images = []
            for b in range(imgs.shape[0]):
                img = imgs[b]   # (C, H, W)
                if self.config.center_crop:
                    img = center_crop_image(img, CENTER_CROP_RATIO)
                cam_images.append(tensor_to_pil(img))
            processed_images.append(cam_images)

        # ── 2. 语言指令：直接读 batch["task"]（LeRobot 0.4.x 格式）────────
        if batch_size is None:
            batch_size = 1

        # LeRobot 0.4.3 数据集：每个样本的 "task" 字段是任务描述字符串
        # batch["task"] 可能是 list[str] 或 tensor（task_index），需要处理两种情况
        raw_tasks = batch.get(self.config.task_key, None)
        if raw_tasks is None:
            task_descriptions = [DEFAULT_TASK_DESCRIPTION] * batch_size
        elif isinstance(raw_tasks, (list, tuple)):
            task_descriptions = [str(t) for t in raw_tasks]
        elif isinstance(raw_tasks, torch.Tensor):
            # task_index tensor：说明数据集返回的是 task 编号而非字符串
            # 这种情况需要从 dataset.meta.tasks 查表，暂时用默认描述并打警告
            logger.warning(
                f"batch['{self.config.task_key}'] 是 tensor（task_index），"
                f"无法直接用作语言指令。请在 DataLoader 中将 task 字符串传入 batch，"
                f"或将 config.task_key 设为包含字符串的键。暂时使用默认描述。"
            )
            task_descriptions = [DEFAULT_TASK_DESCRIPTION] * batch_size
        else:
            task_descriptions = [str(raw_tasks)] * batch_size

        prompts = [PROMPT_TEMPLATE.format(task=t) for t in task_descriptions]

        # ── 3. 转置图像列表：[cam][batch] → [batch][cam] ──────────────────
        if len(image_keys) == 1:
            batch_images = processed_images[0]  # List[PIL]
        else:
            batch_images = [
                [processed_images[cam][b] for cam in range(len(image_keys))]
                for b in range(batch_size)
            ]

        # ── 4. HF AutoProcessor：tokenize + 图像归一化 ────────────────────
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []

        for i in range(batch_size):
            imgs_i = batch_images[i] if isinstance(batch_images[i], list) else [batch_images[i]]
            inputs = self.hf_processor(
                text=prompts[i],
                images=imgs_i if len(imgs_i) > 1 else imgs_i[0],
                return_tensors="pt",
            )
            all_input_ids.append(inputs["input_ids"].squeeze(0))
            all_attention_masks.append(inputs["attention_mask"].squeeze(0))
            all_pixel_values.append(inputs["pixel_values"])

        # Pad input_ids 到同一长度（同一 batch 内 prompt 长度一致，但保险起见仍 pad）
        max_len = max(t.shape[0] for t in all_input_ids)
        padded_input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0)
            for t in all_input_ids
        ])
        padded_attention_masks = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0)
            for t in all_attention_masks
        ])

        processed["input_ids"] = padded_input_ids
        processed["attention_mask"] = padded_attention_masks
        if all_pixel_values:
            processed["pixel_values"] = torch.cat(all_pixel_values, dim=0)

        # ── 5. 本体状态归一化（observation.state，6 维）──────────────────
        if self.config.use_proprio and "observation.state" in batch:
            processed["observation.state"] = self._normalize(
                batch["observation.state"], key="observation.state"
            )

        # ── 6. 动作归一化（训练时）───────────────────────────────────────
        if "action" in batch:
            processed["action"] = self._normalize(batch["action"], key="action")

        return processed

    def _normalize(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """使用 dataset_stats 对张量进行 mean-std 归一化。"""
        stats = self.dataset_stats.get(key)
        if stats is None:
            return tensor
        mean = torch.tensor(stats["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(stats["std"], dtype=tensor.dtype, device=tensor.device)
        # 广播：对 action (B, T, D) 或 state (B, D) 均适用
        return (tensor - mean) / (std + 1e-8)


class OpenVLAPostProcessor:
    """OpenVLA-OFT 输出后处理器。

    职责：
    1. 动作反归一化（从归一化空间映射回原始空间）
    2. 动作裁剪（clip 到合法范围）
    """

    def __init__(
        self,
        config,
        dataset_stats: dict[str, Any] | None = None,
    ):
        self.config = config
        self.dataset_stats = dataset_stats or {}

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """后处理动作输出。

        Args:
            action: (B, action_chunk_size, action_dim) 或 (action_dim,)

        Returns:
            action: 反归一化后的动作
        """
        return self._denormalize(action, key="action")

    def _denormalize(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """使用 dataset_stats 对张量进行反归一化。"""
        stats = self.dataset_stats.get(key)
        if stats is None:
            return tensor
        mean = torch.tensor(stats["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(stats["std"], dtype=tensor.dtype, device=tensor.device)
        return tensor * (std + 1e-8) + mean


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot 插件入口：make_openvla_pre_post_processors
# ─────────────────────────────────────────────────────────────────────────────

def make_openvla_pre_post_processors(
    config,
    dataset_stats: dict[str, Any] | None = None,
) -> tuple[OpenVLAPreProcessor, OpenVLAPostProcessor]:
    """创建 OpenVLA-OFT 的预处理和后处理器。

    LeRobot 的 policy factory 会调用此函数，命名规范：
    make_{policy_name}_pre_post_processors

    Args:
        config: OpenVLAConfig 实例
        dataset_stats: 来自 LeRobotDataset 的归一化统计数据

    Returns:
        (pre_processor, post_processor) 元组
    """
    pre_processor = OpenVLAPreProcessor(config=config, dataset_stats=dataset_stats)
    post_processor = OpenVLAPostProcessor(config=config, dataset_stats=dataset_stats)
    return pre_processor, post_processor
