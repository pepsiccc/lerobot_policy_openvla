"""
OpenVLA-OFT 数据处理 Pipeline。

两个阶段：
  PreProcessor：LeRobot batch → 模型输入
    - 图像：(B, 3, H, W) → Prismatic processor → (B, 6, 224, 224)（SigLIP + DINOv2）
    - 语言：batch["task"] 字符串 → tokenized input_ids
    - labels：从 input_ids 生成（训练时用于 loss 计算）
    - 本体状态：mean-std 归一化（可选）

  PostProcessor：动作 tensor → 执行动作
    - 模型已在 predict_action() 内部完成反归一化，PostProcessor 通常是 identity
    - 保留接口供将来扩展（如动作平滑、clip 等）
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

CENTER_CROP_RATIO = 0.9
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
DEFAULT_TASK_DESCRIPTION = "complete the task"

# Llama-2 的 IGNORE_INDEX，用于生成 labels（非 action 位置不计算 loss）
IGNORE_INDEX = -100


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def center_crop_image(image: torch.Tensor, crop_ratio: float = CENTER_CROP_RATIO) -> torch.Tensor:
    """中心裁剪，保留 crop_ratio 面积后 resize 回原尺寸。"""
    _, H, W = image.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    cropped = TF.center_crop(image, [crop_h, crop_w])
    return TF.resize(cropped, [H, W], antialias=True)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """(C, H, W) float [0,1] tensor → PIL Image。"""
    image = image.float().clamp(0, 1)
    image = (image * 255).byte()
    return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# PreProcessor
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPreProcessor:
    """OpenVLA-OFT 输入预处理器。

    把 LeRobot DataLoader 输出的 batch 转换为模型可直接消费的格式：
      batch["observation.images.*"]  →  batch["observation.images.*"]（6通道）
      batch["task"]                  →  batch["input_ids"], batch["attention_mask"]
      batch["input_ids"]             →  batch["labels"]（训练用）
    """

    def __init__(self, config, dataset_stats: dict[str, Any] | None = None):
        self.config = config
        self.dataset_stats = dataset_stats or {}
        self._hf_processor: Any | None = None

    @property
    def hf_processor(self):
        """懒加载 Prismatic AutoProcessor。"""
        if self._hf_processor is None:
            self._hf_processor = AutoProcessor.from_pretrained(
                self.config.pretrained_backbone,
                trust_remote_code=True,
            )
        return self._hf_processor

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理 batch。

        Args:
            batch: LeRobot DataLoader 输出，包含：
                - "observation.images.<cam>": (B, 3, H, W) float [0,1]
                - "task": list[str] 任务描述
                - "observation.state": (B, 6) 本体状态（可选）
                - "action": (B, chunk_size, 6) 动作标签（训练时）

        Returns:
            处理后的 batch，新增/替换：
                - "observation.images.<cam>": (B, 6, H, W) 6通道 pixel_values
                - "input_ids": (B, seq_len)
                - "attention_mask": (B, seq_len)
                - "labels": (B, seq_len) IGNORE_INDEX 填充（训练用）
        """
        processed = dict(batch)

        # ── 1. 获取任务描述 ───────────────────────────────────────────────
        batch_size = None
        image_keys = list(self.config.image_keys)

        for key in image_keys:
            if key in batch:
                batch_size = batch[key].shape[0]
                break
        if batch_size is None:
            batch_size = 1

        raw_tasks = batch.get(self.config.task_key, None)
        if raw_tasks is None:
            task_descriptions = [DEFAULT_TASK_DESCRIPTION] * batch_size
        elif isinstance(raw_tasks, (list, tuple)):
            task_descriptions = [str(t) for t in raw_tasks]
        elif isinstance(raw_tasks, torch.Tensor):
            logger.warning(
                f"batch['{self.config.task_key}'] 是 tensor（task_index），"
                f"暂时使用默认任务描述。"
            )
            task_descriptions = [DEFAULT_TASK_DESCRIPTION] * batch_size
        else:
            task_descriptions = [str(raw_tasks)] * batch_size

        prompts = [PROMPT_TEMPLATE.format(task=t) for t in task_descriptions]

        # ── 2. 图像预处理：tensor → PIL → Prismatic processor → 6通道 ────
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []   # 每个元素是 (1, 6, H, W)

        for i in range(batch_size):
            # 收集当前样本的所有相机图像
            pil_images = []
            for key in image_keys:
                img = batch[key][i]   # (3, H, W)
                if self.config.center_crop:
                    img = center_crop_image(img, CENTER_CROP_RATIO)
                pil_images.append(tensor_to_pil(img))

            # Prismatic processor 处理
            # 多相机：逐张处理后在 channel 维度拼接 → (1, 6*N, H, W)
            # 模型内部用 torch.split(pixel_values, [6]*N, dim=1) 分离各相机
            if len(pil_images) == 1:
                inputs = self.hf_processor(
                    text=prompts[i],
                    images=pil_images[0],
                    return_tensors="pt",
                )
                pv = inputs["pixel_values"]   # (1, 6, H, W)
            else:
                # 逐相机处理，然后 cat 在 channel 维度
                pv_list = []
                for img in pil_images:
                    out = self.hf_processor(
                        text=prompts[i],
                        images=img,
                        return_tensors="pt",
                    )
                    pv_list.append(out["pixel_values"])   # (1, 6, H, W)
                    # 只取第一次的 input_ids（所有相机共用同一个 prompt）
                    if not all_input_ids or len(all_input_ids) <= i:
                        inputs = out
                pv = torch.cat(pv_list, dim=1)   # (1, 6*N, H, W)

            all_input_ids.append(inputs["input_ids"].squeeze(0))
            all_attention_masks.append(inputs["attention_mask"].squeeze(0))
            all_pixel_values.append(pv)   # (1, 6*N, H, W)

        # ── 3. Pad & stack ────────────────────────────────────────────────
        max_len = max(t.shape[0] for t in all_input_ids)
        processed["input_ids"] = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0)
            for t in all_input_ids
        ])
        processed["attention_mask"] = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0)
            for t in all_attention_masks
        ])

        # pixel_values：(B, 6, H, W)
        # 同时替换 batch 里的图像键，让 modeling 的 _collect_pixel_values 能拿到正确数据
        pixel_values = torch.cat(all_pixel_values, dim=0)   # (B, 6, H, W)
        if len(image_keys) == 1:
            processed[image_keys[0]] = pixel_values
        else:
            # 多相机时，把 6 通道按相机数量拆分存回各自的键
            # 注意：Prismatic 多图像时 processor 输出 (1, 6*N, H, W)
            # 这里直接把合并后的 pixel_values 存到第一个图像键，
            # modeling 的 _collect_pixel_values 会统一处理
            processed["pixel_values"] = pixel_values

        # ── 4. 生成 labels（训练用）──────────────────────────────────────────────
        # labels 格式：语言部分全为 IGNORE_INDEX，action token 部分为 ACTION_TOKEN_BEGIN_IDX+1
        # 最后一个 token 为 STOP_INDEX，与 _prepare_labels_for_action_prediction 对齐
        ACTION_TOKEN_BEGIN_IDX = 31743
        STOP_INDEX = 2
        ACTION_DIM = self.config.action_dim
        NUM_ACTIONS_CHUNK = self.config.action_chunk_size

        input_ids = processed["input_ids"]  # (B, seq_len)
        # 先全填 IGNORE_INDEX
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        # 在末尾追加 action token 数量的占位 label
        action_token_count = ACTION_DIM * NUM_ACTIONS_CHUNK + 1  # +1 for stop token
        action_labels = torch.full(
            (input_ids.shape[0], action_token_count),
            ACTION_TOKEN_BEGIN_IDX + 1,
            dtype=input_ids.dtype,
        )
        action_labels[:, -1] = STOP_INDEX  # 最后一个是 stop token
        labels = torch.cat([labels, action_labels], dim=1)

        # input_ids 也要相应扩展（与 _prepare_input_for_action_prediction 对齐）
        placeholder = torch.full(
            (input_ids.shape[0], action_token_count),
            ACTION_TOKEN_BEGIN_IDX + 1,
            dtype=input_ids.dtype,
        )
        placeholder[:, -1] = STOP_INDEX
        processed["input_ids"] = torch.cat([input_ids, placeholder], dim=1)
        # attention_mask 也扩展
        mask_extension = torch.ones(
            (input_ids.shape[0], action_token_count),
            dtype=processed["attention_mask"].dtype,
        )
        processed["attention_mask"] = torch.cat([processed["attention_mask"], mask_extension], dim=1)
        processed["labels"] = labels

        # ── 5. 本体状态归一化（可选）─────────────────────────────────────
        if self.config.use_proprio and "observation.state" in batch:
            processed["observation.state"] = self._normalize(
                batch["observation.state"], key="observation.state"
            )

        return processed

    def _normalize(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """mean-std 归一化。"""
        stats = self.dataset_stats.get(key)
        if stats is None:
            return tensor
        mean = torch.tensor(stats["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(stats["std"], dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean) / (std + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# PostProcessor
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPostProcessor:
    """OpenVLA-OFT 输出后处理器。

    模型的 predict_action() 已在内部完成反归一化，
    PostProcessor 目前是 identity，保留接口供将来扩展。
    """

    def __init__(self, config, dataset_stats: dict[str, Any] | None = None):
        self.config = config
        self.dataset_stats = dataset_stats or {}

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        """目前直接返回，模型已完成反归一化。"""
        return action


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot 插件入口
# ─────────────────────────────────────────────────────────────────────────────

def make_openvla_pre_post_processors(
    config,
    dataset_stats: dict[str, Any] | None = None,
) -> tuple[OpenVLAPreProcessor, OpenVLAPostProcessor]:
    """LeRobot policy factory 调用的入口函数。

    命名规范：make_{policy_name}_pre_post_processors
    """
    pre_processor = OpenVLAPreProcessor(config=config, dataset_stats=dataset_stats)
    post_processor = OpenVLAPostProcessor(config=config, dataset_stats=dataset_stats)
    return pre_processor, post_processor
