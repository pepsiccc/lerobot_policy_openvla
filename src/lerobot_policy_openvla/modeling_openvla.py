"""
OpenVLA-OFT Policy — LeRobot 集成实现。

架构概览（OpenVLA-OFT）：
┌─────────────────────────────────────────────────────────────────────┐
│  输入                                                                 │
│  ├── 图像 x N (224x224, RGB)                                         │
│  ├── 语言指令 (tokenized)                                             │
│  └── 本体状态 (可选)                                                  │
│                          ↓                                            │
│         OpenVLAForActionPrediction.predict_action()                  │
│         (Prismatic backbone + L1 regression action head)             │
│                          ↓                                            │
│  输出：连续动作 chunk (action_chunk_size × action_dim)                │
└─────────────────────────────────────────────────────────────────────┘

设计决策：
- 直接调用 OpenVLAForActionPrediction.predict_action()，不另建 MLPActionHead
- 模型自带完整的动作预测、归一化/反归一化逻辑
- 通过 importlib 绕过 Auto 注册表加载 Prismatic 自定义模型类
- prismatic_shim 包替代完整的 openvla-oft 仓库依赖（无 TensorFlow）
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import types
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_openvla import OpenVLAConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具：加载 Prismatic 模型类
# ─────────────────────────────────────────────────────────────────────────────

def _load_prismatic_model_class(model_path: str):
    """从本地模型目录加载 OpenVLAForActionPrediction 类。

    Prismatic 使用自定义架构，config.json 的 auto_map 中注册的是
    "AutoModelForVision2Seq"，但 transformers 5.x 已删除该类名。
    这里绕过 Auto 注册表，直接用 importlib 加载 modeling_prismatic.py。

    同时注入 prismatic_shim，替代完整的 openvla-oft 仓库（避免 TensorFlow 依赖）。
    """
    # 1. 读取 auto_map，找到模型类路径
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    auto_map = getattr(model_config, "auto_map", {})
    model_cls_path = (
        auto_map.get("AutoModelForVision2Seq")
        or auto_map.get("AutoModelForImageTextToText")
    )
    if not model_cls_path:
        raise ValueError(f"无法从 config.json 的 auto_map 找到模型类。auto_map={auto_map}")

    module_name, cls_name = model_cls_path.split(".")  # "modeling_prismatic", "OpenVLAForActionPrediction"

    # 2. 注入 prismatic_shim：替代 openvla-oft 的 prismatic 包
    shim_dir = os.path.join(os.path.dirname(__file__), "prismatic_shim")
    if "prismatic" not in sys.modules:
        prismatic_pkg = types.ModuleType("prismatic")
        prismatic_pkg.__path__ = [shim_dir]
        prismatic_pkg.__package__ = "prismatic"
        sys.modules["prismatic"] = prismatic_pkg

    # 3. 将模型目录注册为包，支持 modeling_prismatic.py 内的相对导入
    pkg_name = "prismatic_model"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [model_path]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    # 4. 用 importlib 直接加载 modeling_prismatic.py
    full_module_name = f"{pkg_name}.{module_name}"
    if full_module_name in sys.modules:
        del sys.modules[full_module_name]

    spec = importlib.util.spec_from_file_location(
        full_module_name,
        os.path.join(model_path, f"{module_name}.py"),
        submodule_search_locations=[model_path],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_module_name] = mod
    spec.loader.exec_module(mod)

    ModelClass = getattr(mod, cls_name)

    # 5. 补丁：transformers 5.x 要求这些类变量存在
    for attr, default in [
        ("_supports_sdpa", True),
        ("_supports_flash_attn_2", False),
        ("_supports_flex_attn", False),
    ]:
        if not hasattr(ModelClass, attr):
            setattr(ModelClass, attr, default)

    return ModelClass


# ─────────────────────────────────────────────────────────────────────────────
# 主 Policy 类
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPolicy(PreTrainedPolicy):
    """OpenVLA-OFT Policy，符合 LeRobot PreTrainedPolicy 接口。

    直接复用 OpenVLAForActionPrediction.predict_action() 进行推理，
    不另建独立的 action head。

    使用示例（推理）：
        policy = OpenVLAPolicy(config).cuda().eval()
        policy.reset()
        action = policy.select_action(observation)  # (action_dim,)

    使用示例（训练）：
        lerobot-train --policy.type openvla ...
    """

    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config)
        self.config: OpenVLAConfig = config

        # ── 加载 Prismatic/OpenVLA backbone ──────────────────────────────────
        logger.info(f"加载 OpenVLA backbone：{config.pretrained_backbone}")
        torch_dtype = getattr(torch, config.torch_dtype)

        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=config.load_in_4bit,
                    load_in_8bit=config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                raise ImportError("量化需要安装 bitsandbytes：pip install bitsandbytes")

        ModelClass = _load_prismatic_model_class(config.pretrained_backbone)

        load_kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True)
        # transformers 5.x 用 dtype，4.x 用 torch_dtype
        try:
            import transformers
            from packaging.version import Version
            if Version(transformers.__version__) >= Version("5.0.0"):
                load_kwargs["dtype"] = torch_dtype
            else:
                load_kwargs["torch_dtype"] = torch_dtype
        except Exception:
            load_kwargs["torch_dtype"] = torch_dtype

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.vla = ModelClass.from_pretrained(config.pretrained_backbone, **load_kwargs)

        # 设置多相机输入数量（模型内部用于 pixel_values channel 拆分）
        if config.num_images_in_input > 1:
            if hasattr(self.vla, "vision_backbone") and hasattr(self.vla.vision_backbone, "set_num_images_in_input"):
                self.vla.vision_backbone.set_num_images_in_input(config.num_images_in_input)
                logger.info(f"设置 num_images_in_input={config.num_images_in_input}")

        # ── 可选：注入 LoRA ────────────────────────────────────────────────
        if config.use_lora:
            self._inject_lora()

        # ── 推理用 action chunk 队列 ──────────────────────────────────────
        self._action_queue: deque[torch.Tensor] = deque()

        logger.info(
            f"OpenVLAPolicy 初始化完成。"
            f"action_dim={config.action_dim}, chunk_size={config.action_chunk_size}, "
            f"use_lora={config.use_lora}, use_proprio={config.use_proprio}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_lora(self) -> None:
        """向 VLA backbone 注入 LoRA 适配器。"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("LoRA 需要安装 peft：pip install peft")

        cfg = self.config
        target_modules = cfg.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.vla = get_peft_model(self.vla, lora_config)
        self.vla.print_trainable_parameters()

    def _collect_pixel_values(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """收集 pixel_values。

        优先使用 processor 处理后的 "pixel_values" 键（6通道或12通道）。
        若不存在则按 image_keys 收集原始图像。

        Returns:
            单相机：(B, 6, H, W)
            多相机：(B, 6*N, H, W)
        """
        # processor 处理后的多相机图像存在 "pixel_values" 键
        if "pixel_values" in batch:
            return batch["pixel_values"]

        # 单相机：processor 把结果存回了 image_keys[0]
        image_keys = list(self.config.image_keys)
        images = []
        for key in image_keys:
            if key not in batch:
                raise KeyError(
                    f"batch 中缺少图像键 '{key}'，"
                    f"请检查 config.image_keys={self.config.image_keys}。"
                )
            images.append(batch[key])

        if len(images) == 1:
            return images[0]
        else:
            # 多相机：在 channel 维度拼接
            return torch.cat(images, dim=1)  # (B, 6*N, H, W)

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 接口：forward()（训练）
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """训练前向传播，直接调用 VLA 的 forward 计算 loss。

        Args:
            batch: 包含以下键：
                - "observation.images.*": (B, C, H, W)
                - "input_ids": (B, seq_len)
                - "attention_mask": (B, seq_len)
                - "labels": (B, seq_len) action token 标签
                - "observation.state": (B, proprio_dim) [可选]

        Returns:
            (loss, info)
        """
        pixel_values = self._collect_pixel_values(batch)
        proprio = batch.get("observation.state") if self.config.use_proprio else None

        outputs = self.vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=pixel_values,
            labels=batch.get("labels"),
            proprio=proprio,
            use_film=self.config.use_film,
            return_dict=True,
        )

        loss = outputs.loss
        info = {"loss_action": loss.item()}
        return loss, info

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 接口：select_action()（推理）
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """单步推理，返回下一个要执行的动作。

        实现 Action Chunking：队列为空时调用 predict_action_chunk 填充，
        然后逐步从队列取动作执行。

        Args:
            observation:
                - "observation.images.*": (1, C, H, W)
                - "input_ids": (1, seq_len)
                - "attention_mask": (1, seq_len)
                - "observation.state": (1, proprio_dim) [可选]

        Returns:
            action: (action_dim,)
        """
        self.eval()

        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(observation)  # (1, chunk_size, action_dim)
            for step in range(self.config.num_open_loop_steps):
                self._action_queue.append(chunk[0, step])

        return self._action_queue.popleft()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """调用 VLA 的 predict_action()，返回完整 action chunk。

        直接复用 OpenVLAForActionPrediction 内置的预测逻辑：
        - 准备 input tokens（添加 action placeholder）
        - 并行解码得到 action chunk
        - 反归一化到原始动作空间

        Returns:
            chunk: (1, action_chunk_size, action_dim) float32 tensor
        """
        pixel_values = self._collect_pixel_values(batch)
        proprio = None
        if self.config.use_proprio and "observation.state" in batch:
            proprio = batch["observation.state"].cpu().numpy()

        # predict_action 返回 (numpy_actions, hidden_states)
        # numpy_actions shape: (action_chunk_size, action_dim)
        actions_np, _ = self.vla.predict_action(
            input_ids=batch["input_ids"],
            pixel_values=pixel_values,
            attention_mask=batch["attention_mask"],
            unnorm_key=self.config.unnorm_key,
            proprio=proprio,
            use_film=self.config.use_film,
            action_head=None,  # None = L1 regression（OFT 默认）
        )

        # numpy → tensor，增加 batch 维度
        actions = torch.from_numpy(actions_np).float()
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)  # (chunk_size, action_dim) → (1, chunk_size, action_dim)

        return actions.to(batch["input_ids"].device)

    def reset(self) -> None:
        """清空 action chunk 队列（episode 开始时调用）。"""
        self._action_queue.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 接口：get_optim_params()
    # ─────────────────────────────────────────────────────────────────────────

    def get_optim_params(self) -> dict:
        """返回优化器参数组。

        LoRA 模式：只训练 LoRA 参数（0.1x lr）
        全量微调：backbone 0.1x lr
        """
        if self.config.use_lora:
            trainable = [p for p in self.vla.parameters() if p.requires_grad]
            return [{"params": trainable, "lr_scale": 0.1}]
        else:
            return [{"params": self.vla.parameters(), "lr_scale": 0.1}]

    # ─────────────────────────────────────────────────────────────────────────
    # LoRA 工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def merge_lora_weights(self) -> None:
        """合并 LoRA 权重（推理加速，合并后无法继续训练）。"""
        if not self.config.use_lora:
            logger.warning("模型未启用 LoRA，merge_lora_weights() 无效。")
            return
        try:
            self.vla = self.vla.merge_and_unload()
            logger.info("LoRA 权重已合并。")
        except AttributeError:
            logger.warning("当前 backbone 不支持 merge_and_unload()。")
