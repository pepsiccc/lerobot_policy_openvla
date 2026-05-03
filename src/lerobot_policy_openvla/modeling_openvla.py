"""
OpenVLA-OFT Policy — LeRobot 集成实现。

架构概览：
┌─────────────────────────────────────────────────────────────────────┐
│  训练路径（OFT L1 regression）                                        │
│  图像 + 语言 + 本体 → Prismatic backbone → hidden states             │
│                    → MLPActionHead → pred_actions                    │
│                    → L1 loss(pred, target)                           │
│                                                                       │
│  推理路径                                                              │
│  图像 + 语言 → predict_action() → 连续动作 chunk                     │
└─────────────────────────────────────────────────────────────────────┘
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
# MLPActionHead：OFT L1 regression action head
# ─────────────────────────────────────────────────────────────────────────────

class MLPActionHead(nn.Module):
    """将 LLM action token 位置的 hidden states 映射到连续动作。

    对应 OFT 原版的 action_head.predict_action()：
    输入：(B, action_chunk_size, llm_dim)
    输出：(B, action_chunk_size, action_dim)
    """

    def __init__(
        self,
        llm_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        layers: list[nn.Module] = []
        in_dim = llm_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, action_chunk_size, llm_dim)
        Returns:
            actions: (B, action_chunk_size, action_dim)
        """
        return self.mlp(hidden_states)

    def predict_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """推理时调用，与 forward 等价。"""
        return self.forward(hidden_states)


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具：加载 Prismatic 模型类
# ─────────────────────────────────────────────────────────────────────────────

def _load_prismatic_model_class(model_path: str):
    """从本地模型目录加载 OpenVLAForActionPrediction 类。"""
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    auto_map = getattr(model_config, "auto_map", {})
    model_cls_path = (
        auto_map.get("AutoModelForVision2Seq")
        or auto_map.get("AutoModelForImageTextToText")
    )
    if not model_cls_path:
        raise ValueError(f"无法从 config.json 的 auto_map 找到模型类。auto_map={auto_map}")

    module_name, cls_name = model_cls_path.split(".")

    # 注入 prismatic_shim
    shim_dir = os.path.join(os.path.dirname(__file__), "prismatic_shim")
    if "prismatic" not in sys.modules:
        prismatic_pkg = types.ModuleType("prismatic")
        prismatic_pkg.__path__ = [shim_dir]
        prismatic_pkg.__package__ = "prismatic"
        sys.modules["prismatic"] = prismatic_pkg

    # 将模型目录注册为包
    pkg_name = "prismatic_model"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [model_path]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

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

    for attr, default in [
        ("_supports_sdpa", True),
        ("_supports_flash_attn_2", False),
        ("_supports_flex_attn", False),
    ]:
        if not hasattr(ModelClass, attr):
            setattr(ModelClass, attr, default)

    return ModelClass


def _get_vla_base(vla):
    """获取 VLA 的底层模型（穿透 LoRA peft 包装）。
    
    LoRA 注入后：vla 是 PeftModel，vla.base_model.model 是原始 Prismatic 模型
    未注入 LoRA：vla 本身就是原始模型，vla.base_model 是同一个对象
    """
    # peft 包装后 vla.base_model 是 LoraModel，vla.base_model.model 才是原始模型
    if hasattr(vla, "base_model") and hasattr(vla.base_model, "model"):
        return vla.base_model.model
    return vla


# ─────────────────────────────────────────────────────────────────────────────
# 主 Policy 类
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPolicy(PreTrainedPolicy):
    """OpenVLA-OFT Policy，符合 LeRobot PreTrainedPolicy 接口。

    训练：OFT L1 regression loss（MLPActionHead）
    推理：predict_action()（模型内置）
    """

    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config)
        self.config: OpenVLAConfig = config

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

        # 设置多相机输入数量
        if config.num_images_in_input > 1:
            base = _get_vla_base(self.vla)
            if hasattr(base, "vision_backbone") and hasattr(base.vision_backbone, "set_num_images_in_input"):
                base.vision_backbone.set_num_images_in_input(config.num_images_in_input)
                logger.info(f"设置 num_images_in_input={config.num_images_in_input}")

        # LoRA
        if config.use_lora:
            self._inject_lora()
            if hasattr(self.vla, "enable_input_require_grads"):
                self.vla.enable_input_require_grads()
            if hasattr(self.vla, "gradient_checkpointing_enable"):
                self.vla.gradient_checkpointing_enable()
                logger.info("已启用 gradient checkpointing")

        # 获取 LLM 隐藏维度
        self.llm_dim = self._get_llm_dim()

        # MLPActionHead（训练用）
        self.action_head = MLPActionHead(
            llm_dim=self.llm_dim,
            action_dim=config.action_dim,
            action_chunk_size=config.action_chunk_size,
            hidden_dim=config.action_head_hidden_dim,
            num_hidden_layers=config.action_head_num_hidden_layers,
        ).to(dtype=torch_dtype)

        # 推理用 action chunk 队列
        self._action_queue: deque[torch.Tensor] = deque()

        logger.info(
            f"OpenVLAPolicy 初始化完成。"
            f"llm_dim={self.llm_dim}, action_dim={config.action_dim}, "
            f"chunk_size={config.action_chunk_size}, use_lora={config.use_lora}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _get_llm_dim(self) -> int:
        """从 backbone 中提取 LLM 隐藏层维度。"""
        base = _get_vla_base(self.vla)
        for attr_path in [
            "language_model.config.hidden_size",
            "config.text_config.hidden_size",
            "config.hidden_size",
        ]:
            obj = base
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return int(obj)
            except AttributeError:
                continue
        logger.warning("无法自动检测 LLM hidden_size，使用默认值 4096。")
        return 4096

    def _inject_lora(self) -> None:
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
        """优先使用 processor 处理后的 pixel_values，否则按 image_keys 收集。"""
        if "pixel_values" in batch:
            return batch["pixel_values"]

        image_keys = list(self.config.image_keys)
        images = []
        for key in image_keys:
            if key not in batch:
                raise KeyError(f"batch 中缺少图像键 '{key}'")
            images.append(batch[key])

        if len(images) == 1:
            return images[0]
        return torch.cat(images, dim=1)  # (B, 6*N, H, W)

    def _get_base_model(self):
        """获取底层 Prismatic 模型（穿透 LoRA 包装）。"""
        return _get_vla_base(self.vla)

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 接口：forward()（训练，OFT L1 regression）
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """OFT 训练前向传播：提取 hidden states → MLPActionHead → L1 loss。

        Args:
            batch: LeRobot DataLoader 输出，包含：
                - "observation.images.*" 或 "pixel_values": 图像
                - "input_ids": tokenized 提示词
                - "attention_mask": 注意力掩码
                - "labels": action mask 用的 labels
                - "action": (B, action_dim) 目标动作
                - "observation.state": (B, proprio_dim) [可选]

        Returns:
            (loss, info)
        """
        # 如果没有预处理，先处理
        if "input_ids" not in batch:
            from .processor_openvla import OpenVLAPreProcessor
            if not hasattr(self, "_pre_processor"):
                self._pre_processor = OpenVLAPreProcessor(config=self.config)
            batch = self._pre_processor(batch)
            device = next(self.vla.parameters()).device
            dtype = getattr(torch, self.config.torch_dtype)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if "pixel_values" in k or "images" in k:
                        batch[k] = v.to(device=device, dtype=dtype)
                    else:
                        batch[k] = v.to(device=device)

        pixel_values = self._collect_pixel_values(batch)
        base = self._get_base_model()
        proprio = batch.get("observation.state") if self.config.use_proprio else None

        # 准备 input：加 action placeholder tokens（与推理时对齐）
        input_ids, attention_mask = base._prepare_input_for_action_prediction(
            batch["input_ids"], batch["attention_mask"]
        )
        labels = base._prepare_labels_for_action_prediction(
            batch["labels"], input_ids
        )

        # 获取 input embeddings 和 action mask
        input_embeddings = base.get_input_embeddings()(input_ids)
        all_actions_mask = base._process_action_masks(labels)

        # 提取语言部分 embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 视觉特征
        projected_patch_embeddings = base._process_vision_features(
            pixel_values, language_embeddings, self.config.use_film
        )

        # proprio 特征（可选）
        if proprio is not None:
            projected_patch_embeddings = base._process_proprio_features(
                projected_patch_embeddings, proprio, None
            )

        # 清零 action token embeddings
        all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_3d

        # 构建多模态 embeddings
        multimodal_embeddings, multimodal_attention_mask = base._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # LLM 前向，获取 hidden states
        lm_output = self.vla(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeddings,
            output_hidden_states=True,
            return_dict=True,
        ) if not hasattr(self.vla, "base_model") else self.vla.base_model.model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeddings,
            output_hidden_states=True,
            return_dict=True,
        )

        # 提取 action token 位置的 hidden states
        NUM_PATCHES = (
            base.vision_backbone.get_num_patches()
            * base.vision_backbone.get_num_images_in_input()
        )
        NUM_PROMPT_TOKENS = batch["input_ids"].shape[-1] - 1
        action_len = self.config.action_dim * self.config.action_chunk_size

        last_hidden = lm_output.hidden_states[-1]  # (B, seq_len, llm_dim)
        start = NUM_PATCHES + NUM_PROMPT_TOKENS
        end = start + action_len
        actions_hidden_flat = last_hidden[:, start:end, :]  # (B, action_len, llm_dim)

        # reshape: (B, chunk_size, action_dim, llm_dim) → mean over action_dim → (B, chunk_size, llm_dim)
        B = actions_hidden_flat.shape[0]
        actions_hidden = actions_hidden_flat.reshape(
            B, self.config.action_chunk_size, self.config.action_dim, self.llm_dim
        ).mean(dim=2)  # (B, chunk_size, llm_dim)

        # MLPActionHead 预测连续动作
        pred_actions = self.action_head(actions_hidden)  # (B, chunk_size, action_dim)

        # 目标动作：(B, action_dim) → (B, chunk_size, action_dim)
        target = batch["action"]
        if target.dim() == 2:
            target = target.unsqueeze(1).expand_as(pred_actions)

        loss = nn.functional.l1_loss(pred_actions, target)
        info = {"loss_action": loss.item()}
        return loss, info

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 接口：select_action()（推理）
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """单步推理，返回下一个要执行的动作。"""
        self.eval()

        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(observation)
            for step in range(self.config.num_open_loop_steps):
                self._action_queue.append(chunk[0, step])

        return self._action_queue.popleft()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """调用模型内置的 predict_action()，返回完整 action chunk。

        Returns:
            chunk: (1, action_chunk_size, action_dim)
        """
        pixel_values = self._collect_pixel_values(batch)
        proprio = None
        if self.config.use_proprio and "observation.state" in batch:
            proprio = batch["observation.state"].cpu().numpy()

        actions_np, _ = self.vla.predict_action(
            input_ids=batch["input_ids"],
            pixel_values=pixel_values,
            attention_mask=batch["attention_mask"],
            unnorm_key=self.config.unnorm_key,
            proprio=proprio,
            use_film=self.config.use_film,
            action_head=None,
        )

        actions = torch.from_numpy(actions_np).float()
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)

        return actions.to(batch["input_ids"].device)

    def reset(self) -> None:
        """清空 action chunk 队列。"""
        self._action_queue.clear()

    def predict_action_chunk_train(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """训练时用 MLPActionHead 预测 action chunk（供外部调用）。"""
        loss, _ = self.forward(batch)
        return loss

    def get_optim_params(self) -> dict:
        """返回优化器参数组，差异化学习率。"""
        backbone_params = [p for p in self.vla.parameters() if p.requires_grad]
        head_params = list(self.action_head.parameters())
        return [
            {"params": backbone_params, "lr_scale": 0.1},
            {"params": head_params, "lr_scale": 1.0},
        ]

    def merge_lora_weights(self) -> None:
        """合并 LoRA 权重（推理加速）。"""
        if not self.config.use_lora:
            logger.warning("模型未启用 LoRA。")
            return
        try:
            self.vla = self.vla.merge_and_unload()
            logger.info("LoRA 权重已合并。")
        except AttributeError:
            logger.warning("当前 backbone 不支持 merge_and_unload()。")
