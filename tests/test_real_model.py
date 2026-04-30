import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from lerobot_policy_openvla.configuration_openvla import OpenVLAConfig
from lerobot_policy_openvla.modeling_openvla import OpenVLAPolicy

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-10"
    "/snapshots/95220f9a3421a7ff12d4218e73d09ade830fa9a3"
)

print("1. 加载 Config...")
cfg = OpenVLAConfig(
    pretrained_backbone=MODEL_PATH,
    action_dim=7,
    action_chunk_size=8,
    use_proprio=False,
    num_images_in_input=1,
    image_keys=("observation.images.agentview_rgb",),
    unnorm_key="bridge_orig",    # ← 指定归一化 key
)
print("   ✅ Config OK")

print("2. 加载模型权重（需要几分钟）...")
policy = OpenVLAPolicy(cfg)
policy = policy.cuda().eval()
torch_dtype = getattr(torch, cfg.torch_dtype)
print("   ✅ 模型加载完成")

print("3. 加载 Processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("   ✅ Processor 加载完成")

print("4. 单步推理测试...")
fake_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the block?\nOut:"
inputs = processor(text=prompt, images=fake_img, return_tensors="pt")

obs = {
    "observation.images.agentview_rgb": inputs["pixel_values"].to(device="cuda", dtype=torch_dtype),
    "input_ids": inputs["input_ids"].cuda(),
    "attention_mask": inputs["attention_mask"].cuda(),
}

policy.reset()
with torch.no_grad():
    action = policy.select_action(obs)

print(f"   action shape: {action.shape}")
print(f"   action values: {action}")
assert action.shape == torch.Size([7]), f"期望 (7,)，得到 {action.shape}"
print("   ✅ 全部通过")
