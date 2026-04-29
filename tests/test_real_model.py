import os
import torch
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
)
print("   ✅ Config OK")

print("2. 加载模型权重（需要几分钟）...")
policy = OpenVLAPolicy(cfg)
policy = policy.cuda().eval()
print("   ✅ 模型加载完成")

print("3. 单步推理测试...")
obs = {
    "observation.images.agentview_rgb": torch.randn(1, 3, 224, 224).cuda(),
    "input_ids": torch.randint(0, 32000, (1, 20)).cuda(),
    "attention_mask": torch.ones(1, 20, dtype=torch.long).cuda(),
}
policy.reset()
with torch.no_grad():
    action = policy.select_action(obs)

print(f"   action shape: {action.shape}")
print(f"   action values: {action}")
assert action.shape == torch.Size([7])
print("   ✅ 全部通过")