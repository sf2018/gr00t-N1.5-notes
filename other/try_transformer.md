```import numpy as np
from PIL import Image
from gr00t.model.transforms import GR00TTransform
from gr00t.data.schema import EmbodimentTag
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor
import matplotlib.pyplot as plt

# 读取图片
img = Image.open("/workspace/src/Isaac-GR00T-1.5/getting_started/01.JPEG").convert("RGB")
img_np = np.array(img)

# 假的 [T=1, V=1, H, W, C] 输入（模拟视频帧）
video = img_np[None, None, ...]  # (1, 1, H, W, C)

# 构造 transform
transform = GR00TTransform(
    vlm_processor=EagleProcessor(),      # 默认初始化
    state_horizon=1,
    action_horizon=1,
    max_state_dim=1,
    max_action_dim=1,
    training=False,
    embodiment_tag=EmbodimentTag.GR1     # or your tag
)

# 设置数据
data = {
    "video": video,
    "language": "pick up the red cube"   # 提供语言指令
}

# 应用 transform
out = transform(data)

# 获取图像 tensor 并可视化
tensor = out["pixel_values"]  # (1, 3, H, W)
tensor = tensor[0].to(torch.float32).permute(1, 2, 0).numpy()  # 转为 (H, W, C)
tensor = (tensor * 0.5 + 0.5).clip(0, 1)     # 反归一化

# 显示原图和处理后的图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(tensor)
axs[1].set_title("After GR00TTransform (resized + normalized)")
axs[1].axis("off")

plt.show()
