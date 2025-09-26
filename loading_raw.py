import numpy as np
import cv2
import os

# --- 1. 基本参数设置 ---
WIDTH = 1280
HEIGHT = 800
DTYPE = np.uint16
raw_file_path = r"C:\librealsense\rs_convert_res\raw_\raw_frame_Color_1758105205861.21386718750000.raw"

# D455 通常是 BGGR 或 RGGB，可根据实际情况切换
BAYER_PATTERN = cv2.COLOR_BAYER_GR2RGB_VNG  # GRBG

# --- 2. 加载 RAW 文件 ---
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"错误: 文件不存在 {raw_file_path}")

bayer_raw_1d = np.fromfile(raw_file_path, dtype=DTYPE)
raw_image = bayer_raw_1d.reshape((HEIGHT, WIDTH))
print("RAW 文件加载成功。")

print("dtype:", raw_image.dtype)

# --- 3. 去马赛克 ---
rgb_16bit = cv2.cvtColor(raw_image, BAYER_PATTERN)
print("去马赛克完成。")
print("输出 dtype:", rgb_16bit.dtype)

# 保存原始去马赛克结果 (16位PNG)
cv2.imwrite("demosaic_raw16.png", rgb_16bit)
print("保存原始去马赛克图像 -> demosaic_raw16.png")

# --- 4. 自动白平衡 (灰世界算法) ---
rgb_float = rgb_16bit.astype(np.float32)
b, g, r = cv2.split(rgb_float)

avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
avg_gray = avg_g

gain_b = avg_gray / avg_b
gain_r = avg_gray / avg_r

# 限制增益范围，避免过度
gain_b = np.clip(gain_b, 0.5, 2.0)
gain_r = np.clip(gain_r, 0.5, 2.0)

b_balanced = b * gain_b
r_balanced = r * gain_r
balanced_rgb_float = cv2.merge([b_balanced, g, r_balanced])

# 裁剪到16位范围并转回uint16
wb_rgb_16bit = np.clip(balanced_rgb_float, 0, 65535).astype(np.uint16)
print("自动白平衡完成。")

# 保存白平衡后的 16 位结果
cv2.imwrite("wb_raw16.png", wb_rgb_16bit)
print("保存白平衡矫正图像 (16位) -> wb_raw16.png")

# --- 5. 转换到8位以供显示 ---
display_8bit = cv2.normalize(wb_rgb_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 保存并显示结果
cv2.imwrite("wb_display_8bit.png", display_8bit)
print("保存显示用图像 (8位) -> wb_display_8bit.png")

cv2.imshow("Corrected Image (8-bit)", display_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()