# main_batch_raw.py (已修改以适配混合版 Demosaic)

import imageio
import numpy as np
import os
import glob
from tqdm import tqdm
import cv2

# 导入我们的ISP管道和模块
from isp_pipeline import ISPPipeline
from demosaic import Demosaic  # 导入我们最终的混合版本
from white_balance import WhiteBalance
from gamma_correction import GammaCorrection

def main_batch():
    # --- 1. 定义传感器/图像的元数据 ---
    # !! 这是最关键的改动：必须为无头RAW文件提供元数据 !!
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 800
    IMAGE_DTYPE = np.uint16  # 或 np.uint8, 取决于您的RAW数据位深
    BAYER_PATTERN = 'GRBG'   # 根据传感器规格设置

    # --- 2. 定义输入和输出文件夹 ---
    input_folder = 'raw_images_sequence/' # 存放RAW序列的文件夹
    output_folder = 'isp_processed_frames/'   # 存放处理后PNG帧的文件夹
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 3. 获取所有RAW文件 ---
    # <--- 关键改动: 文件扩展名从.dng改为.raw，以匹配无头RAW文件的场景
    raw_files = sorted(glob.glob(os.path.join(input_folder, '*.raw')))
    
    if not raw_files:
        print(f"在文件夹 '{input_folder}' 中没有找到 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 个 .raw 文件进行处理。")

    # --- 4. 实例化并组装ISP管道 ---
    # <--- 关键改动: 在创建Demosaic实例时，必须传入元数据
    demosaic_module = Demosaic(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        bayer_pattern=BAYER_PATTERN,
        dtype=IMAGE_DTYPE
    )
    
    my_isp = ISPPipeline(modules=[
        demosaic_module,
        WhiteBalance(),
        GammaCorrection()
    ])

    # --- 5. 定义处理参数 (所有帧使用相同参数) ---
    processing_params = {
        'demosaic': {'algorithm': 'LMMSE'},
        'whitebalance': {'algorithm': 'gray_world'},
        'gammacorrection': {'gamma': 2.2}
    }

    # --- 6. 循环处理所有文件 ---
    # <--- 注意: 这里的 my_isp.process 应该使用我们优化后的版本
    for raw_file_path in tqdm(raw_files, desc="Processing RAW sequence"):
        try:
            # 运行管道
            final_image = my_isp.process(raw_file_path, params=processing_params)
            
            # 生成输出文件名
            base_name = os.path.basename(raw_file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_folder, f"{file_name_without_ext}.png")
            
            # 保存为16位PNG
            cv2.imwrite(output_path, final_image)

        except Exception as e:
            print(f"处理文件 {raw_file_path} 时出错: {e}")
            continue

    print(f"所有帧处理完毕，已保存至 '{output_folder}' 文件夹。")

    # --- 7. (可选) 将处理后的帧合成为视频 ---
    print("正在将处理后的帧合成为视频...")
    processed_frames = sorted(glob.glob(os.path.join(output_folder, '*.png')))
    
    if not processed_frames:
        print("没有找到已处理的帧，无法创建视频。")
        return

    # 使用 'I' 模式读取16位PNG
    with imageio.get_writer('output_video.mp4', fps=30, quality=8) as writer:
        for frame_path in tqdm(processed_frames, desc="Creating video"):
            frame = imageio.imread(frame_path, mode='I')
            # 写入视频前需要从16位 (0-65535) 转换为8位 (0-255)
            # (frame / 257) 是比 (frame / 256) 更精确的转换方式
            frame_8bit = (frame / 257).astype(np.uint8)
            writer.append_data(frame_8bit)
            
    print("视频 'output_video.mp4' 创建成功！")


if __name__ == "__main__":
    main_batch()