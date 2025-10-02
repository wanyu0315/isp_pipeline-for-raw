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
        'demosaic': {'algorithm': 'CV'},
        'whitebalance': {'algorithm': 'gray_world'},
        'gammacorrection': {'gamma': 2.2}
    }

    # --- 6. 循环处理所有文件 ---
    # <--- 注意: 这里的 my_isp.process 应该使用我们优化后的版本
    for raw_file_path in tqdm(raw_files, desc="Processing RAW sequence"):
        try:
            # 运行管道
            final_image = my_isp.process(raw_file_path, params=processing_params)
            # final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)  # 转换为RGB顺序
            
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

    """
        --- 额外说明 ---
        注意使用CV的去马赛克算法时，输出图像的颜色通道顺序是BGR而不是RGB。
        因此在后续进行合成视频时，需注意这一点，如果不是使用OpenCV进行视频写入，
        可能需要转换颜色通道顺序。
        当imageio读取您的PNG文件时，它并不知道这个文件是OpenCV以BGR顺序创建的。它只是按顺序读取了三个通道的数据，
        并把它们加载到一个NumPy数组中。
        frame = imageio.imread(frame_path) 这行代码返回的frame变量，其内存中的通道顺序实际上还是 B-G-R。
    """

    # --- 7. (可选) 将处理后的帧合成为视频 (OpenCV-MKV无损方案) ---
    print("正在将处理后的帧合成为无损视频 (FFV1)...")
    processed_frames = sorted(glob.glob(os.path.join(output_folder, '*.png')))
    
    if not processed_frames:
        # ... (错误处理)
        return
        
    first_frame = cv2.imread(processed_frames[0], cv2.IMREAD_UNCHANGED)
    height, width, _ = first_frame.shape

    #  指定输出文件为 .avi 或 .mkv，它们对FFV1支持更好
    output_video_path = 'output_video_lossless.mkv'
    
    #  使用 FFV1 的 FourCC 代码
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') 
    writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    if not writer.isOpened():
        print("无法打开VideoWriter，请检查OpenCV配置。")
        return

    for frame_path in tqdm(processed_frames, desc="Creating Lossless Video"):
        frame_16bit_bgr = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        frame_8bit_bgr = (frame_16bit_bgr / 257.0).astype(np.uint8)
        writer.write(frame_8bit_bgr)
            
    writer.release()
    print(f"无损视频 '{output_video_path}' 创建成功！")

    # # --- 7. (可选) 将处理后的帧合成为视频 (OpenCV-MP4格式) ---
    # print("正在将处理后的帧合成为视频 (使用OpenCV)...")
    # processed_frames = sorted(glob.glob(os.path.join(output_folder, '*.png')))
    
    # if not processed_frames:
    #     print("没有找到已处理的帧，无法创建视频。")
    #     return
        
    # #  从第一张图片获取视频的尺寸
    # first_frame = cv2.imread(processed_frames[0], cv2.IMREAD_UNCHANGED)
    # if first_frame is None:
    #     print(f"无法读取第一帧图像: {processed_frames[0]}")
    #     return
    # height, width, _ = first_frame.shape

    # #  定义视频编码器和创建 VideoWriter 对象
    # # 'mp4v' 是一个常用的MP4编码器
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    # if not writer.isOpened():
    #     print("无法打开VideoWriter，请检查OpenCV配置。")
    #     return

    # for frame_path in tqdm(processed_frames, desc="Creating video with OpenCV"):
    #     #  使用OpenCV读取16位PNG图像 (它会读取为BGR顺序)
    #     frame_16bit_bgr = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        
    #     if frame_16bit_bgr is None:
    #         print(f"警告：跳过无法读取的帧 {frame_path}")
    #         continue

    #     #  将16位数据转换为8位
    #     frame_8bit_bgr = (frame_16bit_bgr / 257.0).astype(np.uint8)
        
    #     #  将8位帧写入视频
    #     writer.write(frame_8bit_bgr)
            
    # # 释放writer对象，这是完成视频写入的关键步骤！
    # writer.release()
    # print("视频 'output_video.mp4' 创建成功！")

if __name__ == "__main__":
    main_batch()