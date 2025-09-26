# white_balance.py
import numpy as np

class WhiteBalance:
    """
    白平衡处理模块
    包含多种白平衡算法。
    """
    def _gray_world(self, image: np.ndarray) -> np.ndarray:
        """灰度世界算法"""
        # 将图像转换为浮点数进行计算
        img_f = image.astype(np.float32)
        
        # 计算每个通道的平均值
        r_avg = np.mean(img_f[:, :, 0])
        g_avg = np.mean(img_f[:, :, 1])
        b_avg = np.mean(img_f[:, :, 2])
        
        # 计算所有通道的全局平均值（灰度值）
        gray_avg = (r_avg + g_avg + b_avg) / 3
        
        # 计算每个通道的增益
        r_gain = gray_avg / r_avg
        g_gain = gray_avg / g_avg
        b_gain = gray_avg / b_avg
        
        # 应用增益
        img_f[:, :, 0] *= r_gain
        img_f[:, :, 1] *= g_gain
        img_f[:, :, 2] *= b_gain
        
        # 裁剪到有效范围并转换回原始数据类型
        # 注意：假设输入是16位图像，最大值为65535
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_f, 0, max_val).astype(image.dtype)
    
    def _gray_world_green(self, image: np.ndarray) -> np.ndarray:
        """灰度世界算法 (以绿色通道为基准)"""
        img_f = image.astype(np.float32)
        
        # 计算每个通道的平均值
        r_avg = np.mean(img_f[:, :, 0])
        g_avg = np.mean(img_f[:, :, 1])
        b_avg = np.mean(img_f[:, :, 2])
        
        # 以绿色通道为基准
        r_gain = g_avg / r_avg
        g_gain = 1.0  # 绿色通道不变
        b_gain = g_avg / b_avg

        # 限制增益范围，避免过度
        b_gain = np.clip(b_gain, 0.5, 2.0)
        r_gain = np.clip(r_gain, 0.5, 2.0)

        # 应用增益
        img_f[:, :, 0] *= r_gain
        img_f[:, :, 1] *= g_gain
        img_f[:, :, 2] *= b_gain
        
        # 根据输入图像image的数据类型裁剪并转换
        max_val = np.iinfo(image.dtype).max
        return np.clip(img_f, 0, max_val).astype(image.dtype)

    def _perfect_reflector(self, image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
        """完美反射算法 (也叫白点算法)"""
        img_f = image.astype(np.float32)
        
        # 找到每个通道的“最亮点”
        # 我们使用百分位数来避免噪点或过曝区域的影响
        r_white = np.percentile(img_f[:, :, 0], percentile)
        g_white = np.percentile(img_f[:, :, 1], percentile)
        b_white = np.percentile(img_f[:, :, 2], percentile)
        
        # 假设最亮的点是白色 (R=G=B=max_val)
        # 我们以G通道为基准（通常G通道信噪比最好）
        # 或者以图像类型的最大值为基准
        max_val = np.iinfo(image.dtype).max
        
        r_gain = max_val / r_white
        g_gain = max_val / g_white
        b_gain = max_val / b_white
        
        # 应用增益
        img_f[:, :, 0] *= r_gain
        img_f[:, :, 1] *= g_gain
        img_f[:, :, 2] *= b_gain
        
        return np.clip(img_f, 0, max_val).astype(image.dtype)


    def execute(self, rgb_image: np.ndarray, algorithm: str = 'gray_world', **kwargs) -> np.ndarray:
        """
        执行白平衡操作。
        
        Args:
            rgb_image: 输入的RGB图像 (Numpy array)。
            algorithm: 'gray_world' 或 'perfect_reflector'。
            **kwargs: 传递给特定算法的额外参数 (例如 'percentile' for perfect_reflector)。

        Returns:
            经过白平衡处理的RGB图像。
        """
        print(f"Executing White Balance with algorithm: {algorithm}")
        if algorithm == 'gray_world':
            return self._gray_world(rgb_image)
        elif algorithm == 'perfect_reflector':
            percentile = kwargs.get('percentile', 99.5)
            return self._perfect_reflector(rgb_image, percentile)
        else:
            raise ValueError(f"不支持的白平衡算法: {algorithm}")
