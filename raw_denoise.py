# raw_denoise.py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# 尝试导入可选库
try:
    import pywt
    _pywt_available = True
except ImportError:
    _pywt_available = False


class RawDenoise:
    """
    原始域降噪模块 (Raw Domain Denoising)
    接收一个 2D Bayer 数组，返回一个 2D Bayer 数组。
    """
    
    def execute(self, raw_data: np.ndarray, bayer_pattern: str = 'GBRG',
                algorithm: str = 'bilateral', **kwargs) -> np.ndarray:
        """
        对Bayer Raw数据执行降噪。
        
        Args:
            raw_data: 输入的Bayer Raw数据 (来自 RawLoader 的 2D 数组)
            ... (其他参数不变)
        """
        # 移除了 'str' object has no attribute 'shape' 检查，
        # 因为我们假设 RawLoader 已经返回了 np.ndarray
        print(f"Executing Raw Domain Denoising using algorithm: {algorithm}")
        
        if algorithm == 'bilateral':
            return self._bilateral_raw(raw_data, **kwargs)
        elif algorithm == 'gaussian':
            return self._gaussian_raw(raw_data, **kwargs)
        elif algorithm == 'median':
            return self._median_raw(raw_data, **kwargs)
        elif algorithm == 'nlm':
            return self._nlm_raw(raw_data, **kwargs)
        elif algorithm == 'bayer_aware':
            return self._bayer_aware_denoise(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'green_uniform':
            return self._green_uniform_denoise(raw_data, bayer_pattern, **kwargs)
        elif algorithm == 'adaptive':
            return self._adaptive_raw_denoise(raw_data, **kwargs)
        elif algorithm == 'wavelet':
            return self._wavelet_raw_denoise(raw_data, **kwargs)
        else:
            raise ValueError(f"Unknown raw denoising algorithm: {algorithm}")
    
    def _bilateral_raw(self, raw_data: np.ndarray, d: int = 9,
                         sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        【高精度重构】双边滤波 - 保边降噪
        
        在 float32 上执行高精度滤波，而不是压缩到 uint8。
        """
        # 1. 转换为 float32 [0, 1]
        #    这是此模块中少数需要归一化的地方，因为 cv2.bilateralFilter
        #    的 sigma_color 参数 (75) 是为 [0, 255] 范围设计的。
        #    更好的做法是在 [0, 65535] 浮点数上操作。
        
        original_dtype = raw_data.dtype
        if original_dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = 255.0
            
        raw_float = raw_data.astype(np.float32) / max_val
        
        # 2. 缩放 sigma_color 以匹配 [0, 1] 范围
        sigma_color_scaled = sigma_color / 255.0
        
        # 3. 在 float32 上执行双边滤波
        denoised_float = cv2.bilateralFilter(raw_float, d, sigma_color_scaled, sigma_space)
        
        # 4. 转换回原始范围和类型
        denoised = (np.clip(denoised_float, 0, 1) * max_val).astype(original_dtype)
        return denoised
    
    def _gaussian_raw(self, raw_data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        高斯滤波 - (此函数已兼容)
        """
        # gaussian_filter 可以在 uint16 上工作，但 float 更精确
        data_float = raw_data.astype(np.float32)
        denoised = gaussian_filter(data_float, sigma=sigma)
        
        # 裁剪并转回
        max_val = np.iinfo(raw_data.dtype).max
        return np.clip(denoised, 0, max_val).astype(raw_data.dtype)
    
    def _median_raw(self, raw_data: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        中值滤波 - (此函数已兼容)
        
        注意: cv2.medianBlur 只支持 uint8, uint16, float32。
             它在 uint16 上工作得很好。
        """
        if ksize % 2 == 0:
            ksize += 1
        # cv2.medianBlur 直接支持 uint16
        return cv2.medianBlur(raw_data, ksize)
    
    def _nlm_raw(self, raw_data: np.ndarray, h: float = 10,
                   template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """
        非局部均值降噪 - (需转uint8)
        
        警告: cv2.fastNlMeansDenoising *只* 支持 uint8。
             这将导致 16-bit 数据的精度损失。
        """
        original_dtype = raw_data.dtype
        
        if original_dtype == np.uint16:
            print("警告: NLM 降噪需要转为 uint8，将导致 16-bit 精度损失。")
            raw_uint8 = (raw_data.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
            
            denoised_uint8 = cv2.fastNlMeansDenoising(
                raw_uint8, None, h, template_window_size, search_window_size
            )
            
            # 转换回 16-bit 范围
            denoised_float = denoised_uint8.astype(np.float32) / 255.0
            return (denoised_float * 65535.0).astype(np.uint16)
        
        elif original_dtype == np.uint8:
            return cv2.fastNlMeansDenoising(
                raw_data, None, h, template_window_size, search_window_size
            )
        else:
            raise TypeError(f"NLM 不支持的数据类型: {original_dtype}")
    
    def _bayer_aware_denoise(self, raw_data: np.ndarray, bayer_pattern: str,
                             strength: float = 1.0) -> np.ndarray:
        """
        【高精度重构】Bayer模式感知降噪 - 分别处理R/G/B通道
        使用高精度的 _bilateral_raw 进行通道降噪
        这是最适合Raw数据的方法！
        
        Args:
            bayer_pattern: Bayer模式
            strength: 降噪强度
        """
        h, w = raw_data.shape
        original_dtype = raw_data.dtype
        
        # 提取四个子通道
        channels = self._extract_bayer_channels(raw_data, bayer_pattern)
        
        # 对每个通道分别降噪
        denoised_channels = {}
        for color, (channel_data, mask) in channels.items():
            # 使用我们重构的 _bilateral_raw
            denoised_ch = self._bilateral_raw(
                channel_data,
                d=5,
                sigma_color=50 * strength,
                sigma_space=50 * strength
            )
            denoised_channels[color] = denoised_ch
        
        # 重组Bayer图像 (float32)
        denoised_float = self._reconstruct_bayer(denoised_channels, bayer_pattern, h, w)
        
        # 转换回原始类型
        max_val = np.iinfo(original_dtype).max
        return np.clip(denoised_float, 0, max_val).astype(original_dtype)

    def _green_uniform_denoise(self, raw_data: np.ndarray, bayer_pattern: str,
                               balance_strength: float = 0.5) -> np.ndarray:
        """
        绿色通道均匀化降噪 - 处理Gr/Gb差异
        Bayer模式中有两个绿色通道，它们应该相似但可能有噪声差异
        
        Args:
            bayer_pattern: Bayer模式
            balance_strength: 均衡强度 (0-1)
        """
        denoised = raw_data.copy().astype(np.float32)
        h, w = raw_data.shape
        
        # 提取两个绿色通道
        if bayer_pattern == 'RGGB':
            gr = raw_data[0::2, 1::2].astype(np.float32)  # R行的G
            gb = raw_data[1::2, 0::2].astype(np.float32)  # B行的G
        elif bayer_pattern == 'BGGR':
            gb = raw_data[0::2, 1::2].astype(np.float32)
            gr = raw_data[1::2, 0::2].astype(np.float32)
        elif bayer_pattern == 'GRBG':
            gr = raw_data[0::2, 0::2].astype(np.float32)
            gb = raw_data[1::2, 1::2].astype(np.float32)
        elif bayer_pattern == 'GBRG':
            gb = raw_data[0::2, 0::2].astype(np.float32)
            gr = raw_data[1::2, 1::2].astype(np.float32)
        else:
            return raw_data
        
        # 计算两个绿色通道的平均值
        min_h = min(gr.shape[0], gb.shape[0])
        min_w = min(gr.shape[1], gb.shape[1])
        
        gr_crop = gr[:min_h, :min_w]
        gb_crop = gb[:min_h, :min_w]
        
        g_mean = (gr_crop + gb_crop) / 2.0
        
        # 均衡化
        gr_balanced = gr_crop * (1 - balance_strength) + g_mean * balance_strength
        gb_balanced = gb_crop * (1 - balance_strength) + g_mean * balance_strength
        
        # 写回
        if bayer_pattern == 'RGGB':
            denoised[0::2, 1::2][:min_h, :min_w] = gr_balanced
            denoised[1::2, 0::2][:min_h, :min_w] = gb_balanced
        elif bayer_pattern == 'BGGR':
            denoised[0::2, 1::2][:min_h, :min_w] = gb_balanced
            denoised[1::2, 0::2][:min_h, :min_w] = gr_balanced
        elif bayer_pattern == 'GRBG':
            denoised[0::2, 0::2][:min_h, :min_w] = gr_balanced
            denoised[1::2, 1::2][:min_h, :min_w] = gb_balanced
        elif bayer_pattern == 'GBRG':
            denoised[0::2, 0::2][:min_h, :min_w] = gb_balanced
            denoised[1::2, 1::2][:min_h, :min_w] = gr_balanced
        
        return denoised.astype(raw_data.dtype)
    
    def _adaptive_raw_denoise(self, raw_data: np.ndarray, bayer_pattern: str,
                             base_strength: float = 1.0, edge_threshold: float = 50) -> np.ndarray:
        """
        自适应Raw降噪 - 根据局部梯度调整降噪强度
        平坦区域强降噪，边缘区域弱降噪
        
        Args:
            bayer_pattern: Bayer模式
            base_strength: 基础降噪强度
            edge_threshold: 边缘检测阈值
        """
        # 归一化
        if raw_data.dtype == np.uint16:
            data_float = raw_data.astype(np.float32) / 65535.0
        else:
            data_float = raw_data.astype(np.float32) / 255.0
        
        # 计算梯度幅值（边缘强度）
        grad_x = np.abs(np.diff(data_float, axis=1, prepend=data_float[:, :1]))
        grad_y = np.abs(np.diff(data_float, axis=0, prepend=data_float[:1, :]))
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化边缘强度到 [0, 1]
        edge_strength = edge_strength / (edge_strength.max() + 1e-6)
        
        # 自适应权重：边缘区域权重低（少降噪），平坦区域权重高（多降噪）
        denoise_weight = 1.0 - np.clip(edge_strength * edge_threshold, 0, 1)
        
        # 高斯滤波
        denoised = gaussian_filter(data_float, sigma=base_strength)
        
        # 混合原始和降噪结果
        result = data_float * (1 - denoise_weight) + denoised * denoise_weight
        
        # 转换回原始数据类型
        if raw_data.dtype == np.uint16:
            return (result * 65535).astype(np.uint16)
        else:
            return (result * 255).astype(np.uint8)
    
    def _wavelet_raw_denoise(self, raw_data: np.ndarray, wavelet: str = 'db1',
                            level: int = 2, threshold_scale: float = 1.0) -> np.ndarray:
        """
        小波降噪 - 多尺度分析
        
        Args:
            wavelet: 小波基类型
            level: 分解层数
            threshold_scale: 阈值缩放因子
        """
        try:
            import pywt
        except ImportError:
            print("Warning: pywt not installed, falling back to bilateral filter")
            return self._bilateral_raw(raw_data)
        
        # 归一化
        if raw_data.dtype == np.uint16:
            data_float = raw_data.astype(np.float32) / 65535.0
        else:
            data_float = raw_data.astype(np.float32) / 255.0
        
        # 小波分解
        coeffs = pywt.wavedec2(data_float, wavelet, level=level)
        
        # 估计噪声标准差（使用最高频子带）
        sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
        
        # 计算阈值
        threshold = sigma * threshold_scale * np.sqrt(
            2 * np.log(data_float.shape[0] * data_float.shape[1])
        )
        
        # 软阈值处理高频系数
        coeffs_denoised = [coeffs[0]]
        for detail in coeffs[1:]:
            coeffs_denoised.append(
                tuple(pywt.threshold(d, threshold, mode='soft') for d in detail)
            )
        
        # 小波重构
        denoised = pywt.waverec2(coeffs_denoised, wavelet)
        
        # 裁剪到原始大小（小波变换可能改变尺寸）
        denoised = denoised[:raw_data.shape[0], :raw_data.shape[1]]
        
        # 转换回原始数据类型
        if raw_data.dtype == np.uint16:
            return (np.clip(denoised, 0, 1) * 65535).astype(np.uint16)
        else:
            return (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    
    # ==================== 辅助函数 ====================
    
    def _extract_bayer_channels(self, raw_data: np.ndarray, 
                               bayer_pattern: str) -> dict:
        """
        从Bayer Raw数据中提取R/Gr/Gb/B四个通道
        
        Returns:
            字典: {'R': (data, mask), 'Gr': (data, mask), 'Gb': (data, mask), 'B': (data, mask)}
        """
        h, w = raw_data.shape
        
        # 根据Bayer模式定义位置
        patterns = {
            'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
            'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
            'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
            'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
        }
        
        if bayer_pattern not in patterns:
            raise ValueError(f"Unknown Bayer pattern: {bayer_pattern}")
        
        pattern = patterns[bayer_pattern]
        channels = {}
        
        for color, (row_offset, col_offset) in pattern.items():
            # 提取子通道
            channel_data = raw_data[row_offset::2, col_offset::2]
            
            # 创建mask
            mask = np.zeros((h, w), dtype=bool)
            mask[row_offset::2, col_offset::2] = True
            
            channels[color] = (channel_data, mask)
        
        return channels
    
    def _reconstruct_bayer(self, channels: dict, bayer_pattern: str, 
                          h: int, w: int) -> np.ndarray:
        """
        从分离的通道重构Bayer Raw数据
        
        Args:
            channels: 字典包含处理后的通道数据
            bayer_pattern: Bayer模式
            h, w: 输出图像尺寸
        
        Returns:
            重构的Bayer Raw数据
        """
        reconstructed = np.zeros((h, w), dtype=np.float32)
        
        patterns = {
            'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
            'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
            'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
            'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
        }
        
        pattern = patterns[bayer_pattern]
        
        for color, (row_offset, col_offset) in pattern.items():
            channel_data = channels[color]
            ch_h = (h - row_offset + 1) // 2
            ch_w = (w - col_offset + 1) // 2
            reconstructed[row_offset::2, col_offset::2] = channel_data[:ch_h, :ch_w]
        
        return reconstructed