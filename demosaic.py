# demosaic.py (最终混合版本)

import numpy as np
import colour_demosaicing as cdm
import cv2
import tifffile
import rawpy
import io

class Demosaic:
    """
    去马赛克处理模块（混合版本）。
    - 支持处理无头信息的'裸'.raw格式图像。
    - 通过动态创建内存中的TIFF文件，重新启用了rawpy后端，以支持AHD等高级算法。
    - 同时保留了colour-demosaicing和OpenCV作为备选方案。
    """
    def __init__(self, width: int, height: int, bayer_pattern: str, dtype: np.dtype = np.uint8):
        """
        初始化模块，必须提供RAW图像的元数据。
        """
        self.width = width
        self.height = height
        self.bayer_pattern = bayer_pattern.upper()
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        print(f"Demosaic模块已初始化: W={width}, H={height}, Pattern={self.bayer_pattern}, DType={self.dtype}")

    def _read_raw_file(self, file_path: str) -> np.ndarray:
        """从裸.raw文件中读取并重塑为2D Bayer阵列"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            expected_size = self.width * self.height * self.itemsize
            if len(raw_data) != expected_size:
                raise ValueError(f"文件大小 ({len(raw_data)}) 与预期尺寸 ({expected_size}) 不符。")
            bayer_array = np.frombuffer(raw_data, dtype=self.dtype)
            return bayer_array.reshape((self.height, self.width))
        except Exception as e:
            print(f"读取或解析RAW文件时出错: {e}")
            raise

    def _demosaic_with_rawpy(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用rawpy后端进行去马赛克"""
        print(f"  - 使用 rawpy 后端 (算法: {algorithm})")
        # 1. 定义CFA Pattern的数字映射和黑白电平
        # 0=Red, 1=Green, 2=Blue

        BLACK_LEVEL = 1024 if self.dtype == np.uint16 else 16
        WHITE_LEVEL = np.iinfo(self.dtype).max # 使用数据类型的最大值作为白电平
        black_level_4_channels = (BLACK_LEVEL,) * 4

        pattern_map = {
            'RGGB': (0, 1, 1, 2),
            'GRBG': (1, 0, 2, 1),
            'GBRG': (1, 2, 0, 1),
            'BGGR': (2, 1, 1, 0),
        }
        cfa_pattern = pattern_map.get(self.bayer_pattern)  # 对一个字典，返回其键对应的值
        if cfa_pattern is None:
            raise ValueError(f"不支持的Bayer Pattern: {self.bayer_pattern}")

        # 2. 内存里构造一个带 CFA 标签的TIFF，再交给 rawpy 读取，从而让 rawpy 能够识别并调用其内置的去马赛克算法
        with io.BytesIO() as tiff_buffer:
            with tifffile.TiffWriter(tiff_buffer, bigtiff=False) as tif:
                tif.write(
                    bayer_array,
                    photometric='cfa',
                    # 定义CFA Pattern的标签
                    extratags=[
                        (33421, 'H', 2, (2, 2)), # CFARepeatPatternDim（描述 CFA pattern 的重复维度）
                        (33422, 'B', 4, cfa_pattern), # CFAPattern（CFA排列）
                        (37380, 'H', 4, black_level_4_channels), # BlackLevel
                        (37384, 'H', 1, WHITE_LEVEL),        # WhiteLevel tag code
                    ]
                )
            
            # 3. 让rawpy从内存缓冲区中读取这个TIFF文件
            tiff_buffer.seek(0)
            with rawpy.imread(tiff_buffer) as raw:
                # 4. 使用rawpy的postprocess方法和指定算法
                algo_map = {
                    'AHD': rawpy.DemosaicAlgorithm.AHD,
                    'LMMSE': rawpy.DemosaicAlgorithm.LMMSE,
                    'EA': rawpy.DemosaicAlgorithm.EA,
                }
                rgb_image = raw.postprocess(
                    demosaic_algorithm=algo_map[algorithm.upper()],
                    use_camera_wb=False,
                    no_auto_bright=True,
                    output_bps=16 if self.dtype == np.uint16 else 8
                )
        return rgb_image

    def _demosaic_with_colour(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用colour-demosaicing后端"""
        print(f"  - 使用 colour-demosaicing 后端 (算法: {algorithm})")
        max_val = np.iinfo(self.dtype).max
        bayer_float = bayer_array.astype(np.float64) / max_val

        if algorithm.lower() == 'bilinear':
            rgb_float = cdm.demosaicing_CFA_Bayer_bilinear(bayer_float, pattern=self.bayer_pattern)
        elif algorithm.lower() == 'malvar2004':
            rgb_float = cdm.demosaicing_CFA_Bayer_Malvar2004(bayer_float, pattern=self.bayer_pattern)
        elif algorithm.lower() == 'menon2007':
            rgb_float = cdm.demosaicing_CFA_Bayer_Menon2007(bayer_float, pattern=self.bayer_pattern)
        else:
            raise ValueError("内部错误：不应由此函数处理的算法。")
            
        return np.clip(rgb_float * max_val, 0, max_val).astype(self.dtype)
        
    def _demosaic_with_cv2(self, bayer_array: np.ndarray, algorithm: str) -> np.ndarray:
        """使用OpenCV后端"""
        print(f"  - 使用 OpenCV 后端 (算法: {algorithm})")
        pattern_map = {
            'RGGB': cv2.COLOR_BAYER_RG2RGB,
            'GRBG': cv2.COLOR_BAYER_GR2RGB,
            'GBRG': cv2.COLOR_BAYER_GB2RGB,
            'BGGR': cv2.COLOR_BAYER_BG2RGB,
        }
        return cv2.cvtColor(bayer_array, pattern_map[self.bayer_pattern])

    def execute(self, raw_file_path: str, algorithm: str = 'AHD') -> np.ndarray:
        """
        执行去马赛克操作，并根据算法自动选择后端。

        Args:
            raw_file_path (str): 裸.raw文件的路径。
            algorithm (str):
              - rawpy后端: 'AHD', 'LMMSE', 'EA'
              - colour-demosaicing后端: 'Bilinear', 'Malvar2004', 'Menon2007'
              - OpenCV后端: 'cv_VNG'
        """
        print(f"Executing Demosaic from file: {raw_file_path} with algorithm: {algorithm}")
        
        # 定义算法归属
        RAWPY_ALGOS = ['AHD', 'LMMSE', 'EA']
        COLOUR_ALGOS = ['BILINEAR', 'MALVAR2004', 'MENON2007']
        CV2_ALGOS = ['CV']

        # 1. 读取裸RAW文件为2D Bayer阵列
        bayer_array = self._read_raw_file(raw_file_path)

        # 2. 根据算法选择合适的处理函数
        algo_upper = algorithm.upper()
        if algo_upper in RAWPY_ALGOS:
            return self._demosaic_with_rawpy(bayer_array, algo_upper)
        elif algo_upper in COLOUR_ALGOS:
            return self._demosaic_with_colour(bayer_array, algo_upper)
        elif algo_upper in CV2_ALGOS:
            return self._demosaic_with_cv2(bayer_array, algo_upper)
        else:
            raise ValueError(f"不支持的去马赛克算法: {algorithm}")