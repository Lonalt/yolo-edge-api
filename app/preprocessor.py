from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessorConfig:
    target_size: tuple[int, int] = (640, 640)
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    denoise: bool = False
    normalize: bool = False
    color_space: str = "RGB"

# Configurações obrigatórias exigidas pelo critério
CONFIG_DEFAULT = PreprocessorConfig(
    target_size=(640, 640),
    apply_clahe=False,
    denoise=False,
    normalize=False,
    color_space="RGB"
)

CONFIG_LOW_LIGHT = PreprocessorConfig(
    target_size=(640, 640),
    apply_clahe=True,
    clahe_clip_limit=3.0,
    clahe_tile_grid=(8, 8),
    denoise=True,
    normalize=False,
    color_space="RGB"
)

CONFIG_HIGH_QUALITY = PreprocessorConfig(
    target_size=(1280, 1280),
    apply_clahe=True,
    clahe_clip_limit=1.5,
    clahe_tile_grid=(8, 8),
    denoise=False,
    normalize=True,
    color_space="RGB"
)

class ImagePreprocessor:
    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or CONFIG_DEFAULT

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Entrada invalida. Esperado numpy.ndarray.")

        processed = image.copy()

        # Denoise (suavização)
        if self.config.denoise:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # CLAHE (ajuste de contraste para baixa luminosidade)
        if self.config.apply_clahe:
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=self.config.clahe_tile_grid
                )
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif len(processed.shape) == 2:
                clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=self.config.clahe_tile_grid
                )
                processed = clahe.apply(processed)

        # Espaço de cor
        if self.config.color_space.upper() == "RGB" and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Redimensionamento
        if self.config.target_size:
            processed = cv2.resize(
                processed,
                self.config.target_size,
                interpolation=cv2.INTER_LINEAR
            )

        # Normalização opcional
        if self.config.normalize:
            processed = processed.astype(np.float32) / 255.0

        return processed
