import pytest
import numpy as np
from app.preprocessor import (
    ImagePreprocessor,
    CONFIG_DEFAULT,
    CONFIG_LOW_LIGHT,
    CONFIG_HIGH_QUALITY,
    PreprocessorConfig
)

@pytest.fixture
def dummy_image():
    # Gera imagem sintética BGR 480x640x3
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

def test_config_constants_exist():
    """Teste 1: Valida se as 3 configuracoes foram declaradas corretamente."""
    assert isinstance(CONFIG_DEFAULT, PreprocessorConfig)
    assert isinstance(CONFIG_LOW_LIGHT, PreprocessorConfig)
    assert isinstance(CONFIG_HIGH_QUALITY, PreprocessorConfig)

def test_default_config_dimensions(dummy_image):
    """Teste 2: Valida dimensao de saida da configuracao DEFAULT (640x640)."""
    p = ImagePreprocessor(CONFIG_DEFAULT)
    out = p.process(dummy_image)
    assert out.shape == (640, 640, 3)

def test_high_quality_dimensions(dummy_image):
    """Teste 3: Valida dimensao de saida da configuracao HIGH_QUALITY (1280x1280)."""
    p = ImagePreprocessor(CONFIG_HIGH_QUALITY)
    out = p.process(dummy_image)
    assert out.shape == (1280, 1280, 3)

def test_high_quality_normalization(dummy_image):
    """Teste 4: Valida se a normalizacao converte valores para float entre 0.0 e 1.0."""
    p = ImagePreprocessor(CONFIG_HIGH_QUALITY)
    out = p.process(dummy_image)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0

def test_low_light_clahe_execution(dummy_image):
    """Teste 5: Valida que a configuracao LOW_LIGHT altera a distribuicao de luminancia."""
    p = ImagePreprocessor(CONFIG_LOW_LIGHT)
    out = p.process(dummy_image)
    assert out.shape == (640, 640, 3)
    assert not np.array_equal(out, dummy_image)

def test_invalid_input_raises_value_error():
    """Teste 6: Valida levantamento de excecao para entradas nulas ou invalidas."""
    p = ImagePreprocessor(CONFIG_DEFAULT)
    with pytest.raises(ValueError):
        p.process(None)
    with pytest.raises(ValueError):
        p.process("caminho/invalido/imagem.jpg")

def test_custom_config_grayscale():
    """Teste 7: Valida configuracao customizada sem quebrar o pipeline."""
    custom = PreprocessorConfig(target_size=(320, 320), color_space="BGR")
    p = ImagePreprocessor(custom)
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    out = p.process(gray)
    assert out.shape == (320, 320)

def test_color_space_transformation(dummy_image):
    """Teste 8: Valida se canais sao convertidos corretamente de BGR para RGB."""
    solid_blue_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    solid_blue_bgr[:, :, 0] = 255  # Azul no BGR
    p = ImagePreprocessor(CONFIG_DEFAULT)
    out = p.process(solid_blue_bgr)
    # No RGB, canal azul e o indice 2
    assert out[0, 0, 2] == 255
    assert out[0, 0, 0] == 0
