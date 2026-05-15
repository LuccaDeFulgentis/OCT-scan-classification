import torch
from src.model import OCTModel

def test_model_instantiation():
    model = OCTModel(num_classes=4)
    assert model is not None

def test_model_forward_pass():
    model = OCTModel(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 4)