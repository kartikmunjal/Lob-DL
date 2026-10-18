import torch

from lobdl.models import DeepLOB, TransformerLOB, count_parameters


def test_models_accept_locked_shape():
    x = torch.zeros(2, 100, 40)
    for model in (DeepLOB(), TransformerLOB()):
        assert model(x).shape == (2, 3)
        assert count_parameters(model) > 0


def test_depth_adapters_preserve_output_shape():
    for levels in (1, 5):
        x = torch.zeros(2, 100, levels * 4)
        for model in (DeepLOB(levels=levels), TransformerLOB(features=levels * 4)):
            assert model(x).shape == (2, 3)
