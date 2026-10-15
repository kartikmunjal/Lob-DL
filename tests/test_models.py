import torch

from lobdl.models import DeepLOB, TransformerLOB, count_parameters


def test_models_accept_locked_shape():
    x = torch.zeros(2, 100, 40)
    for model in (DeepLOB(), TransformerLOB()):
        assert model(x).shape == (2, 3)
        assert count_parameters(model) > 0

