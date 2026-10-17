from __future__ import annotations

import json

import torch

from lobdl.models import count_parameters
from lobdl.neural import M1_ANCHOR, build_model


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    configurations = {
        "M1": M1_ANCHOR,
        "M2": {"learning_rate": 1e-3, "layers": 3, "dropout": 0.1},
    }
    report = {}
    for name, parameters in configurations.items():
        model = build_model(name, parameters).to(device)
        inputs = torch.zeros(2, 100, 40, device=device)
        loss = model(inputs).sum()
        loss.backward()
        report[name] = {
            "output_shape": list(model(inputs).shape),
            "trainable_parameters": count_parameters(model),
        }
    report["device"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
