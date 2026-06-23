import os
import struct
from pathlib import Path

import torch
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
WTS_PATH = MODELS_DIR / "squeezenet.wts"


def print_cache_env() -> None:
    for key in ("TORCH_HOME", "HF_HOME"):
        value = os.environ.get(key)
        if value:
            print(f"{key}={value}")


def export_wts(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    with open(path, "w") as f:
        f.write(f"{len(state_dict)}\n")
        for key, value in state_dict.items():
            values = value.reshape(-1).cpu().numpy()
            f.write(f"{key} {len(values)}")
            print(key, value.shape)
            for item in values:
                f.write(" ")
                f.write(struct.pack(">f", float(item)).hex())
            f.write("\n")


def main() -> None:
    print_cache_env()
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    model = model.eval()
    export_wts(model, WTS_PATH)


if __name__ == "__main__":
    main()
