import argparse
import os
import struct
from pathlib import Path

import torch
from torchvision import models


MODEL_NAMES = (
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
)


def require_cache_env() -> None:
    missing = [name for name in ("TORCH_HOME", "HF_HOME") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Please set required cache environment variables: {', '.join(missing)}"
        )
    print(f"Using TORCH_HOME={os.environ['TORCH_HOME']}")
    print(f"Using HF_HOME={os.environ['HF_HOME']}")


def build_model(name: str) -> torch.nn.Module:
    weights_cls = getattr(models, f"{name.upper()}_Weights")
    model_fn = getattr(models, name)
    model = model_fn(weights=weights_cls.DEFAULT)
    model.eval()
    return model


def write_wts(model: torch.nn.Module, output_path: Path) -> None:
    state_dict = model.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(f"{len(state_dict.keys())}\n")
        for key, value in state_dict.items():
            print(f"key: {key}\tvalue: {value.shape}")
            values = value.reshape(-1).cpu().numpy()
            f.write(f"{key} {len(values)}")
            for item in values:
                f.write(" ")
                f.write(struct.pack(">f", float(item)).hex())
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export torchvision VGG weights to tensorrtx .wts files."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        action="append",
        help="VGG variant to export. Repeat the option for multiple variants. Defaults to all variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../models"),
        help="Directory for generated .wts files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    require_cache_env()
    model_names = args.model or MODEL_NAMES
    for model_name in model_names:
        print(f"writing {model_name}.wts")
        write_wts(build_model(model_name), args.output_dir / f"{model_name}.wts")
