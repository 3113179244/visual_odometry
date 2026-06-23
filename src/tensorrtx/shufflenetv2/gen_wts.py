import argparse
import os
import struct
from pathlib import Path

import torch
from torchvision import models


MODEL_NAMES = (
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODELS_DIR = REPO_ROOT / "models"


def require_cache_env() -> None:
    missing = [name for name in ("TORCH_HOME", "HF_HOME") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Please set required cache environment variables: {', '.join(missing)}"
        )
    print(f"Using TORCH_HOME={os.environ['TORCH_HOME']}")
    print(f"Using HF_HOME={os.environ['HF_HOME']}")


def build_model(name: str) -> torch.nn.Module:
    suffix = name.removeprefix("shufflenet_").upper()
    weights_cls = getattr(models, f"ShuffleNet_{suffix}_Weights")
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
            values = value.reshape(-1).cpu().numpy()
            f.write(f"{key} {len(values)}")
            for item in values:
                f.write(" ")
                f.write(struct.pack(">f", float(item)).hex())
            f.write("\n")
    size_mib = output_path.stat().st_size / 1024 / 1024
    print(f"[ok] wrote {output_path} ({size_mib:.1f} MiB, {len(state_dict)} tensors)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export torchvision ShuffleNetV2 weights to tensorrtx .wts files."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        action="append",
        help="ShuffleNetV2 variant to export. Repeat for multiple variants. Defaults to all variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for generated .wts files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cache_env()
    model_names = args.model or MODEL_NAMES
    for model_name in model_names:
        print(f"writing {model_name}.wts")
        write_wts(build_model(model_name), args.output_dir / f"{model_name}.wts")


if __name__ == "__main__":
    main()
