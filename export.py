import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.onnx
import sys

sys.path.append("Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2")

    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
    )
    parser.add_argument("--output", type=str, default="models")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(
            f"Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth",
            map_location="cpu",
        )
    )
    depth_anything = depth_anything.to("cpu").eval()
    # Define dummy input data
    dummy_input = torch.ones((3, args.input_size, args.input_size)).unsqueeze(0)

    onnx_path = os.path.join(args.output, f"depth_anything_v2_{args.encoder}.onnx")

    torch.onnx.export(
        depth_anything,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
    )

    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    main()
