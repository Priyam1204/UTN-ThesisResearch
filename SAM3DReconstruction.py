# demo.py

import sys
import argparse
from pathlib import Path

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image


def BuildArgParser():
    Parser = argparse.ArgumentParser(
        description="SAM3D reconstruction: image + mask → .ply"
    )
    Parser.add_argument("--image_path", required=True)
    Parser.add_argument("--mask_path", required=True)
    Parser.add_argument("--output_path", required=True)
    return Parser


def Main():
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ImagePath = Path(Args.image_path)
    MaskPath = Path(Args.mask_path)

    if not ImagePath.exists():
        raise FileNotFoundError(f"Image not found: {ImagePath}")

    if not MaskPath.exists():
        raise FileNotFoundError(f"Mask not found: {MaskPath}")

    # ---- model (fixed hf, same as original) ----
    config_path = "checkpoints/hf/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # ---- load inputs ----
    image = load_image(str(ImagePath))
    mask = load_image(str(MaskPath))  # using mask directly

    # ---- run ----
    output = inference(image, mask, seed=42)

    # ---- save ----
    output["gs"].save_ply(Args.output_path)

    print(f"Reconstruction saved to {Args.output_path}")


if __name__ == "__main__":
    Main()