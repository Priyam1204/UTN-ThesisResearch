# bbox_to_mask_stage.py

import os
import json
import cv2
import numpy as np
import mimetypes
import argparse
from pathlib import Path
from typing import List
from PIL import Image

from google import genai
from segment_anything import sam_model_registry, SamPredictor

# --- Constants ---
DefaultModelName = "gemini-robotics-er-1.5-preview"
DefaultDevice = "cuda"
DefaultSamModelType = "vit_h"

BBoxPrompt = """Detect the bounding box of the following object:

"{object_description}"

Return ONLY JSON:
{
  "bbox": [ymin, xmin, ymax, xmax]
}

Values must be normalized between 0 and 1.
"""

# --- Core Functions ---

def GeminiBBOX(
    Client: genai.Client, Description: str, ImagePath: Path
) -> List[float]:
    """Calls Gemini to get a normalized bounding box for an object in an image."""
    with open(ImagePath, "rb") as FileHandle:
        ImageBytes = FileHandle.read()

    MimeType, _ = mimetypes.guess_type(str(ImagePath))
    MimeType = MimeType or "image/jpeg"

    Prompt = BBoxPrompt.format(object_description=Description)

    Response = Client.models.generate_content(
        model=DefaultModelName,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": Prompt},
                    {"inline_data": {"mime_type": MimeType, "data": ImageBytes}},
                ],
            }
        ],
    )

    RawResponse = Response.text or ""

    StartIndex = RawResponse.find("{")
    EndIndex = RawResponse.rfind("}")
    if StartIndex == -1 or EndIndex == -1:
        raise RuntimeError("No JSON found in Gemini bbox response.")

    BBoxJson = json.loads(RawResponse[StartIndex : EndIndex + 1])
    BBox = BBoxJson.get("bbox")

    if not BBox or len(BBox) != 4:
        raise RuntimeError("Invalid bbox format returned by Gemini.")

    return BBox


def BinarySegmentation(
    ImagePath: Path,
    SamCheckpointPath: Path,
    BboxNorm: List[float],
) -> np.ndarray:
    """Generates a binary mask from a normalized bounding box using SAM."""
    ImageObject = Image.open(ImagePath)
    ImageWidth, ImageHeight = ImageObject.size

    ymin, xmin, ymax, xmax = BboxNorm
    InputBoxPx = np.array([
        int(xmin * ImageWidth),
        int(ymin * ImageHeight),
        int(xmax * ImageWidth),
        int(ymax * ImageHeight),
    ])

    SamModel = sam_model_registry[DefaultSamModelType](checkpoint=str(SamCheckpointPath))
    SamModel.to(device=DefaultDevice)
    Predictor = SamPredictor(SamModel)

    ImageArray = cv2.imread(str(ImagePath))
    if ImageArray is None:
        raise RuntimeError(f"Could not read image with OpenCV: {ImagePath}")

    ImageRgb = cv2.cvtColor(ImageArray, cv2.COLOR_BGR2RGB)
    Predictor.set_image(ImageRgb)

    Masks, Scores, _ = Predictor.predict(box=InputBoxPx, multimask_output=True)

    BestMask = Masks[np.argmax(Scores)]
    return BestMask


# --- CLI and Orchestration ---

def BuildArgParser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for the script."""
    Parser = argparse.ArgumentParser(
        description="Stage 2: Use stage-1 JSON and an image to generate a binary mask."
    )
    Parser.add_argument("--image_path", type=str, required=True, help="Input image path.")
    Parser.add_argument(
        "--stage1_json_path", type=str, required=True, help="Path to stage-1 JSON output."
    )
    Parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="Path to the SAM checkpoint file."
    )
    Parser.add_argument(
        "--output_mask_path", type=str, required=True, help="Path to save the output binary mask."
    )
    return Parser


def AutomaticSegmentation() -> None:
    """Main function to orchestrate the bounding box to mask pipeline."""
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ApiKey = os.getenv("GEMINI_API_KEY")
    if not ApiKey:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")

    ImagePath = Path(Args.image_path)
    if not ImagePath.exists():
        raise FileNotFoundError(f"Image not found: {ImagePath}")

    JsonPath = Path(Args.stage1_json_path)
    if not JsonPath.exists():
        raise FileNotFoundError(f"Stage-1 JSON not found: {JsonPath}")

    SamCheckpointPath = Path(Args.sam_checkpoint)
    if not SamCheckpointPath.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SamCheckpointPath}")

    # 1. Get object description from Stage 1 JSON
    with open(JsonPath, "r", encoding="utf-8") as FileHandle:
        StageOneJson = json.load(FileHandle)
    Description = StageOneJson.get("Description")
    if not Description:
        raise RuntimeError("A 'Description' key was not found in the stage-1 JSON.")

    # 2. Get bounding box from Gemini
    Client = genai.Client(api_key=ApiKey)
    print("Requesting bounding box from Gemini...")
    BboxNorm = GeminiBBOX(Client, Description, ImagePath)
    print("Received bounding box.")

    # 3. Generate mask from bounding box using SAM
    print("Generating mask with SAM...")
    BinaryMask = BinarySegmentation(ImagePath, SamCheckpointPath, BboxNorm)
    print("Mask generated.")

    # 4. Save the final mask
    OutputMaskPath = Path(Args.output_mask_path)
    OutputMaskPath.parent.mkdir(parents=True, exist_ok=True)
    MaskImage = Image.fromarray((BinaryMask * 255).astype(np.uint8))
    MaskImage.save(OutputMaskPath)

    print(f"\nBinary mask saved to: {OutputMaskPath}")


if __name__ == "__main__":
    AutomaticSegmentation()