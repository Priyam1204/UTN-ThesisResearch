# automatic_nonusable_segmentation.py

import os
import json
import cv2
import numpy as np
import mimetypes
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from google import genai
from segment_anything import sam_model_registry, SamPredictor

# --- Constants ---
DefaultModelName = "gemini-robotics-er-1.5-preview"
DefaultDevice = "cuda"
DefaultSamModelType = "vit_h"

NonUsableRegionPrompt = """You are an expert in robotic manipulation and object interaction.

Given an RGB image of an object, identify all visible regions that are NOT suitable for physical interaction, including grasping, pushing, or applying force.

Non-usable regions may include:
- Fragile parts
- Functional components
- Slippery or unstable surfaces

Tasks:
1. Write a short description of the object.
2. Detect all visually identifiable non-usable regions.
3. For each region, return:
   - a short noun phrase suitable as an OWLv2 text prompt
   - one normalized bounding box in the format [ymin, xmin, ymax, xmax]

Rules:
- Bounding box values must be normalized between 0 and 1.
- Use short, specific noun phrases only.
- Do not use vague descriptions.
- Include only regions that are clearly visible in the image.
- Return all relevant regions.
- Return ONLY valid JSON.

Output format:
{
  "Description": "short object description",
  "NonUsableRegions": [
    {
      "Phrase": "glass panel",
      "Bbox": [ymin, xmin, ymax, xmax]
    }
  ]
}
"""


def GetNonUsableRegionsFromGemini(
    Client: genai.Client,
    ImagePath: Path,
) -> Dict[str, Any]:
    """Calls Gemini to get object description and multiple non-usable regions with bboxes."""
    with open(ImagePath, "rb") as FileHandle:
        ImageBytes = FileHandle.read()

    MimeType, _ = mimetypes.guess_type(str(ImagePath))
    MimeType = MimeType or "image/jpeg"

    Response = Client.models.generate_content(
        model=DefaultModelName,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": NonUsableRegionPrompt},
                    {"inline_data": {"mime_type": MimeType, "data": ImageBytes}},
                ],
            }
        ],
    )

    RawResponse = Response.text or ""

    StartIndex = RawResponse.find("{")
    EndIndex = RawResponse.rfind("}")
    if StartIndex == -1 or EndIndex == -1:
        raise RuntimeError("No JSON found in Gemini response.")

    ParsedJson = json.loads(RawResponse[StartIndex : EndIndex + 1])

    Description = ParsedJson.get("Description")
    NonUsableRegions = ParsedJson.get("NonUsableRegions")

    if not isinstance(Description, str) or not Description.strip():
        raise RuntimeError("Invalid or missing 'Description' in Gemini response.")

    if not isinstance(NonUsableRegions, list):
        raise RuntimeError("Invalid or missing 'NonUsableRegions' in Gemini response.")

    for Region in NonUsableRegions:
        if not isinstance(Region, dict):
            raise RuntimeError("Each non-usable region must be a JSON object.")

        Phrase = Region.get("Phrase")
        Bbox = Region.get("Bbox")

        if not isinstance(Phrase, str) or not Phrase.strip():
            raise RuntimeError(f"Invalid Phrase in region: {Region}")

        if not isinstance(Bbox, list) or len(Bbox) != 4:
            raise RuntimeError(f"Invalid Bbox format in region: {Region}")

    return ParsedJson


def ClampBoxToImageBounds(BoxXYXY: np.ndarray, ImageWidth: int, ImageHeight: int) -> np.ndarray:
    """Clamps a box [xmin, ymin, xmax, ymax] to valid image bounds."""
    
    xmin, ymin, xmax, ymax = BoxXYXY

    xmin = max(0, min(xmin, ImageWidth - 1))
    xmax = max(0, min(xmax, ImageWidth - 1))
    ymin = max(0, min(ymin, ImageHeight - 1))
    ymax = max(0, min(ymax, ImageHeight - 1))

    return np.array([xmin, ymin, xmax, ymax], dtype=np.int32)


def ConvertNormalizedBoxToPixels(
    BboxNorm: List[float],
    ImageWidth: int,
    ImageHeight: int,
) -> np.ndarray:
    """Converts normalized [ymin, xmin, ymax, xmax] box to pixel [xmin, ymin, xmax, ymax]."""
    ymin, xmin, ymax, xmax = BboxNorm

    InputBoxPx = np.array([
        int(round(xmin * ImageWidth)),
        int(round(ymin * ImageHeight)),
        int(round(xmax * ImageWidth)),
        int(round(ymax * ImageHeight)),
    ], dtype=np.int32)

    InputBoxPx = ClampBoxToImageBounds(InputBoxPx, ImageWidth, ImageHeight)

    xmin_px, ymin_px, xmax_px, ymax_px = InputBoxPx
    if xmax_px <= xmin_px or ymax_px <= ymin_px:
        raise RuntimeError(f"Invalid bbox after conversion/clamping: {BboxNorm}")

    return InputBoxPx


def BuildSamPredictor(SamCheckpointPath: Path) -> SamPredictor:
    """Loads SAM and returns a predictor."""
    SamModel = sam_model_registry[DefaultSamModelType](checkpoint=str(SamCheckpointPath))
    SamModel.to(device=DefaultDevice)
    return SamPredictor(SamModel)


def GetUnifiedMaskFromRegions(
    ImagePath: Path,
    SamCheckpointPath: Path,
    NonUsableRegions: List[Dict[str, Any]],
) -> np.ndarray:
    """Generates a unified binary mask from multiple non-usable region bounding boxes using SAM."""
    ImageObject = Image.open(ImagePath)
    ImageWidth, ImageHeight = ImageObject.size

    ImageArray = cv2.imread(str(ImagePath))
    if ImageArray is None:
        raise RuntimeError(f"Could not read image with OpenCV: {ImagePath}")

    ImageRgb = cv2.cvtColor(ImageArray, cv2.COLOR_BGR2RGB)

    Predictor = BuildSamPredictor(SamCheckpointPath)
    Predictor.set_image(ImageRgb)

    UnifiedMask = np.zeros((ImageHeight, ImageWidth), dtype=bool)

    for Region in NonUsableRegions:
        BboxNorm = Region["Bbox"]
        InputBoxPx = ConvertNormalizedBoxToPixels(BboxNorm, ImageWidth, ImageHeight)

        Masks, Scores, _ = Predictor.predict(box=InputBoxPx, multimask_output=True)
        BestMask = Masks[np.argmax(Scores)]

        UnifiedMask = np.logical_or(UnifiedMask, BestMask)

    return UnifiedMask


def SaveMask(OutputMaskPath: Path, BinaryMask: np.ndarray) -> None:
    """Saves a binary mask as an 8-bit image."""
    OutputMaskPath.parent.mkdir(parents=True, exist_ok=True)
    MaskImage = Image.fromarray((BinaryMask.astype(np.uint8) * 255))
    MaskImage.save(OutputMaskPath)


def BuildArgParser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for the script."""
    Parser = argparse.ArgumentParser(
        description="Detect non-usable object regions with Gemini, segment them with SAM, and save one unified binary mask."
    )
    Parser.add_argument("--image_path", type=str, required=True, help="Input RGB image path.")
    Parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to the SAM checkpoint file.",
    )
    Parser.add_argument(
        "--output_mask_path",
        type=str,
        required=True,
        help="Path to save the final unified binary mask.",
    )
    Parser.add_argument(
        "--output_json_path",
        type=str,
        default="",
        help="Optional path to save Gemini JSON output.",
    )
    return Parser


def RunPipeline() -> None:
    """Main function to orchestrate the non-usable region segmentation pipeline."""
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ApiKey = os.getenv("GEMINI_API_KEY")
    if not ApiKey:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")

    ImagePath = Path(Args.image_path)
    if not ImagePath.exists():
        raise FileNotFoundError(f"Image not found: {ImagePath}")

    SamCheckpointPath = Path(Args.sam_checkpoint)
    if not SamCheckpointPath.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SamCheckpointPath}")

    OutputMaskPath = Path(Args.output_mask_path)
    OutputJsonPath = Path(Args.output_json_path) if Args.output_json_path else None

    Client = genai.Client(api_key=ApiKey)

    print("Requesting non-usable regions from Gemini...")
    GeminiOutput = GetNonUsableRegionsFromGemini(Client, ImagePath)

    Description = GeminiOutput["Description"]
    NonUsableRegions = GeminiOutput["NonUsableRegions"]

    print(f"Object description: {Description}")
    print(f"Detected {len(NonUsableRegions)} non-usable region(s).")

    if OutputJsonPath is not None:
        OutputJsonPath.parent.mkdir(parents=True, exist_ok=True)
        with open(OutputJsonPath, "w", encoding="utf-8") as FileHandle:
            json.dump(GeminiOutput, FileHandle, indent=2)
        print(f"Gemini JSON saved to: {OutputJsonPath}")

    if len(NonUsableRegions) == 0:
        ImageObject = Image.open(ImagePath)
        ImageWidth, ImageHeight = ImageObject.size
        UnifiedMask = np.zeros((ImageHeight, ImageWidth), dtype=bool)
        print("No non-usable regions found. Saving empty mask.")
    else:
        print("Generating unified mask with SAM...")
        UnifiedMask = GetUnifiedMaskFromRegions(
            ImagePath=ImagePath,
            SamCheckpointPath=SamCheckpointPath,
            NonUsableRegions=NonUsableRegions,
        )
        print("Unified mask generated.")

    SaveMask(OutputMaskPath, UnifiedMask)
    print(f"Unified binary mask saved to: {OutputMaskPath}")


if __name__ == "__main__":
    RunPipeline()