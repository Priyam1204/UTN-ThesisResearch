# material_prediction_stage.py

import os
import json
import mimetypes
import argparse
from pathlib import Path

from google import genai


DefaultPrompt = """You are an expert in material science and visual reasoning for robotic applications.

Given an RGB image of an object, your task is to:
1. Identify the object and provide a short, clear description.
2. Predict the main materials present in the object based on visual appearance.
3. Assign a realistic approximate mass density (in kg/m³) for each material.

Guidelines:
- Only include dominant materials.
- Use common material categories (e.g., wood, metal, plastic, glass, cardboard, rubber).
- Use realistic density values based on standard physical properties.
- Ensure the output is strictly in JSON format.

Output format:
{
  "Description": "Short description of the object",
  "Material_1": density_value,
  "Material_2": density_value
}
"""

DefaultModelName = "gemini-robotics-er-1.5-preview"


def BuildArgParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(
        description="Pipeline stage 1: predict material from image with Gemini."
    )
    Parser.add_argument("--image_path", type=str, required=True, help="Input image path.")
    Parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSON file.")
    return Parser


def MaterialPrediction() -> None:
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ApiKey = os.getenv("GEMINI_API_KEY")
    if not ApiKey:
        raise RuntimeError("Set GEMINI_API_KEY environment variable first.")

    ImagePath = Path(Args.image_path)
    if not ImagePath.exists():
        raise FileNotFoundError(f"Image not found: {ImagePath}")

    OutputJsonPath = Path(Args.output_dir) / f"{ImagePath.stem}.json"
    OutputJsonPath.parent.mkdir(parents=True, exist_ok=True)

    with open(ImagePath, "rb") as FileHandle:
        ImageBytes = FileHandle.read()

    MimeType, _ = mimetypes.guess_type(str(ImagePath))
    MimeType = MimeType or "image/jpeg"

    Client = genai.Client(api_key=ApiKey)
    print("Client ready, model:", DefaultModelName)

    Response = Client.models.generate_content(
        model=DefaultModelName,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": DefaultPrompt},
                    {"inline_data": {"mime_type": MimeType, "data": ImageBytes}},
                ],
            }
        ],
    )

    RawResponse = Response.text or ""

    print("\n=== RAW MODEL RESPONSE ===\n")
    print(RawResponse)

    ParsedResponse = json.loads(RawResponse)

    with open(OutputJsonPath, "w", encoding="utf-8") as FileHandle:
        json.dump(ParsedResponse, FileHandle, indent=2, ensure_ascii=False)

    print(f"\nParsed output saved to: {OutputJsonPath}")


if __name__ == "__main__":
    MaterialPrediction()