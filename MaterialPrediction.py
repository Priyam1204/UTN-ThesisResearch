#!/usr/bin/env python3
"""
Material prediction from an image using Google Gemini (google-genai SDK).

Usage:
  python material_prediction.py /path/to/image.jpg

Auth options (pick one):
  1) Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
  2) Pass --key-path /path/to/service_account.json
  3) Use Application Default Credentials (e.g., gcloud auth application-default login)

This script saves a JSON file named:
  "Material prediction.json"
in the SAME folder as the input image.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from google.oauth2 import service_account
from PIL import Image


DEFAULT_PROMPT = (
    "List materials of the main object found in this image. "
    "Return the result strictly as JSON with this schema: "
    "{'object': [['material', 'density_kg_per_m3'], ...]}."
)


def _build_client(key_path: Optional[str]) -> genai.Client:
    """
    Build a Gemini client. If key_path is provided, use a service account.
    Otherwise rely on GOOGLE_APPLICATION_CREDENTIALS or ADC.
    """
    if key_path:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(
            key_path, scopes=scopes
        )
        return genai.Client(credentials=credentials)

    # GOOGLE_APPLICATION_CREDENTIALS / ADC
    return genai.Client()


def _extract_json(text: str) -> Any:
    """
    Ensure valid JSON even if the model wraps it in markdown.
    """
    text = text.strip()

    # Remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON block
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in model response.")
    return json.loads(match.group(1))


def predict_materials(
    image_path: Path,
    key_path: Optional[str],
    model: str,
    prompt: str,
) -> Any:
    client = _build_client(key_path)

    image = Image.open(image_path)

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    if not getattr(response, "text", None):
        raise RuntimeError("Model returned no text.")

    return _extract_json(response.text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict materials from an image and save JSON next to it."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image (jpg/png/...)",
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="Service account JSON key path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt sent with the image",
    )

    args = parser.parse_args()
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_path = image_path.parent / "Material prediction.json"

    result = predict_materials(
        image_path=image_path,
        key_path=args.key_path,
        model=args.model,
        prompt=args.prompt,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
