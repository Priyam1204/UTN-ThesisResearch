import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch


ClipModelName = "ViT-B/32"


def LoadNpZ(InputPath: Path):
    Data = np.load(str(InputPath), allow_pickle=True)
    return {Key: Data[Key] for Key in Data.files}


def SaveNpZ(OutputPath: Path, Payload: dict):
    np.savez_compressed(str(OutputPath), **Payload)


def BuildAutoPrompts(MaterialName: str):
    Name = str(MaterialName).strip().lower()
    return [
        f"image of {Name} object",
        f"picture of {Name}",
        f"photo of {Name} material",
    ]


def LoadPrompts(JsonPath: Path):
    with open(JsonPath, "r") as f:
        Data = json.load(f)

    # Supports two formats:
    # 1) Prompt dictionary: {"label": ["prompt1", "prompt2"]}
    # 2) MaterialPrediction-style: {"Description": "...", "wood": 700, "metal": 7800}
    labels = []
    prompts = []
    label_ids = []

    PromptDictFormat = all(isinstance(Value, list) for Value in Data.values())

    if PromptDictFormat:
        for Index, (Label, PromptList) in enumerate(Data.items()):
            labels.append(str(Label))
            for Prompt in PromptList:
                prompts.append(str(Prompt))
                label_ids.append(Index)
        return labels, prompts, np.array(label_ids, dtype=np.int64)

    IgnoreKeys = {"description", "object", "summary"}
    MaterialNames = []
    for Key, Value in Data.items():
        KeyText = str(Key).strip()
        KeyLower = KeyText.lower()
        if KeyLower in IgnoreKeys:
            continue

        if KeyText.lower().startswith("material_"):
            # If a key like Material_wood exists, use the suffix as label.
            Parsed = KeyText.split("_", 1)[1].strip()
            if Parsed:
                MaterialNames.append(Parsed)
                continue

        if KeyText.lower().startswith("material") and KeyText[8:].strip():
            Parsed = KeyText[8:].strip(" _-")
            if Parsed:
                MaterialNames.append(Parsed)
                continue

        if isinstance(Value, (int, float)):
            MaterialNames.append(KeyText)

    if not MaterialNames:
        raise ValueError("No materials found in llm_json.")

    for Index, Label in enumerate(MaterialNames):
        labels.append(Label)
        for Prompt in BuildAutoPrompts(Label):
            prompts.append(Prompt)
            label_ids.append(Index)

    return labels, prompts, np.array(label_ids, dtype=np.int64)


def ClipFeatureMatching():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--input_path", required=True)
    Parser.add_argument("--llm_json", required=True)
    Parser.add_argument("--output_path", default=None)
    Args = Parser.parse_args()

    InputPath = Path(Args.input_path)
    OutputPath = Path(Args.output_path) if Args.output_path else InputPath

    Data = LoadNpZ(InputPath)

    ClipFeatures = Data["clip_keep"]  # (N, D)

    if ClipFeatures.shape[0] == 0:
        Data["clip_label"] = np.array([], dtype=object)
        SaveNpZ(OutputPath, Data)
        print("No points found.")
        return

    Labels, Prompts, PromptLabelIds = LoadPrompts(Path(Args.llm_json))

    Device = "cuda" if torch.cuda.is_available() else "cpu"
    Model, _ = clip.load(ClipModelName, device=Device)
    Model.eval()

    # Encode text
    with torch.no_grad():
        TextTokens = clip.tokenize(Prompts).to(Device)
        TextFeatures = Model.encode_text(TextTokens).float()
        TextFeatures /= TextFeatures.norm(dim=-1, keepdim=True)

    # Normalize point features
    ClipTensor = torch.from_numpy(ClipFeatures).to(Device)
    ClipTensor = ClipTensor / (ClipTensor.norm(dim=-1, keepdim=True) + 1e-12)

    # Similarity
    Similarity = ClipTensor @ TextFeatures.T  # (N, P)

    # Best prompt per point
    BestPromptIndex = Similarity.argmax(dim=1).cpu().numpy()
    BestLabelIndex = PromptLabelIds[BestPromptIndex]

    # Assign label names
    PointLabels = np.array([Labels[Index] for Index in BestLabelIndex], dtype=object)

    # Save labels
    Data["clip_label"] = PointLabels

    SaveNpZ(OutputPath, Data)

    print("Assigned labels to", len(PointLabels), "points")


if __name__ == "__main__":
    ClipFeatureMatching()