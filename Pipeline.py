import argparse
import os
import subprocess
import sys
from pathlib import Path


DefaultSamCheckpoint = "checkpoints/sam_vit_h_4b8939.pth"


def BuildArgParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(
        description="Run full thesis pipeline from a single image and produce final NPZ output."
    )
    Parser.add_argument("--image_path", type=str, required=True, help="Input image path.")
    return Parser


def EnsureExists(PathValue: Path, Label: str) -> None:
    if not PathValue.exists():
        raise FileNotFoundError(f"{Label} not found: {PathValue}")


def RunStage(Command: list[str], StageName: str) -> None:
    print(f"\n=== {StageName} ===")
    print("Command:", " ".join(Command))
    subprocess.run(Command, check=True)


def RunPipeline() -> None:
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ImagePath = Path(Args.image_path).resolve()
    EnsureExists(ImagePath, "Image")

    WorkspaceRoot = Path(__file__).resolve().parent
    ImageStem = ImagePath.stem

    SamCheckpointPath = (WorkspaceRoot / DefaultSamCheckpoint).resolve()
    EnsureExists(SamCheckpointPath, "SAM checkpoint")

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY environment variable first.")

    ImageDir = ImagePath.parent
    IntrinsicPath = ImageDir / "intrinsic.txt"
    ExtrinsicPath = ImageDir / "extrinsic.txt"
    EnsureExists(IntrinsicPath, "intrinsic.txt")
    EnsureExists(ExtrinsicPath, "extrinsic.txt")

    ResultsRoot = WorkspaceRoot / "results"
    MaterialsDir = ResultsRoot / "materials"
    MasksDir = ResultsRoot / "masks"
    ReconstructionDir = ResultsRoot / "reconstruction"
    ProjectionDir = ResultsRoot / "projection"

    for Directory in [MaterialsDir, MasksDir, ReconstructionDir, ProjectionDir]:
        Directory.mkdir(parents=True, exist_ok=True)

    MaterialJsonPath = MaterialsDir / f"{ImageStem}.json"
    ObjectMaskPath = MasksDir / f"{ImageStem}_object_mask.png"
    NonUsabilityMaskPath = MasksDir / f"{ImageStem}_nonusable_mask.png"
    ReconstructionPlyPath = ReconstructionDir / f"{ImageStem}.ply"
    ProjectionNpZPath = ProjectionDir / f"{ImageStem}.npz"

    PythonExe = sys.executable

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "MaterialPrediction.py"),
            "--image_path",
            str(ImagePath),
            "--output_dir",
            str(MaterialsDir),
        ],
        "Stage 1 - Material Prediction",
    )
    EnsureExists(MaterialJsonPath, "Material JSON")

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "AutomaticSegmentation.py"),
            "--image_path",
            str(ImagePath),
            "--stage1_json_path",
            str(MaterialJsonPath),
            "--sam_checkpoint",
            str(SamCheckpointPath),
            "--output_mask_path",
            str(ObjectMaskPath),
        ],
        "Stage 2A - Automatic Segmentation",
    )
    EnsureExists(ObjectMaskPath, "Object mask")

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "SAM3DReconstruction.py"),
            "--image_path",
            str(ImagePath),
            "--mask_path",
            str(ObjectMaskPath),
            "--output_path",
            str(ReconstructionPlyPath),
        ],
        "Stage 3 - SAM3D Reconstruction",
    )
    EnsureExists(ReconstructionPlyPath, "Reconstruction PLY")

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "PointCloudProjection.py"),
            "--rgb",
            str(ImagePath),
            "--mask",
            str(ObjectMaskPath),
            "--ply",
            str(ReconstructionPlyPath),
            "--intrinsics",
            str(IntrinsicPath),
            "--extrinsics",
            str(ExtrinsicPath),
            "--save-pipeline",
            str(ProjectionNpZPath),
        ],
        "Stage 4 - Point Cloud Projection",
    )
    EnsureExists(ProjectionNpZPath, "Projection NPZ")

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "ClipFeature.py"),
            "--input_path",
            str(ProjectionNpZPath),
        ],
        "Stage 5 - CLIP Feature",
    )

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "ClipFeatureMatching.py"),
            "--input_path",
            str(ProjectionNpZPath),
            "--llm_json",
            str(MaterialJsonPath),
        ],
        "Stage 6 - CLIP Feature Matching",
    )

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "NonUsabilityMask.py"),
            "--image_path",
            str(ImagePath),
            "--sam_checkpoint",
            str(SamCheckpointPath),
            "--output_mask_path",
            str(NonUsabilityMaskPath),
        ],
        "Stage 7 - Non-Usability Mask",
    )
    EnsureExists(NonUsabilityMaskPath, "Non-usability mask")

    RunStage(
        [
            PythonExe,
            str(WorkspaceRoot / "3DNonUsability.py"),
            "--projection_input",
            str(ProjectionNpZPath),
            "--nonuse_mask_path",
            str(NonUsabilityMaskPath),
        ],
        "Stage 8 - 3D Non-Usability",
    )

    EnsureExists(ProjectionNpZPath, "Final NPZ")
    print("\nPipeline complete.")
    print("Final NPZ:", ProjectionNpZPath)


if __name__ == "__main__":
    RunPipeline()
