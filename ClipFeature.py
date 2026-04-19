import argparse
from pathlib import Path
from typing import Dict, Tuple

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


DefaultInputPath = "projection_stage_output.npz"
DefaultProbThreshold = 0.5

# Fixed CLIP-related settings
ClipPatchSize = 28
ClipStride = 28
ClipBatchSize = 64
ClipModelName = "ViT-B/32"


def LoadProjectionStageOutput(InputPath: Path) -> Dict[str, np.ndarray]:
    Data = np.load(str(InputPath), allow_pickle=True)
    return {
        "Rgb": Data["rgb"],
        "Mask": Data["mask"].astype(bool),
        "Pw": Data["Pw"].astype(np.float32),
        "Prob": Data["prob"].astype(np.float32),
        "IdxVis": Data["idx_vis"].astype(np.int64),
        "UuVis": Data["uu_vis"].astype(np.float32),
        "VvVis": Data["vv_vis"].astype(np.float32),
    }


def LoadNpZAsDict(InputPath: Path) -> Dict[str, np.ndarray]:
    Data = np.load(str(InputPath), allow_pickle=True)
    return {Key: Data[Key] for Key in Data.files}


def SaveNpZAtomic(OutputPath: Path, Payload: Dict[str, np.ndarray]) -> None:
    TempPath = OutputPath.with_name(OutputPath.stem + ".tmp.npz")
    np.savez_compressed(str(TempPath), **Payload)
    TempPath.replace(OutputPath)


def BuildVisibleOnIndices(
    Mask: np.ndarray,
    UuVis: np.ndarray,
    VvVis: np.ndarray,
    IdxVis: np.ndarray,
) -> np.ndarray:
    if UuVis.size == 0 or VvVis.size == 0 or IdxVis.size == 0:
        return np.array([], dtype=np.int64)

    OnMask = Mask[VvVis.astype(np.int32), UuVis.astype(np.int32)] > 0
    return IdxVis[OnMask].astype(np.int64)


def FilterIndicesByProbability(
    ProbAll: np.ndarray,
    IdxOn: np.ndarray,
    ProbThreshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if IdxOn.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    ProbOn = ProbAll[IdxOn]
    KeepMask = ProbOn >= ProbThreshold
    IdxKeep = IdxOn[KeepMask]
    ProbKeep = ProbOn[KeepMask].astype(np.float32)
    return IdxKeep, ProbKeep


def BuildUvForSelectedIndices(
    IdxVis: np.ndarray,
    UuVis: np.ndarray,
    VvVis: np.ndarray,
    IdxKeep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if IdxKeep.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    PosInVis = {int(PointIndex): Position for Position, PointIndex in enumerate(IdxVis)}
    Positions = np.array([PosInVis[int(PointIndex)] for PointIndex in IdxKeep], dtype=np.int64)

    UKeep = UuVis[Positions].astype(np.float32)
    VKeep = VvVis[Positions].astype(np.float32)
    return UKeep, VKeep


def BuildPatchGrid(
    Height: int,
    Width: int,
    PatchSizePx: int,
    StridePx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs = np.arange(0, Width - PatchSizePx + 1, StridePx, dtype=np.int32)
    Ys = np.arange(0, Height - PatchSizePx + 1, StridePx, dtype=np.int32)
    XCenters = Xs + PatchSizePx * 0.5
    YCenters = Ys + PatchSizePx * 0.5
    return Xs, Ys, XCenters, YCenters


def LoadClipModel(Device: str):
    Model, Preprocess = clip.load(ClipModelName, device=Device)
    Model = Model.eval()
    return Model, Preprocess


def NormalizeRgbToUint8(Rgb: np.ndarray) -> np.ndarray:
    if Rgb.dtype == np.uint8:
        return Rgb
    return np.clip(Rgb * 255.0, 0, 255).astype(np.uint8)


def EncodePatchFeatures(
    Rgb: np.ndarray,
    Model,
    Preprocess,
    Device: str,
    PatchSizePx: int,
    StridePx: int,
    BatchSize: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    RgbUint8 = NormalizeRgbToUint8(Rgb)
    ImagePil = Image.fromarray(RgbUint8)
    Height, Width = RgbUint8.shape[:2]

    Xs, Ys, XCenters, YCenters = BuildPatchGrid(Height, Width, PatchSizePx, StridePx)

    FeatureBatches = []
    Batch = []

    with torch.no_grad():
        for Y in Ys:
            for X in Xs:
                Crop = ImagePil.crop((int(X), int(Y), int(X + PatchSizePx), int(Y + PatchSizePx)))
                Batch.append(Preprocess(Crop))

                if len(Batch) == BatchSize:
                    InputTensor = torch.stack(Batch).to(Device)
                    Features = Model.encode_image(InputTensor).float()
                    Features = Features / (Features.norm(dim=-1, keepdim=True) + 1e-12)
                    FeatureBatches.append(Features.cpu())
                    Batch = []

        if Batch:
            InputTensor = torch.stack(Batch).to(Device)
            Features = Model.encode_image(InputTensor).float()
            Features = Features / (Features.norm(dim=-1, keepdim=True) + 1e-12)
            FeatureBatches.append(Features.cpu())

    PatchFeatures = torch.cat(FeatureBatches, dim=0) if FeatureBatches else torch.empty((0, 0))
    return PatchFeatures, Xs, Ys, XCenters, YCenters


def NearestPatchIndices(
    UKeep: np.ndarray,
    VKeep: np.ndarray,
    Xs: np.ndarray,
    Ys: np.ndarray,
    XCenters: np.ndarray,
    YCenters: np.ndarray,
) -> np.ndarray:
    Gx = np.clip(np.searchsorted(XCenters, UKeep) - 1, 0, len(Xs) - 1)
    Gx2 = np.clip(Gx + 1, 0, len(Xs) - 1)
    Gx = np.where(np.abs(XCenters[Gx2] - UKeep) < np.abs(XCenters[Gx] - UKeep), Gx2, Gx)

    Gy = np.clip(np.searchsorted(YCenters, VKeep) - 1, 0, len(Ys) - 1)
    Gy2 = np.clip(Gy + 1, 0, len(Ys) - 1)
    Gy = np.where(np.abs(YCenters[Gy2] - VKeep) < np.abs(YCenters[Gy] - VKeep), Gy2, Gy)

    return (Gy * len(Xs) + Gx).astype(np.int64)


def SaveSparseClipFeatures(
    NpzPath: Path,
    IdxKeep: np.ndarray,
    UKeep: np.ndarray,
    VKeep: np.ndarray,
    ProbKeep: np.ndarray,
    ClipKeep: torch.Tensor,
    ProbThreshold: float,
    PatchSizePx: int,
    StridePx: int,
    ClipModelName: str,
) -> None:
    Payload = LoadNpZAsDict(NpzPath)
    ClipKeepNp = ClipKeep.detach().cpu().numpy().astype(np.float32)

    Payload["clip_idx_keep"] = IdxKeep.astype(np.int64)
    Payload["clip_u_keep"] = UKeep.astype(np.float32)
    Payload["clip_v_keep"] = VKeep.astype(np.float32)
    Payload["clip_prob_keep"] = ProbKeep.astype(np.float32)
    Payload["clip_keep"] = ClipKeepNp
    Payload["clip_meta_prob_thr"] = np.array([float(ProbThreshold)], dtype=np.float32)
    Payload["clip_meta_patch_px"] = np.array([int(PatchSizePx)], dtype=np.int32)
    Payload["clip_meta_stride_px"] = np.array([int(StridePx)], dtype=np.int32)
    Payload["clip_meta_model"] = np.array([ClipModelName])

    SaveNpZAtomic(NpzPath, Payload)


def PlotSelectedPoints(Rgb: np.ndarray, UKeep: np.ndarray, VKeep: np.ndarray, ProbThreshold: float) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(Rgb)
    plt.scatter(UKeep, VKeep, s=3, alpha=0.8)
    plt.title(f"Points with prob >= {ProbThreshold}: {len(UKeep)}")
    plt.axis("off")
    plt.show()


def BuildArgParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(description="Pipeline stage: assign CLIP features to selected points.")
    Parser.add_argument("--input_path", type=str, default=DefaultInputPath, help="Projection stage NPZ path.")
    Parser.add_argument("--prob_threshold", type=float, default=DefaultProbThreshold, help="Point keep threshold.")
    Parser.add_argument("--show_plot", action="store_true", help="Show selected points over RGB image.")
    return Parser


def ClipFeature() -> None:
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    InputPath = Path(Args.input_path)
    if not InputPath.exists():
        raise FileNotFoundError(f"Input file not found: {InputPath}")

    StageData = LoadProjectionStageOutput(InputPath)
    Rgb = StageData["Rgb"]
    Mask = StageData["Mask"]
    ProbAll = StageData["Prob"]
    IdxVis = StageData["IdxVis"]
    UuVis = StageData["UuVis"]
    VvVis = StageData["VvVis"]

    IdxOn = BuildVisibleOnIndices(Mask=Mask, UuVis=UuVis, VvVis=VvVis, IdxVis=IdxVis)
    IdxKeep, ProbKeep = FilterIndicesByProbability(
        ProbAll=ProbAll,
        IdxOn=IdxOn,
        ProbThreshold=float(Args.prob_threshold),
    )

    print("idx_on:", len(IdxOn))
    print(f"selected prob >= {Args.prob_threshold}:", len(IdxKeep))
    if ProbKeep.size > 0:
        print("prob_keep range:", (float(ProbKeep.min()), float(ProbKeep.max())))
    else:
        print("prob_keep range: None")

    if IdxKeep.size == 0:
        EmptyClip = torch.empty((0, 512), dtype=torch.float32)
        SaveSparseClipFeatures(
            NpzPath=InputPath,
            IdxKeep=np.array([], dtype=np.int64),
            UKeep=np.array([], dtype=np.float32),
            VKeep=np.array([], dtype=np.float32),
            ProbKeep=np.array([], dtype=np.float32),
            ClipKeep=EmptyClip,
            ProbThreshold=float(Args.prob_threshold),
            PatchSizePx=ClipPatchSize,
            StridePx=ClipStride,
            ClipModelName=ClipModelName,
        )
        print("No points selected. Wrote empty clip_* arrays into NPZ.")
        return

    UKeep, VKeep = BuildUvForSelectedIndices(
        IdxVis=IdxVis,
        UuVis=UuVis,
        VvVis=VvVis,
        IdxKeep=IdxKeep,
    )

    Device = "cuda" if torch.cuda.is_available() else "cpu"
    Model, Preprocess = LoadClipModel(Device=Device)

    PatchFeatures, Xs, Ys, XCenters, YCenters = EncodePatchFeatures(
        Rgb=Rgb,
        Model=Model,
        Preprocess=Preprocess,
        Device=Device,
        PatchSizePx=ClipPatchSize,
        StridePx=ClipStride,
        BatchSize=ClipBatchSize,
    )

    print("Patch grid:", len(Xs), "x", len(Ys), "=", len(Xs) * len(Ys))
    print("patch_feats:", tuple(PatchFeatures.shape))

    PatchIndex = NearestPatchIndices(
        UKeep=UKeep,
        VKeep=VKeep,
        Xs=Xs,
        Ys=Ys,
        XCenters=XCenters,
        YCenters=YCenters,
    )
    ClipKeep = PatchFeatures[torch.from_numpy(PatchIndex)]

    SaveSparseClipFeatures(
        NpzPath=InputPath,
        IdxKeep=IdxKeep,
        UKeep=UKeep,
        VKeep=VKeep,
        ProbKeep=ProbKeep,
        ClipKeep=ClipKeep,
        ProbThreshold=float(Args.prob_threshold),
        PatchSizePx=ClipPatchSize,
        StridePx=ClipStride,
        ClipModelName=ClipModelName,
    )

    print(
        "Assigned CLIP to:",
        len(IdxKeep),
        "points | updated NPZ:",
        str(InputPath),
        "| clip dim:",
        ClipKeep.shape[1],
    )

    if Args.show_plot:
        PlotSelectedPoints(Rgb=Rgb, UKeep=UKeep, VKeep=VKeep, ProbThreshold=float(Args.prob_threshold))


if __name__ == "__main__":
    ClipFeature()