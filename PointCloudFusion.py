import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_INPUT_PATH = "single.npz"
DEFAULT_OUTPUT_PATH = "fused_output.npz"
DEFAULT_PROB_THRESHOLD = 0.35
DEFAULT_VOXEL_SIZE = 0.02


def GetExistingKey(Data: np.lib.npyio.NpzFile, Candidates: List[str]) -> str:
    for Key in Candidates:
        if Key in Data.files:
            return Key
    raise KeyError(f"None of these keys were found: {Candidates}. Available keys: {Data.files}")


def ToViewList(Array: np.ndarray) -> List[np.ndarray]:
    if isinstance(Array, np.ndarray) and Array.dtype == object:
        return [np.asarray(X) for X in Array.tolist()]

    if not isinstance(Array, np.ndarray):
        raise TypeError("Expected numpy array.")

    if Array.ndim >= 3:
        return [Array[I] for I in range(Array.shape[0])]

    if Array.ndim in (1, 2):
        return [Array]

    raise ValueError(f"Unsupported array shape for view parsing: {Array.shape}")


def FlattenViews(Views: List[np.ndarray], Kind: str) -> np.ndarray:
    Normalized: List[np.ndarray] = []
    for View in Views:
        Current = np.asarray(View)

        if Kind == "points":
            Current = Current.astype(np.float32)
            if Current.ndim != 2 or Current.shape[1] != 3:
                raise ValueError(f"Points must have shape (N, 3), got {Current.shape}")
            Normalized.append(Current)

        elif Kind in ("prob", "usable"):
            Current = Current.reshape(-1)
            if Kind == "prob":
                Current = Current.astype(np.float32)
            else:
                Current = Current.astype(np.float32)
            Normalized.append(Current)

        else:
            raise ValueError(f"Unsupported flatten kind: {Kind}")

    if not Normalized:
        raise ValueError(f"No views available for {Kind}.")

    return np.concatenate(Normalized, axis=0)


def BuiledAligned(
    Data: np.lib.npyio.NpzFile,
    TotalPoints: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    ClipFeatures: Optional[np.ndarray] = None
    ClipLabels: Optional[np.ndarray] = None
    ClipAvailableMask: Optional[np.ndarray] = None

    if "clip_keep" in Data.files:
        SparseClip = np.asarray(Data["clip_keep"], dtype=np.float32)

        if "clip_idx_keep" in Data.files:
            ClipIdxKeep = np.asarray(Data["clip_idx_keep"], dtype=np.int64).reshape(-1)
            if SparseClip.ndim == 2 and SparseClip.shape[0] == ClipIdxKeep.shape[0]:
                ValidIdx = (ClipIdxKeep >= 0) & (ClipIdxKeep < TotalPoints)
                ClipIdxKeep = ClipIdxKeep[ValidIdx]
                SparseClip = SparseClip[ValidIdx]

                if SparseClip.shape[0] > 0:
                    FeatureDim = SparseClip.shape[1]
                    ClipFeatures = np.zeros((TotalPoints, FeatureDim), dtype=np.float32)
                    ClipAvailableMask = np.zeros((TotalPoints,), dtype=bool)
                    ClipFeatures[ClipIdxKeep] = SparseClip
                    ClipAvailableMask[ClipIdxKeep] = True
        elif SparseClip.ndim == 2 and SparseClip.shape[0] == TotalPoints:
            ClipFeatures = SparseClip
            ClipAvailableMask = np.ones((TotalPoints,), dtype=bool)

    if "clip_label" in Data.files:
        RawLabels = np.asarray(Data["clip_label"], dtype=object).reshape(-1)

        if RawLabels.shape[0] == TotalPoints:
            ClipLabels = RawLabels
        elif "clip_idx_keep" in Data.files:
            ClipIdxKeep = np.asarray(Data["clip_idx_keep"], dtype=np.int64).reshape(-1)
            if RawLabels.shape[0] == ClipIdxKeep.shape[0]:
                ValidIdx = (ClipIdxKeep >= 0) & (ClipIdxKeep < TotalPoints)
                ClipIdxKeep = ClipIdxKeep[ValidIdx]
                RawLabels = RawLabels[ValidIdx]

                ClipLabels = np.empty((TotalPoints,), dtype=object)
                ClipLabels[:] = None
                ClipLabels[ClipIdxKeep] = RawLabels

    return ClipFeatures, ClipLabels, ClipAvailableMask


def LoadNPZ(InputPath: Path) -> Dict[str, np.ndarray]:
    Data = np.load(str(InputPath), allow_pickle=True)

    PointsKey = GetExistingKey(Data, ["points_world", "Pw", "points", "xyz", "pcd_points"])
    ProbKey = GetExistingKey(Data, ["probability", "prob", "probs", "confidence", "p"])

    PointsViews = ToViewList(Data[PointsKey])
    ProbViews = ToViewList(Data[ProbKey])

    if "usability" in Data.files:
        UsableViews = ToViewList(Data["usability"])
    elif "usable" in Data.files:
        UsableViews = ToViewList(Data["usable"])
    elif "is_usable" in Data.files:
        UsableViews = ToViewList(Data["is_usable"])
    elif "usable_mask" in Data.files:
        UsableViews = ToViewList(Data["usable_mask"])
    else:
        TotalPointsNoUsability = sum(np.asarray(View).shape[0] for View in PointsViews)
        UsableViews = [np.ones((TotalPointsNoUsability,), dtype=np.float32)]

    PointsAll = FlattenViews(PointsViews, "points")
    ProbAll = FlattenViews(ProbViews, "prob")
    UsabilityAll = FlattenViews(UsableViews, "usable")

    if PointsAll.shape[0] != ProbAll.shape[0]:
        raise ValueError(f"Prob length {ProbAll.shape[0]} does not match points {PointsAll.shape[0]}")
    if UsabilityAll.shape[0] != PointsAll.shape[0]:
        if UsabilityAll.shape[0] == 1:
            UsabilityAll = np.ones((PointsAll.shape[0],), dtype=np.float32)
        else:
            raise ValueError(f"Usability length {UsabilityAll.shape[0]} does not match points {PointsAll.shape[0]}")

    ClipFeaturesAll, ClipLabelsAll, ClipAvailableMask = BuiledAligned(Data, PointsAll.shape[0])

    return {
        "Points": PointsAll.astype(np.float32),
        "Prob": ProbAll.astype(np.float32),
        "Usability": UsabilityAll.astype(np.float32),
        "ClipFeatures": ClipFeaturesAll,
        "ClipLabels": ClipLabelsAll,
        "ClipAvailableMask": ClipAvailableMask,
    }


def FusePointClouds(
    Points: np.ndarray,
    Prob: np.ndarray,
    Usability: np.ndarray,
    ClipFeatures: Optional[np.ndarray],
    ClipLabels: Optional[np.ndarray],
    ClipAvailableMask: Optional[np.ndarray],
    ProbThreshold: float,
    VoxelSize: float,
) -> Dict[str, np.ndarray]:
    Keep = Prob >= ProbThreshold
    if not np.any(Keep):
        raise ValueError("No points remain after probability filtering.")

    PointsKept = Points[Keep].astype(np.float32)
    ProbKept = Prob[Keep].astype(np.float32)
    UsabilityKept = Usability[Keep].astype(np.float32)

    if ClipFeatures is not None:
        ClipFeaturesKept = ClipFeatures[Keep].astype(np.float32)
    else:
        ClipFeaturesKept = None

    if ClipLabels is not None:
        ClipLabelsKept = np.asarray(ClipLabels[Keep], dtype=object)
    else:
        ClipLabelsKept = None

    if ClipAvailableMask is not None:
        ClipAvailableKept = ClipAvailableMask[Keep]
    else:
        ClipAvailableKept = None

    VoxelIdx = np.floor(PointsKept / VoxelSize).astype(np.int64)
    UniqueVoxels, Inverse = np.unique(VoxelIdx, axis=0, return_inverse=True)

    NumVoxels = UniqueVoxels.shape[0]
    FusedPoints = np.zeros((NumVoxels, 3), dtype=np.float32)
    FusedProb = np.zeros((NumVoxels,), dtype=np.float32)
    FusedUsability = np.zeros((NumVoxels,), dtype=np.int8)
    PointCounts = np.zeros((NumVoxels,), dtype=np.int32)

    if ClipFeaturesKept is not None:
        FeatureDim = ClipFeaturesKept.shape[1]
        FusedClipFeatures = np.zeros((NumVoxels, FeatureDim), dtype=np.float32)
        HasClipFeaturesInput = True
    else:
        FusedClipFeatures = None
        HasClipFeaturesInput = False

    ProbSum = np.bincount(Inverse, weights=ProbKept, minlength=NumVoxels).astype(np.float32)
    PointCounts[:] = np.bincount(Inverse, minlength=NumVoxels).astype(np.int32)

    for Axis in range(3):
        WeightedSum = np.bincount(
            Inverse,
            weights=PointsKept[:, Axis] * ProbKept,
            minlength=NumVoxels,
        ).astype(np.float32)
        FusedPoints[:, Axis] = WeightedSum / np.maximum(ProbSum, 1e-8)

    FusedProb[:] = ProbSum / np.maximum(PointCounts, 1)

    UsableWeightedSum = np.bincount(
        Inverse,
        weights=UsabilityKept * ProbKept,
        minlength=NumVoxels,
    ).astype(np.float32)
    UsableRatio = UsableWeightedSum / np.maximum(ProbSum, 1e-8)
    FusedUsability[:] = np.where(UsableRatio >= 0.0, 1, -1).astype(np.int8)

    if HasClipFeaturesInput:
        if ClipAvailableKept is not None:
            ClipProbWeights = ProbKept * ClipAvailableKept.astype(np.float32)
        else:
            ClipProbWeights = ProbKept

        ClipProbSum = np.bincount(Inverse, weights=ClipProbWeights, minlength=NumVoxels).astype(np.float32)

        for Dim in range(FeatureDim):
            WeightedFeatSum = np.bincount(
                Inverse,
                weights=ClipFeaturesKept[:, Dim] * ClipProbWeights,
                minlength=NumVoxels,
            ).astype(np.float32)
            FusedClipFeatures[:, Dim] = WeightedFeatSum / np.maximum(ClipProbSum, 1e-8)

    FusedClipLabels = np.empty((NumVoxels,), dtype=object)
    FusedClipLabels[:] = ""

    if ClipLabelsKept is not None:

        for VoxelIndex in range(NumVoxels):
            Members = np.where(Inverse == VoxelIndex)[0]
            if Members.size == 0:
                continue

            LabelsVoxel = ClipLabelsKept[Members]
            ProbsVoxel = ProbKept[Members]

            LabelToWeight: Dict[str, float] = {}
            for LabelValue, Weight in zip(LabelsVoxel, ProbsVoxel):
                if LabelValue is None:
                    continue
                LabelText = str(LabelValue).strip()
                if not LabelText:
                    continue
                LabelToWeight[LabelText] = LabelToWeight.get(LabelText, 0.0) + float(Weight)

            if LabelToWeight:
                BestLabel = max(LabelToWeight.items(), key=lambda X: X[1])[0]
                FusedClipLabels[VoxelIndex] = BestLabel

    Result = {
        # Primary keys: keep close to original pipeline NPZ naming.
        "Pw": FusedPoints,
        "prob": FusedProb,
        "usability": FusedUsability,
        "voxel_indices": UniqueVoxels,
        "point_count_per_voxel": PointCounts,
    }

    if FusedClipFeatures is not None:
        Result["clip_keep"] = FusedClipFeatures
        Result["clip_features_fused"] = FusedClipFeatures

    Result["clip_label"] = FusedClipLabels
    Result["clip_label_fused"] = FusedClipLabels

    return Result


def SaveFusedOutput(OutputPath: Path, Fused: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(str(OutputPath), **Fused)


def PointCloudFusion() -> None:
    Parser = argparse.ArgumentParser(description="Fuse point clouds from pipeline NPZ output.")
    Parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT_PATH), help="Path to input NPZ file")
    Parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_PATH), help="Path to output fused NPZ file")
    Parser.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD, help="Minimum probability to keep a point")
    Parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE, help="Voxel size for fusion")

    Args = Parser.parse_args()

    Loaded = LoadNPZ(Args.input)
    Fused = FusePointClouds(
        Points=Loaded["Points"],
        Prob=Loaded["Prob"],
        Usability=Loaded["Usability"],
        ClipFeatures=Loaded["ClipFeatures"],
        ClipLabels=Loaded["ClipLabels"],
        ClipAvailableMask=Loaded["ClipAvailableMask"],
        ProbThreshold=Args.prob_threshold,
        VoxelSize=Args.voxel_size,
    )
    SaveFusedOutput(Args.output, Fused)

    print("Fused NPZ saved:", Args.output.resolve())
    print("Fused points:", int(Fused["Pw"].shape[0]))
    if "clip_label_fused" in Fused:
        NonEmpty = np.count_nonzero(np.array([str(X).strip() != "" for X in Fused["clip_label_fused"]]))
        print("Fused points with clip labels:", int(NonEmpty))


if __name__ == "__main__":
    PointCloudFusion()