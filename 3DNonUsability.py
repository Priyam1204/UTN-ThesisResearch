import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


DefaultProjectionInputPath = "projection_stage_output.npz"

Forward = np.array([1, 0, 0], dtype=np.float32)
Right = np.array([0, -1, 0], dtype=np.float32)
Up = np.array([0, 0, -1], dtype=np.float32)


def LoadProjectionStageData(InputPath: Path) -> Dict[str, np.ndarray]:
    Data = np.load(str(InputPath), allow_pickle=True)
    return {
        "Rgb": Data["rgb"],
        "K": Data["K"].astype(np.float32),
        "W2C": Data["w2c"].astype(np.float32),
        "Pw": Data["Pw"].astype(np.float32),
        "Prob": Data["prob"].astype(np.float32),
        "Colors": Data["colors"].astype(np.float32) if "colors" in Data else None,
    }


def LoadNpZAsDict(InputPath: Path) -> Dict[str, np.ndarray]:
    Data = np.load(str(InputPath), allow_pickle=True)
    return {Key: Data[Key] for Key in Data.files}


def SaveNpZAtomic(OutputPath: Path, Payload: Dict[str, np.ndarray]) -> None:
    TempPath = OutputPath.with_name(OutputPath.stem + ".tmp.npz")
    np.savez_compressed(str(TempPath), **Payload)
    TempPath.replace(OutputPath)


def ProjectWithBasis(
    PointsWorld: np.ndarray,
    K: np.ndarray,
    W2C: np.ndarray,
    ForwardAxis: np.ndarray,
    RightAxis: np.ndarray,
    UpAxis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Points = np.asarray(PointsWorld, dtype=np.float32).reshape(-1, 3)
    Count = Points.shape[0]
    PointsH = np.hstack([Points, np.ones((Count, 1), dtype=np.float32)])
    PointsCamera = (W2C @ PointsH.T).T[:, :3]

    Depth = PointsCamera @ ForwardAxis
    XAxis = PointsCamera @ RightAxis
    YAxis = PointsCamera @ UpAxis

    Eps = 1e-8
    SafeDepth = np.where(np.abs(Depth) < Eps, Eps, Depth)

    U = K[0, 0] * (XAxis / SafeDepth) + K[0, 2]
    V = K[1, 1] * (YAxis / SafeDepth) + K[1, 2]
    return U, V, Depth


def LoadBinaryMask(MaskPath: Path, TargetShape: Tuple[int, int]) -> np.ndarray:
    MaskGray = cv2.imread(str(MaskPath), cv2.IMREAD_GRAYSCALE)
    if MaskGray is None:
        raise FileNotFoundError(f"Mask not found: {MaskPath}")

    TargetHeight, TargetWidth = TargetShape
    if MaskGray.shape[:2] != (TargetHeight, TargetWidth):
        MaskGray = cv2.resize(MaskGray, (TargetWidth, TargetHeight), interpolation=cv2.INTER_NEAREST)

    return (MaskGray > 127).astype(np.uint8)


def AssignUsabilityFromMask(
    Pw: np.ndarray,
    Prob: np.ndarray,
    Rgb: np.ndarray,
    K: np.ndarray,
    W2C: np.ndarray,
    NonUseMaskBin: np.ndarray,
) -> Dict[str, np.ndarray]:
    U, V, Depth = ProjectWithBasis(Pw, K, W2C, Forward, Right, Up)
    Height, Width = Rgb.shape[:2]

    Ui = np.round(U).astype(np.int32)
    Vi = np.round(V).astype(np.int32)

    Valid = (Depth > 1e-6) & (Ui >= 0) & (Ui < Width) & (Vi >= 0) & (Vi < Height)

    OnNonUse = np.zeros_like(Valid, dtype=bool)
    OnNonUse[Valid] = NonUseMaskBin[Vi[Valid], Ui[Valid]] == 1

    Usability = np.ones(len(Pw), dtype=np.int8)  # +1 usable, -1 non-usable
    CandidateIdx = np.where(OnNonUse)[0]

    if CandidateIdx.size > 0:
        PixelId = Vi[CandidateIdx].astype(np.int64) * Width + Ui[CandidateIdx].astype(np.int64)
        SortOrder = np.lexsort((-Prob[CandidateIdx], PixelId))
        SortedIdx = CandidateIdx[SortOrder]
        SortedPix = PixelId[SortOrder]

        FirstPerPixel = np.ones(SortedIdx.shape[0], dtype=bool)
        FirstPerPixel[1:] = SortedPix[1:] != SortedPix[:-1]
        ChosenNonUseIdx = SortedIdx[FirstPerPixel]

        Usability[ChosenNonUseIdx] = -1
    else:
        ChosenNonUseIdx = np.array([], dtype=np.int64)

    return {
        "U": U,
        "V": V,
        "Ui": Ui,
        "Vi": Vi,
        "Depth": Depth,
        "Valid": Valid,
        "OnNonUse": OnNonUse,
        "Usability": Usability,
        "ChosenNonUseIdx": ChosenNonUseIdx,
    }


def PlotProjectionUsability(
    Rgb: np.ndarray,
    Ui: np.ndarray,
    Vi: np.ndarray,
    Valid: np.ndarray,
    Usability: np.ndarray,
    MaxPoints: int = 200000,
) -> None:
    ValidIdx = np.where(Valid)[0]
    if ValidIdx.size > MaxPoints:
        Sample = np.random.choice(ValidIdx.size, MaxPoints, replace=False)
        PlotIdx = ValidIdx[Sample]
    else:
        PlotIdx = ValidIdx

    UseIdx = PlotIdx[Usability[PlotIdx] == 1]
    NonUseIdx = PlotIdx[Usability[PlotIdx] == -1]

    plt.figure(figsize=(9, 9))
    plt.imshow(Rgb)
    plt.scatter(Ui[UseIdx], Vi[UseIdx], s=0.2, c="lime", alpha=0.3, label="usable")
    plt.scatter(Ui[NonUseIdx], Vi[NonUseIdx], s=1.0, c="red", alpha=0.9, label="non-usable")
    plt.title("Projection: usable vs non-usable")
    plt.legend(markerscale=10)
    plt.axis("off")
    plt.show()


def AppendUsabilityToProjectionNpZ(
    ProjectionInputPath: Path,
    Pw: np.ndarray,
    Prob: np.ndarray,
    Usability: np.ndarray,
    Colors: np.ndarray,
) -> None:
    Payload = LoadNpZAsDict(ProjectionInputPath)

    Payload["points_world"] = np.asarray(Pw, dtype=np.float32)
    Payload["probability"] = np.asarray(Prob, dtype=np.float32)
    Payload["usability"] = np.asarray(Usability, dtype=np.int8)

    if "clip_meta_prob_thr" in Payload:
        ClipMetaProbThr = np.asarray(Payload["clip_meta_prob_thr"], dtype=np.float32).reshape(-1)
        ProbThrValue = float(ClipMetaProbThr[0]) if ClipMetaProbThr.size > 0 else -1.0
    else:
        ProbThrValue = -1.0

    Payload["meta_prob_thr"] = np.array([ProbThrValue], dtype=np.float32)

    if Colors is not None:
        ColorsAll = np.asarray(Colors, dtype=np.float32)
        PointsAll = np.asarray(Pw, dtype=np.float32)
        if ColorsAll.shape[0] == PointsAll.shape[0]:
            Payload["colors"] = ColorsAll
        else:
            print("Warning: colors length mismatch. Colors skipped.")

    SaveNpZAtomic(ProjectionInputPath, Payload)


def SetAxesEqual(Ax) -> None:
    XLim = Ax.get_xlim3d()
    YLim = Ax.get_ylim3d()
    ZLim = Ax.get_zlim3d()

    XRange = abs(XLim[1] - XLim[0])
    YRange = abs(YLim[1] - YLim[0])
    ZRange = abs(ZLim[1] - ZLim[0])

    Radius = 0.5 * max([XRange, YRange, ZRange])

    XMid = np.mean(XLim)
    YMid = np.mean(YLim)
    ZMid = np.mean(ZLim)

    Ax.set_xlim3d([XMid - Radius, XMid + Radius])
    Ax.set_ylim3d([YMid - Radius, YMid + Radius])
    Ax.set_zlim3d([ZMid - Radius, ZMid + Radius])


def PlotUsability3D(
    Points: np.ndarray,
    Prob: np.ndarray,
    Usability: np.ndarray,
    ProbThreshold: float,
    MaxPoints: int = 250000,
) -> None:
    Count = len(Points)
    if Count > MaxPoints:
        Idx = np.random.choice(Count, MaxPoints, replace=False)
        Pts = Points[Idx]
        Pr = Prob[Idx]
        Us = Usability[Idx]
    else:
        Pts = Points
        Pr = Prob
        Us = Usability

    HighProb = Pr >= ProbThreshold
    LowProb = ~HighProb

    UsableMask = HighProb & (Us == 1)
    NonUsableMask = HighProb & (Us == -1)

    Figure = plt.figure(figsize=(9, 9))
    Ax = Figure.add_subplot(111, projection="3d")

    Ax.scatter(Pts[LowProb, 0], Pts[LowProb, 1], Pts[LowProb, 2], c="lightgrey", s=0.3, alpha=0.3, label="Low prob")
    Ax.scatter(Pts[UsableMask, 0], Pts[UsableMask, 1], Pts[UsableMask, 2], c="lime", s=0.6, alpha=0.8, label="Usable")
    Ax.scatter(
        Pts[NonUsableMask, 0],
        Pts[NonUsableMask, 1],
        Pts[NonUsableMask, 2],
        c="red",
        s=1.2,
        alpha=1.0,
        label="Non-usable",
    )

    Ax.set_title(f"3D Usability (prob >= {ProbThreshold})")
    Ax.legend(loc="upper right")
    Ax.view_init(elev=20, azim=35)
    SetAxesEqual(Ax)
    plt.tight_layout()
    plt.show()

    print("High-prob usable:", int(UsableMask.sum()))
    print("High-prob non-usable:", int(NonUsableMask.sum()))
    print("Low-prob:", int(LowProb.sum()))


def BuildArgParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(description="Pipeline stage: assign 3D non-usability labels from a 2D mask.")
    Parser.add_argument("--projection_input", type=str, default=DefaultProjectionInputPath, help="Projection NPZ path.")
    Parser.add_argument("--nonuse_mask_path", type=str, required=True, help="Binary non-usability mask image path.")
    Parser.add_argument("--show_2d_plot", action="store_true", help="Show 2D projection usability plot.")
    Parser.add_argument("--show_3d_plots", action="store_true", help="Show 3D usability plots.")
    return Parser


def RunNonUsabilityStage() -> None:
    Parser = BuildArgParser()
    Args = Parser.parse_args()

    ProjectionInputPath = Path(Args.projection_input)
    if not ProjectionInputPath.exists():
        raise FileNotFoundError(f"Projection input not found: {ProjectionInputPath}")

    StageData = LoadProjectionStageData(ProjectionInputPath)
    Rgb = StageData["Rgb"]
    K = StageData["K"]
    W2C = StageData["W2C"]
    Pw = StageData["Pw"]
    Prob = StageData["Prob"]
    Colors = StageData["Colors"]

    NonUseMaskPath = Path(Args.nonuse_mask_path)
    NonUseMaskBin = LoadBinaryMask(NonUseMaskPath, TargetShape=Rgb.shape[:2])

    Result = AssignUsabilityFromMask(
        Pw=Pw,
        Prob=Prob,
        Rgb=Rgb,
        K=K,
        W2C=W2C,
        NonUseMaskBin=NonUseMaskBin,
    )

    Valid = Result["Valid"]
    OnNonUse = Result["OnNonUse"]
    Usability = Result["Usability"]

    print("Total points:", len(Pw))
    print("Valid projected:", int(Valid.sum()))
    print("On non-use mask:", int(OnNonUse.sum()))
    print("Non-usable points assigned:", int((Usability == -1).sum()))
    print("Still usable points:", int((Usability == 1).sum()))

    if Args.show_2d_plot:
        PlotProjectionUsability(
            Rgb=Rgb,
            Ui=Result["Ui"],
            Vi=Result["Vi"],
            Valid=Valid,
            Usability=Usability,
        )

    AppendUsabilityToProjectionNpZ(
        ProjectionInputPath=ProjectionInputPath,
        Pw=Pw,
        Prob=Prob,
        Usability=Usability,
        Colors=Colors,
    )
    print("Updated projection NPZ with usability/master keys:", str(ProjectionInputPath))

    if Args.show_3d_plots:
        PlotUsability3D(Pw, Prob, Usability, ProbThreshold=0.3)
        PlotUsability3D(Pw, Prob, Usability, ProbThreshold=0.6)
        PlotUsability3D(Pw, Prob, Usability, ProbThreshold=0.9)

    print("Single-file pipeline mode: no sidecar PT/NPZ artifacts written.")


if __name__ == "__main__":
    RunNonUsabilityStage()