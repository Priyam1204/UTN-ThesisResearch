import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData


DefaultOutDir = "PointCloudProjectionOutput"
DefaultPipelineOutput = "projection_stage_output.npz"


ForwardAxis = np.array([1, 0, 0], dtype=np.float32)
RightAxis = np.array([0, -1, 0], dtype=np.float32)
UpAxis = np.array([0, 0, -1], dtype=np.float32)


def LoadRgbImage(PathText: str) -> np.ndarray:
    return np.array(Image.open(PathText).convert("RGB"))


def LoadBinaryMask(PathText: str, Threshold: int = 127) -> np.ndarray:
    MaskGray = np.array(Image.open(PathText).convert("L"))
    return MaskGray > Threshold


def LoadIntrinsicMatrix(IntrinsicPath: str) -> np.ndarray:
    Rows = []
    with open(IntrinsicPath, "r", encoding="utf-8") as FileHandle:
        for RawLine in FileHandle:
            Line = RawLine.strip().replace("[", "").replace("]", "")
            if not Line or Line.startswith("K") or Line.startswith("0 0 1"):
                continue
            Parts = Line.split()
            if len(Parts) == 3:
                Rows.append([float(Parts[0]), float(Parts[1]), float(Parts[2])])

    if len(Rows) == 2:
        Rows.append([0.0, 0.0, 1.0])

    K = np.array(Rows, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Parsed intrinsic matrix has shape {K.shape}, expected (3, 3).")
    return K


def LoadExtrinsicMatrix4x4(ExtrinsicPath: str) -> np.ndarray:
    Rows = []
    with open(ExtrinsicPath, "r", encoding="utf-8") as FileHandle:
        for RawLine in FileHandle:
            Line = RawLine.strip()
            if not Line or "Extrinsic" in Line:
                continue
            Parts = Line.split()
            if len(Parts) == 4:
                Rows.append([float(Value) for Value in Parts])

    Matrix3x4 = np.array(Rows, dtype=np.float32)
    if Matrix3x4.shape != (3, 4):
        raise ValueError(f"Parsed extrinsic matrix has shape {Matrix3x4.shape}, expected (3, 4).")

    Matrix4x4 = np.eye(4, dtype=np.float32)
    Matrix4x4[:3, :4] = Matrix3x4
    return Matrix4x4


def BuildRotationZ(Degrees: float) -> np.ndarray:
    Angle = np.deg2rad(Degrees)
    CosValue = np.cos(Angle)
    SinValue = np.sin(Angle)
    return np.array(
        [
            [CosValue, -SinValue, 0.0],
            [SinValue, CosValue, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def BuildWorldPoints(
    PointsRaw: np.ndarray,
    ObjectCenterWorld: np.ndarray,
    YawDeg: float,
    ScaleXY: float = 1.0,
    ScaleZ: float = 1.0,
    LiftMode: str = "q01",
) -> np.ndarray:
    Points = PointsRaw.astype(np.float32).copy()

    Points[:, 0] -= Points[:, 0].mean()
    Points[:, 1] -= Points[:, 1].mean()
    Points[:, 0] *= ScaleXY
    Points[:, 1] *= ScaleXY
    Points[:, 2] *= ScaleZ

    Points = Points @ BuildRotationZ(YawDeg).T

    if LiftMode == "none":
        ZOffset = 0.0
    elif LiftMode == "min":
        ZOffset = float(Points[:, 2].min())
    else:
        ZOffset = float(np.quantile(Points[:, 2], 0.01))

    Points[:, 2] -= ZOffset
    return Points + ObjectCenterWorld.reshape(1, 3).astype(np.float32)


def ProjectWithBasis(
    PointsWorld: np.ndarray,
    K: np.ndarray,
    WorldToCamera: np.ndarray,
    Forward: np.ndarray,
    Right: np.ndarray,
    Up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Points = np.asarray(PointsWorld, dtype=np.float32).reshape(-1, 3)
    Count = Points.shape[0]

    PointsH = np.hstack([Points, np.ones((Count, 1), dtype=np.float32)])
    PointsCamera = (WorldToCamera @ PointsH.T).T[:, :3]

    Depth = PointsCamera @ Forward
    X = PointsCamera @ Right
    Y = PointsCamera @ Up

    SafeDepth = np.where(np.abs(Depth) < 1e-8, 1e-8, Depth)
    U = K[0, 0] * (X / SafeDepth) + K[0, 2]
    V = K[1, 1] * (Y / SafeDepth) + K[1, 2]
    return U, V, Depth


def ComputeVisibleIndicesWithZBuffer(
    U: np.ndarray,
    V: np.ndarray,
    Depth: np.ndarray,
    Width: int,
    Height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    InFront = Depth > 1e-3
    InImage = InFront & (U >= 0) & (U < Width) & (V >= 0) & (V < Height)
    CandidateIdx = np.where(InImage)[0]

    if CandidateIdx.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    Uu = np.clip(np.rint(U[CandidateIdx]).astype(np.int32), 0, Width - 1)
    Vv = np.clip(np.rint(V[CandidateIdx]).astype(np.int32), 0, Height - 1)
    DepthVisible = Depth[CandidateIdx]

    PixelId = Vv.astype(np.int64) * Width + Uu.astype(np.int64)

    SortOrder = np.lexsort((DepthVisible, PixelId))
    PixelSorted = PixelId[SortOrder]

    FirstPerPixel = np.ones_like(PixelSorted, dtype=bool)
    FirstPerPixel[1:] = PixelSorted[1:] != PixelSorted[:-1]
    KeepOrder = SortOrder[FirstPerPixel]

    VisibleIdx = CandidateIdx[KeepOrder]
    return VisibleIdx, Uu[KeepOrder], Vv[KeepOrder]


def FindBestYaw(
    PointsRaw: np.ndarray,
    ObjectCenterWorld: np.ndarray,
    K: np.ndarray,
    WorldToCamera: np.ndarray,
    Mask: np.ndarray,
    Width: int,
    Height: int,
    YawCandidates: np.ndarray,
    LiftMode: str = "q01",
) -> float:
    BestYaw = 0.0
    BestScore = -1

    for Yaw in YawCandidates:
        Pw = BuildWorldPoints(
            PointsRaw=PointsRaw,
            ObjectCenterWorld=ObjectCenterWorld,
            YawDeg=float(Yaw),
            LiftMode=LiftMode,
        )

        U, V, D = ProjectWithBasis(
            PointsWorld=Pw,
            K=K,
            WorldToCamera=WorldToCamera,
            Forward=ForwardAxis,
            Right=RightAxis,
            Up=UpAxis,
        )

        IdxVisible, Uu, Vv = ComputeVisibleIndicesWithZBuffer(
            U=U,
            V=V,
            Depth=D,
            Width=Width,
            Height=Height,
        )

        if Uu.size == 0:
            continue

        OnMask = Mask[Vv, Uu] > 0
        Score = int(OnMask.sum())

        if Score > BestScore:
            BestScore = Score
            BestYaw = float(Yaw)

    return BestYaw


def ComputeDepthProbability(
    Depth: np.ndarray,
    Yaw: float,
    InvertRange: Tuple[float, float] = (170.0, 190.0),
) -> np.ndarray:
    DepthArray = np.asarray(Depth, dtype=np.float32)
    DepthMin, DepthMax = np.percentile(DepthArray, [1, 99])
    DepthNorm = (DepthArray - DepthMin) / (DepthMax - DepthMin + 1e-8)

    if InvertRange[0] <= Yaw <= InvertRange[1]:
        Probability = DepthNorm
    else:
        Probability = 1.0 - DepthNorm

    return np.clip(Probability, 0.0, 1.0)


def LoadPlyPointsAndColors(PlyPath: str) -> Tuple[np.ndarray, np.ndarray]:
    HarmonicConstant = 0.28209479177387814

    Ply = PlyData.read(PlyPath)
    VertexData = Ply["vertex"].data
    FieldNames = set(VertexData.dtype.names)

    Points = np.stack([VertexData["x"], VertexData["y"], VertexData["z"]], axis=1).astype(np.float32)

    HasDcFields = all(Field in FieldNames for Field in ["f_dc_0", "f_dc_1", "f_dc_2"])
    if not HasDcFields:
        raise ValueError("PLY is missing f_dc_* fields needed for color reconstruction.")

    Dc = np.stack([VertexData["f_dc_0"], VertexData["f_dc_1"], VertexData["f_dc_2"]], axis=1).astype(np.float32)
    Colors = np.clip(HarmonicConstant * Dc + 0.5, 0.0, 1.0).astype(np.float32)
    return Points, Colors


def SetAxesEqual(Axes) -> None:
    XLimits = Axes.get_xlim3d()
    YLimits = Axes.get_ylim3d()
    ZLimits = Axes.get_zlim3d()

    XRange = abs(XLimits[1] - XLimits[0])
    YRange = abs(YLimits[1] - YLimits[0])
    ZRange = abs(ZLimits[1] - ZLimits[0])

    Radius = 0.5 * max([XRange, YRange, ZRange])
    XMiddle = np.mean(XLimits)
    YMiddle = np.mean(YLimits)
    ZMiddle = np.mean(ZLimits)

    Axes.set_xlim3d([XMiddle - Radius, XMiddle + Radius])
    Axes.set_ylim3d([YMiddle - Radius, YMiddle + Radius])
    Axes.set_zlim3d([ZMiddle - Radius, ZMiddle + Radius])


def SaveOrShowFigure(Figure, OutputPath: Path, ShowPlots: bool) -> None:
    Figure.savefig(OutputPath, dpi=200, bbox_inches="tight")
    if ShowPlots:
        plt.show()
    else:
        plt.close(Figure)


def PlotColoredPointCloudPreview(
    Points: np.ndarray,
    Colors: np.ndarray,
    MaxPoints: int,
    OutputPath: Path,
    ShowPlots: bool,
) -> None:
    Count = Points.shape[0]
    Take = min(Count, MaxPoints)
    SampleIdx = np.random.choice(Count, Take, replace=False) if Count > Take else np.arange(Count)

    Figure = plt.figure(figsize=(6, 6))
    Axes = Figure.add_subplot(111, projection="3d")
    Axes.scatter(Points[SampleIdx, 0], Points[SampleIdx, 1], Points[SampleIdx, 2], c=Colors[SampleIdx], s=0.5)
    Axes.set_title("Colored PCD preview")
    SetAxesEqual(Axes)
    SaveOrShowFigure(Figure, OutputPath, ShowPlots)


def PlotProjectionOverlay(
    Rgb: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    Indices: np.ndarray,
    ColorsOrProb: np.ndarray,
    OutputPath: Path,
    Title: str,
    ShowPlots: bool,
    Colormap: str = "",
) -> None:
    Figure = plt.figure(figsize=(6, 6))
    plt.imshow(Rgb)

    if Indices.size > 0:
        if Colormap:
            plt.scatter(U[Indices], V[Indices], s=0.8, c=ColorsOrProb[Indices], cmap=Colormap)
        else:
            plt.scatter(U[Indices], V[Indices], s=0.8, c=ColorsOrProb[Indices])

    plt.title(Title)
    plt.axis("off")
    plt.tight_layout()
    SaveOrShowFigure(Figure, OutputPath, ShowPlots)


def PlotProbabilityPointCloud(
    PointsWorld: np.ndarray,
    Probability: np.ndarray,
    OutputPath: Path,
    ShowPlots: bool,
) -> None:
    Figure = plt.figure(figsize=(6, 6))
    Axes = Figure.add_subplot(111, projection="3d")
    Axes.scatter(PointsWorld[:, 0], PointsWorld[:, 1], PointsWorld[:, 2], c=Probability, cmap="viridis", s=1)
    Axes.set_title("3D point cloud colored by probability")
    SetAxesEqual(Axes)
    SaveOrShowFigure(Figure, OutputPath, ShowPlots)

def BuildArgParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(
        description="Projection stage: project 3D points to image and compute point probability."
    )

    Parser.add_argument("--rgb", type=str, required=True, help="RGB image path.")
    Parser.add_argument("--mask", type=str, required=True, help="Binary mask image path.")
    Parser.add_argument("--ply", type=str, required=True, help="Input PLY path.")
    Parser.add_argument("--intrinsics", type=str, required=True, help="Camera intrinsics txt path.")
    Parser.add_argument("--extrinsics", type=str, required=True, help="Camera extrinsics txt path.")

    Parser.add_argument(
        "--object-center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Object center in world coordinates.",
    )

    YawGroup = Parser.add_mutually_exclusive_group()
    YawGroup.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Fixed yaw in degrees. If provided, this yaw is used directly.",
    )
    YawGroup.add_argument(
        "--yaw-search",
        action="store_true",
        help="Search for the best yaw by maximizing projected visible points inside the mask.",
    )

    Parser.add_argument(
        "--yaw-search-start",
        type=float,
        default=0.0,
        help="Yaw search start angle in degrees.",
    )
    Parser.add_argument(
        "--yaw-search-stop",
        type=float,
        default=360.0,
        help="Yaw search stop angle in degrees.",
    )
    Parser.add_argument(
        "--yaw-search-step",
        type=float,
        default=5.0,
        help="Yaw search step in degrees.",
    )

    Parser.add_argument(
        "--lift-mode",
        type=str,
        default="q01",
        choices=["q01", "min", "none"],
        help="Z lift mode for world placement.",
    )

    Parser.add_argument("--preview-max-points", type=int, default=50000, help="Max preview points.")
    Parser.add_argument("--out-dir", type=str, default=DefaultOutDir, help="Plot output directory.")
    Parser.add_argument("--save-pipeline", type=str, default=DefaultPipelineOutput, help="Pipeline NPZ output.")
    Parser.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    return Parser


def PointCloudProjection() -> None:
    Args = BuildArgParser().parse_args()
    np.random.seed(0)

    OutputDir = Path(Args.out_dir)
    OutputDir.mkdir(parents=True, exist_ok=True)

    ObjectCenterWorld = np.array(Args.object_center, dtype=np.float32)

    PointsRaw, Colors = LoadPlyPointsAndColors(Args.ply)
    print("points:", PointsRaw.shape, "colors:", Colors.shape, "range:", (float(Colors.min()), float(Colors.max())))

    PlotColoredPointCloudPreview(
        Points=PointsRaw,
        Colors=Colors,
        MaxPoints=int(Args.preview_max_points),
        OutputPath=OutputDir / "pcd_preview.png",
        ShowPlots=Args.show_plots,
    )

    Rgb = LoadRgbImage(Args.rgb)
    Mask = LoadBinaryMask(Args.mask)
    Height, Width = Rgb.shape[:2]

    K = LoadIntrinsicMatrix(Args.intrinsics)
    TRaw = LoadExtrinsicMatrix4x4(Args.extrinsics)
    WorldToCamera = TRaw.astype(np.float32)

    if Args.yaw_search:
        YawCandidates = np.arange(
            Args.yaw_search_start,
            Args.yaw_search_stop,
            Args.yaw_search_step,
            dtype=np.float32,
        )
        if YawCandidates.size == 0:
            raise ValueError("Yaw search produced no candidates. Check yaw-search-start/stop/step.")
        YawUsed = FindBestYaw(
            PointsRaw=PointsRaw,
            ObjectCenterWorld=ObjectCenterWorld,
            K=K,
            WorldToCamera=WorldToCamera,
            Mask=Mask,
            Width=Width,
            Height=Height,
            YawCandidates=YawCandidates,
            LiftMode=Args.lift_mode,
        )
        print(f"Best yaw found by search: {YawUsed:.2f}")
    else:
        YawUsed = float(Args.yaw) if Args.yaw is not None else 90.0
        print(f"Using fixed yaw: {YawUsed:.2f}")

    PointsWorld = BuildWorldPoints(
        PointsRaw=PointsRaw,
        ObjectCenterWorld=ObjectCenterWorld,
        YawDeg=YawUsed,
        ScaleXY=1.0,
        ScaleZ=1.0,
        LiftMode=Args.lift_mode,
    )

    U, V, Depth = ProjectWithBasis(
        PointsWorld=PointsWorld,
        K=K,
        WorldToCamera=WorldToCamera,
        Forward=ForwardAxis,
        Right=RightAxis,
        Up=UpAxis,
    )

    IdxVisible, UuVisible, VvVisible = ComputeVisibleIndicesWithZBuffer(
        U=U,
        V=V,
        Depth=Depth,
        Width=Width,
        Height=Height,
    )

    OnMask = (Mask[VvVisible, UuVisible] > 0) if UuVisible.size > 0 else np.array([], dtype=bool)
    IdxOnMask = IdxVisible[OnMask] if IdxVisible.size > 0 and OnMask.size > 0 else np.array([], dtype=np.int64)

    print("visible:", int(IdxVisible.size), "on-mask:", int(OnMask.sum()) if OnMask.size > 0 else 0)

    PlotProjectionOverlay(
        Rgb=Rgb,
        U=U,
        V=V,
        Indices=IdxVisible,
        ColorsOrProb=Colors,
        OutputPath=OutputDir / "overlay_rgb_color_visible.png",
        Title=f"Visible projected points (yaw={YawUsed:.2f})",
        ShowPlots=Args.show_plots,
    )

    PlotProjectionOverlay(
        Rgb=Rgb,
        U=U,
        V=V,
        Indices=IdxOnMask,
        ColorsOrProb=Colors,
        OutputPath=OutputDir / "overlay_rgb_color_onmask.png",
        Title="Visible + on-mask projected points",
        ShowPlots=Args.show_plots,
    )

    Probability = ComputeDepthProbability(Depth=Depth, Yaw=YawUsed)

    PlotProjectionOverlay(
        Rgb=Rgb,
        U=U,
        V=V,
        Indices=IdxVisible,
        ColorsOrProb=Probability,
        OutputPath=OutputDir / "overlay_probability.png",
        Title=f"Projected points colored by depth-prob (yaw={YawUsed:.2f})",
        ShowPlots=Args.show_plots,
        Colormap="viridis",
    )

    PlotProbabilityPointCloud(
        PointsWorld=PointsWorld,
        Probability=Probability,
        OutputPath=OutputDir / "pcd_probability_3d.png",
        ShowPlots=Args.show_plots,
    )

    PipelineOutput = Path(Args.save_pipeline)
    np.savez_compressed(
        PipelineOutput,
        rgb=Rgb,
        mask=Mask.astype(np.uint8),
        K=K.astype(np.float32),
        T_raw=TRaw.astype(np.float32),
        w2c=WorldToCamera.astype(np.float32),
        points=PointsRaw.astype(np.float32),
        colors=Colors.astype(np.float32),
        Pw=PointsWorld.astype(np.float32),
        u=U.astype(np.float32),
        v=V.astype(np.float32),
        d=Depth.astype(np.float32),
        prob=Probability.astype(np.float32),
        idx_vis=IdxVisible.astype(np.int64),
        uu_vis=UuVisible.astype(np.int32),
        vv_vis=VvVisible.astype(np.int32),
        yaw_used=np.array([YawUsed], dtype=np.float32),
        object_center_world=ObjectCenterWorld.astype(np.float32),
    )
    print("Saved pipeline handoff:", PipelineOutput.resolve())


if __name__ == "__main__":
    PointCloudProjection()