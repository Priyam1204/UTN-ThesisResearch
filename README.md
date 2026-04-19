# VLM-based Object Intrinsics Estimation for Cooperative Object Transportation

This repository contains the implementation for a thesis pipeline that estimates object properties from a single RGB image and projects them into 3D reconstruction for downstream robotics tasks.

The main objective is to enrich a reconstructed 3D object with:
- material labels using vision-language model,
- usability labels depending on task prompt.

The thesis was done at the [University of Technology Nuremberg](https://www.utn.de/) as part of the [MSc AI & Robotics](https://www.utn.de/studium/ai-robotics/) program.


## Pipeline

<p align="center">
  <img src="PipelineArch.png" />
</p>

<p align="center">
Pipeline Architecture.
</p>


The complete pipeline with fixed task prompts is [Pipeline.py](Pipeline.py).

Final output:
- `results/projection/<image_stem>.npz`

This NPZ is updated throughout the pipeline and includes projection data, material labels, and usability info.


### Stage Order

1. [MaterialPrediction.py](MaterialPrediction.py)
- Input: image
- Output: `results/materials/<image_stem>.json`

2. [AutomaticSegmentation.py](AutomaticSegmentation.py)
- Input: image + material JSON description
- Output: `results/masks/<image_stem>_object_mask.png`

3. [SAM3DReconstruction.py](SAM3DReconstruction.py)
- Input: image + object mask
- Output: `results/reconstruction/<image_stem>.ply`

4. [PointCloudProjection.py](PointCloudProjection.py)
- Input: image + reconstruction PLY + camera intrinsics/extrinsics
- Output: `results/projection/<image_stem>.npz`

5. [ClipFeature.py](ClipFeature.py)
- Input: projection NPZ
- Action: appends CLIP sparse features into the same NPZ

6. [ClipFeatureMatching.py](ClipFeatureMatching.py)
- Input: projection NPZ + material JSON
- Action: assigns CLIP-based material label per retained point and appends `clip_label`

7. [NonUsabilityMask.py](NonUsabilityMask.py)
- Input: image
- Output: `results/masks/<image_stem>_nonusable_mask.png`

8. [3DNonUsability.py](3DNonUsability.py)
- Input: projection NPZ + non-usability mask
- Action: appends 3D usability data into the same NPZ

## Requirements

- `GEMINI_API_KEY` (env variable)
- SAM checkpoint: `checkpoints/sam_vit_h_4b8939.pth`
- Camera files (same folder as image):
  - `intrinsic.txt`
  - `extrinsic.txt`


## Quick Start Checklist

1. Activate your Python environment.
2. Set `GEMINI_API_KEY`.
3. Setup [SAM3D](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)**

4. Install additional dependencies for this pipeline
```bash
pip install -r requirements.thesis.txt 
```

5. Ensure checkpoint file exists at `checkpoints/sam_vit_h_4b8939.pth`.
6. Place `intrinsic.txt` and `extrinsic.txt` next to your input image.
7. Run:

```bash
python Pipeline.py --image_path "path/to/image.jpg"
```



## Point Cloud Fusion

If a single view does not capture all relevant parts of the object the 3D reconstruction will also contain partial information, then run a fusion step after the initial pipeline.

- Script: [PointCloudFusion.py](PointCloudFusion.py)
- Purpose: fuse point-level outputs into a denser, consolidated NPZ for downstream analysis.

## Pipeline Example

### 🔹 1. Input Image

<p align="center">
  <img src="Webots\controllers\tb4\ninthangle\tb4_image.jpg" width="45%" />
  <img src="Webots\controllers\tb4\secondangle\tb4_image.jpg" width="45%" />

</p>

---

### 🔹 2. Pipeline Output

<p align="center">
  <img src="boundary_material_rotation_updated.gif" width="45%" />
  <img src="usability_rotation.gif" width="45%" />
</p>



---

### 🔹 3. Output Usage

Material labels enable estimation of physical properties; here, they are used to compute the center of mass for robotic transportation. The right shows the usable contact footprint for swarm-based carrying. This information can further support improved swarm formation strategies (e.g., MARL-based coordination).

<p align="center">
  <img src="OutputUsage.png" />
</p>

<p align="center">
Left: geometric vs material-aware center of mass. Right: usable contact regions excluding fragile surfaces (glass, cardboard, PCB).
</p>


## Acknowledgement

This repository is forked from [SAM3D](https://github.com/facebookresearch/sam-3d-objects) and builds upon its 3D reconstruction by adding VLM-based reasoning, CLIP feature alignment, and usability-aware 3D labeling.