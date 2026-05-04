"""
Compare intermediate diffusion steps against a ground-truth mask.

For each saved step_T.pt file:
  1. Extract a mesh from the voxel grid using FlexiCubes.
  2. Render a silhouette using the camera pose and intrinsics saved in the file.
  3. Compute IoU against the ground-truth mask.
  4. Save a side-by-side comparison image.

Usage:
    uv run python scripts/compare_steps.py \
        --steps-dir .cache/ss_steps \
        --mask images/.../14.png \
        --output outputs/step_comparison
"""

import glob
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import (
    FlexiCubes,
)

# ── Grid construction ─────────────────────────────────────────────────────────


def _build_flexicubes_grid(
    N: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the inputs FlexiCubes needs for a regular N×N×N voxel grid.

    Vertex positions follow the pipeline's canonical voxel space:
        position = coord / N - 0.5
    so the grid spans roughly [-0.5, 0.5] in each axis, matching how
    the pipeline normalises sparse-structure coordinates.

    Returns:
        voxelgrid_vertices: (N^3, 3) float tensor of vertex positions.
        cube_idx: ((N-1)^3, 8) long tensor of corner indices per cube.
            Corner order matches FlexiCubes' cube_corners convention:
            (0,0,0), (1,0,0), (0,1,0), (1,1,0),
            (0,0,1), (1,0,1), (0,1,1), (1,1,1).
    """
    xs = torch.arange(N, device=device, dtype=torch.float) / N - 0.5
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    voxelgrid_vertices = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    # Flat vertex index array so we can look up cube corners easily
    flat_idx = torch.arange(N * N * N, device=device).reshape(N, N, N)
    i, j, k = torch.meshgrid(
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        indexing="ij",
    )
    i, j, k = i.reshape(-1), j.reshape(-1), k.reshape(-1)
    cube_idx = torch.stack(
        [
            flat_idx[i, j, k],
            flat_idx[i + 1, j, k],
            flat_idx[i, j + 1, k],
            flat_idx[i + 1, j + 1, k],
            flat_idx[i, j, k + 1],
            flat_idx[i + 1, j, k + 1],
            flat_idx[i, j + 1, k + 1],
            flat_idx[i + 1, j + 1, k + 1],
        ],
        dim=-1,
    )
    return voxelgrid_vertices, cube_idx


# ── Mesh extraction ───────────────────────────────────────────────────────────


def extract_mesh(
    ss_grid: torch.Tensor, device: str = "cpu"
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract a triangle mesh from a decoded sparse-structure voxel grid.

    ss_grid is the output of ss_decoder, shape (1, C, N, N, N).
    Channel 0 is treated as an occupancy/SDF field where positive = inside.
    FlexiCubes expects negative = inside, so we negate before passing in.

    Returns (verts, faces) tensors, or None if the grid is empty.
    """
    dev = torch.device(device)
    N = ss_grid.shape[-1]  # 64 (ss_decoder upsamples 16→64)

    # Negate: our grid is positive-inside, FlexiCubes needs negative-inside
    scalar_field = -ss_grid[0, 0].float().to(dev).reshape(-1)

    if (scalar_field < 0).sum() == 0:
        return None  # nothing occupied

    voxelgrid_vertices, cube_idx = _build_flexicubes_grid(N, dev)
    fc = FlexiCubes(device=dev)

    with torch.no_grad():
        verts, faces, _, _ = fc(
            voxelgrid_vertices, scalar_field, cube_idx, resolution=N - 1
        )

    if verts.shape[0] == 0:
        return None

    return verts, faces


# ── Silhouette rendering ──────────────────────────────────────────────────────


def render_silhouette(
    verts: torch.Tensor,
    faces: torch.Tensor,
    step_data: dict,
    image_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Render a binary silhouette of the mesh using the camera saved in step_data.

    step_data must contain:
        pose_rotation:    (4,) quaternion (w, x, y, z) — object-to-camera rotation
        pose_translation: (3,) translation in camera space
        pose_scale:       (3,) or (1, 3) uniform object scale (optional)
        intrinsics:       (3, 3) normalised camera matrix (fx/W, fy/H, cx/W, cy/H)

    Returns a (image_size, image_size) bool array: True = foreground.
    """
    dev = torch.device(device)

    # Scale vertices from canonical object space to camera/scene space
    scaled_verts = verts.to(dev)
    if "pose_scale" in step_data:
        scale = step_data["pose_scale"].float().to(dev).mean()
        scaled_verts = scaled_verts * scale

    mesh = Meshes(
        verts=scaled_verts.unsqueeze(0),
        faces=faces.to(dev).long().unsqueeze(0),
    )

    # Camera pose: quaternion → rotation matrix
    quat = step_data["pose_rotation"].reshape(1, 4).to(dev)
    R = quaternion_to_matrix(quat)  # (1, 3, 3)
    T = step_data["pose_translation"].reshape(1, 3).to(dev)

    # Intrinsics are normalised (0–1 range); convert to pixel units for pytorch3d
    K = step_data["intrinsics"].to(dev)  # (3, 3)
    fx_px = K[0, 0] * image_size
    fy_px = K[1, 1] * image_size
    cx_px = K[0, 2] * image_size
    cy_px = K[1, 2] * image_size

    cameras = PerspectiveCameras(
        focal_length=((fx_px, fy_px),),
        principal_point=((cx_px, cy_px),),
        R=R,
        T=T,
        in_ndc=False,
        image_size=((image_size, image_size),),
        device=dev,
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=50,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    with torch.no_grad():
        output = renderer(mesh)  # (1, H, W, 4)

    alpha = output[0, ..., 3].cpu().numpy()
    return alpha > 0.5


# ── Mask / metric helpers ─────────────────────────────────────────────────────


def load_gt_mask(mask_path: str, size: int = 256) -> np.ndarray:
    """
    Load a ground-truth mask PNG and resize it to (size, size).
    Uses the alpha channel if present (matches how the pipeline loads masks).
    Returns a bool array of shape (size, size).
    """
    arr = np.array(Image.open(mask_path))
    mask = arr > 0
    if mask.ndim == 3:
        mask = mask[..., -1]  # alpha channel
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), Image.NEAREST
    )
    return np.array(resized) > 0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection) / float(union + 1e-8)


# ── Visualisation ─────────────────────────────────────────────────────────────


def save_comparison(
    silhouette: np.ndarray,
    gt_mask: np.ndarray,
    t_step: float,
    iou: float,
    output_path: str,
) -> None:
    """
    Save a 3-panel figure: predicted silhouette | GT mask | overlap.
    Overlap colour code: white = TP, red = FP, blue = FN.
    """
    overlap = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlap[silhouette & gt_mask] = [255, 255, 255]  # TP
    overlap[silhouette & ~gt_mask] = [255, 80, 80]  # FP
    overlap[~silhouette & gt_mask] = [80, 80, 255]  # FN

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f"t={t_step:.3f}  IoU={iou:.3f}")
    axes[0].imshow(silhouette, cmap="gray")
    axes[0].set_title("Predicted")
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("GT mask")
    axes[2].imshow(overlap)
    axes[2].set_title("Overlap  (W=TP  R=FP  B=FN)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(
    steps_dir: str,
    mask_path: str,
    output_dir: str,
    image_size: int = 256,
    device: str = "cpu",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(steps_dir, "step_*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No step_*.pt files found in {steps_dir}")

    gt_mask = load_gt_mask(mask_path, size=image_size)
    print(f"GT mask: {gt_mask.sum()} foreground pixels")

    for pt_path in pt_files:
        step_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        t_step = float(step_data["t_step"])
        print(f"t={t_step:.3f}", end="  ")

        mesh = extract_mesh(step_data["ss_grid"], device=device)
        if mesh is None:
            print("no mesh")
            continue

        verts, faces = mesh
        silhouette = render_silhouette(verts, faces, step_data, image_size, device)
        iou = compute_iou(silhouette, gt_mask)
        print(f"IoU={iou:.3f}")

        out_path = os.path.join(output_dir, f"step_{t_step:.3f}.png")
        save_comparison(silhouette, gt_mask, t_step, iou, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-dir", default=".cache/ss_steps")
    parser.add_argument(
        "--mask",
        default="notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png",
        dest="mask_path",
    )
    parser.add_argument(
        "--output", default="outputs/step_comparison", dest="output_dir"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    main(
        steps_dir=args.steps_dir,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        device=args.device,
    )
