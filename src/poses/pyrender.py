import numpy as np
import pyrender
import trimesh
import os
from PIL import Image
import numpy as np
import os.path as osp
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from src.utils.trimesh_utils import as_mesh
from src.utils.trimesh_utils import get_obj_diameter
os.environ["DISPLAY"] = ":1"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def _resolve(p):
    p = Path(p)
    return str((Path(__file__).resolve().parents[2] / p).resolve() if not p.is_absolute() else p)

def render(
    mesh,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    light_itensity=0.6,
    is_tless=False,
    re_center_transform=np.eye(4),
):
    # camera pose is fixed as np.eye(4)
    cam_pose = np.eye(4)
    # convert openCV camera
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1
    # create scene config
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])  # np.array([1.0, 1.0, 1.0, 1.0])
    if light_itensity != 0.6:
        ambient_light = np.array([1.0, 1.0, 1.0, 1.0])
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=ambient_light
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=light_itensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=cam_pose)

    # create camera and render engine
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
    )
    scene.add(camera, pose=cam_pose)
    render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

    for idx_frame in range(obj_poses.shape[0]):
        scene.set_pose(cad_node, obj_poses[idx_frame] @ re_center_transform)
        rgb, depth = render_engine.render(scene, pyrender.constants.RenderFlags.RGBA)
        rgb = Image.fromarray(np.uint8(rgb))
        rgb.save(osp.join(output_dir, f"{idx_frame:06d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "config.json"))
    # legacy positional args still supported but optional now:
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the pose npy")
    parser.add_argument("output_dir", nargs="?", help="Where to save PNGs")
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Unused")
    parser.add_argument("light_itensity", nargs="?", type=float, default=0.6)
    parser.add_argument("radius", nargs="?", type=float, default=1.0)
    args = parser.parse_args()

    # ---- load config ----
    with open(args.config) as f:
        cfg = json.load(f)

    cad_path      = _resolve(cfg["paths"]["cad_path"]     if args.cad_path is None   else args.cad_path)
    poses_path    = _resolve(cfg["paths"]["poses_path"]   if args.obj_pose is None   else args.obj_pose)
    output_dir    = _resolve(cfg["paths"]["templates_out"]if args.output_dir is None else args.output_dir)

    gpus_devices  = cfg["render"].get("gpus_devices", "0") if args.gpus_devices is None else args.gpus_devices
    light_int     = cfg["render"].get("lighting_intensity", 0.6) if args.light_itensity == 0.6 else args.light_itensity
    radius        = cfg["render"].get("radius", 1.0) if args.radius == 1.0 else args.radius
    intrinsic     = np.array(cfg["render"]["intrinsics"], dtype=np.float32)
    img_size      = cfg["render"]["img_size"]  # [H,W]

    print(f"[render] cad={cad_path}\n[render] poses={poses_path}\n[render] out={output_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_devices)
    is_hot3d = "hot3d" in output_dir

    poses = np.load(poses_path)
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    if radius != 1.0:
        poses[:, :3, 3] *= radius

    # --- Load OBJ+MTL+textures as a Scene to preserve materials/UVs ---
    tm = trimesh.load(cad_path, force='scene', skip_materials=False)

    # --- Build a single Trimesh in world coords for measurements ---
    def _combined_trimesh(scene_or_mesh):
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Prefer new API; fall back to deprecated dump(concatenate=True)
            try:
                geoms_world = list(scene_or_mesh.to_geometry())  # list[Trimesh] in world coords
                return trimesh.util.concatenate(geoms_world)
            except Exception:
                return scene_or_mesh.dump(concatenate=True)
        else:
            return scene_or_mesh  # already a Trimesh

    combined = _combined_trimesh(tm)

    # --- Decide units using bbox diagonal (approx. diameter) ---
    diameter = float(np.linalg.norm(combined.bounding_box.extents))  # ~bbox diagonal length
    # If the object is likely in millimeters, convert scene to meters
    if not is_hot3d and diameter > 100.0:  # heuristic: > 100 means probably mm
        scale = 0.001
        if isinstance(tm, trimesh.Scene):
            for g in tm.geometry.values():
                g.apply_scale(scale)
        else:
            tm.apply_scale(scale)
        # recompute combined *after* scaling
        combined = _combined_trimesh(tm)

    # --- Recenter AFTER scaling so translation is correct ---
    center = combined.bounding_box.centroid
    re_center_transform = np.eye(4)
    re_center_transform[:3, 3] = -center
    print(f"Object center (scaled) at {center}")

    # --- Convert to pyrender.Mesh while keeping textures/materials ---
    if isinstance(tm, trimesh.Scene):
        geoms = list(tm.geometry.values())  # keep TextureVisuals
        mesh = pyrender.Mesh.from_trimesh(geoms, smooth=False)
    else:
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)

    os.makedirs(output_dir, exist_ok=True)
    render(
        output_dir=output_dir,
        mesh=mesh,
        obj_poses=poses,
        intrinsic=intrinsic,
        img_size=img_size,                  # <- use config
        light_itensity=light_int,
        re_center_transform=re_center_transform,
    )
