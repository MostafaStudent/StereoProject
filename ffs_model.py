import sys, os

# ── Point this to your Fast-FoundationStereo repo folder ────────────────────
FFS_ROOT = r'C:\Users\Mosti\PycharmProjects\Fast-FoundationStereo'
sys.path.insert(0, FFS_ROOT)

IMG_SET  = '188/3d'
IMG_NAME = '20260410_170144'
MODEL_PATH = os.path.join(FFS_ROOT, 'weights', '23-36-37', 'model_best_bp2_serialize.pth')


VALID_ITERS = 20
MAX_DISP    = 192
SCALE       = 1.0
HIERA       = False

REMOVE_INVISIBLE  = False   # ignore non-overlapping region's depth
ZFAR              = 30000
SOR_ENABLED       = False   # statistical outlier removal
SOR_NB_NEIGHBORS  = 30      # number of neighbors to consider
SOR_STD_RATIO     = 1.5     # lower = more aggressive removal

DENOISE_CLOUD     = False   # apply extra radius outlier removal after SOR
DENOISE_VOXEL_SIZE = 1.0   # voxel downsampling size
DENOISE_NB_POINTS = 30      # number of points to consider for radius outlier removal
DENOISE_RADIUS    = 30.0    # radius for outlier removal in mm

X_FILTER_ENABLED  = False
X_FILTER_RANGE    = 1500

Y_FILTER_ENABLED  = False
Y_FILTER_RANGE    = 2500

import numpy as np
import open3d as o3d
import cv2 as cv
import torch
import yaml
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud

os.environ["TORCHDYNAMO_DISABLE"] = "1"


def load_ffs_model(model_path):
    cfg_path = os.path.join(os.path.dirname(model_path), 'cfg.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['model_dir'] = model_path
    cfg['valid_iters'] = VALID_ITERS
    cfg['max_disp'] = MAX_DISP
    args = OmegaConf.create(cfg)

    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()
    print("FFS model loaded.")
    return model, args


def run_ffs(rectL_bgr, rectR_bgr, model, args):
    """Takes rectified BGR images (numpy), returns disparity map (H x W float32)."""
    img0 = rectL_bgr[:, :, ::-1].copy()
    img1 = rectR_bgr[:, :, ::-1].copy()

    img0 = cv.resize(img0, fx=SCALE, fy=SCALE, dsize=None)
    img1 = cv.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    H, W = img0.shape[:2]

    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        if not HIERA:
            disp = model.forward(t0, t1, iters=args.valid_iters, test_mode=True,
                                 optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(t0, t1, iters=args.valid_iters, test_mode=True,
                                         small_ratio=0.5)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    return disp


def reconstruct_point_cloud_ffs(disp, rectL_bgr, camMatrix, baseline):
    """Back-projects disparity to a point cloud. All spatial values are in mm."""
    if REMOVE_INVISIBLE:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        disp[xx - disp < 0] = np.inf

    K  = camMatrix.astype(np.float32).copy()
    K[:2] *= SCALE
    fx = K[0, 0]; cx = K[0, 2]; cy = K[1, 2]

    H, W = disp.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    with np.errstate(divide='ignore', invalid='ignore'):
        z_mm = np.where(disp > 0, fx * baseline / disp, 0).astype(np.float32)

    pts_mm  = np.stack([(uu - cx) * z_mm / fx, (vv - cy) * z_mm / fx, z_mm], axis=-1).reshape(-1, 3)
    img_rgb = rectL_bgr[:, :, ::-1].reshape(-1, 3)

    keep = (pts_mm[:, 2] > 0) & (pts_mm[:, 2] <= ZFAR)
    if X_FILTER_ENABLED:
        keep &= (pts_mm[:, 0] >= -X_FILTER_RANGE) & (pts_mm[:, 0] <= X_FILTER_RANGE)
    if Y_FILTER_ENABLED:
        keep &= (pts_mm[:, 1] >= -Y_FILTER_RANGE) & (pts_mm[:, 1] <= Y_FILTER_RANGE)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm[keep])
    pcd.colors = o3d.utility.Vector3dVector(img_rgb[keep].astype(np.float64) / 255.0)

    print(f"  Z  <= {ZFAR} mm")
    if X_FILTER_ENABLED:
        print(f"  X  in [{-X_FILTER_RANGE}, +{X_FILTER_RANGE}] mm")
    if Y_FILTER_ENABLED:
        print(f"  Y  in [{-Y_FILTER_RANGE}, +{Y_FILTER_RANGE}] mm")
    print(f"  {len(pts_mm[keep])} points after filtering")

    if SOR_ENABLED:
        print("Running Statistical Outlier Removal...")
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
        pcd = pcd.select_by_index(ind)
        print(f"SOR done. {len(np.asarray(pcd.points))} points remaining.")

    if DENOISE_CLOUD:
        print("Denoising point cloud...")
        pcd = pcd.voxel_down_sample(voxel_size=DENOISE_VOXEL_SIZE)
        _, ind = pcd.remove_radius_outlier(nb_points=DENOISE_NB_POINTS, radius=DENOISE_RADIUS)
        pcd = pcd.select_by_index(ind)
        print(f"Denoising done. {len(np.asarray(pcd.points))} points remaining.")

    output_path = os.path.join(os.getcwd(), 'point_cloud_ffs.ply')
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud to: {output_path}")

    torch.cuda.empty_cache()
    o3d.visualization.draw_geometries([pcd], window_name='FFS Point Cloud', width=1456, height=816)


def main():
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    try:
        calib = np.load('stereo_calibration.npz')
    except FileNotFoundError:
        print("Error: stereo_calibration.npz not found. Run calibration first.")
        return

    imgLeft  = cv.imread(f'images/{IMG_SET}/cam0/cam0_{IMG_NAME}.jpg')
    imgRight = cv.imread(f'images/{IMG_SET}/cam1/cam1_{IMG_NAME}.jpg')
    if imgLeft is None or imgRight is None:
        print("Error: Check your image paths.")
        return

    size = imgLeft.shape[:2][::-1]
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        calib['camMatrix0'], calib['distCoeff0'],
        calib['camMatrix1'], calib['distCoeff1'],
        size, calib['R'], calib['T'], alpha=0)
    mapLX, mapLY = cv.initUndistortRectifyMap(calib['camMatrix0'], calib['distCoeff0'], R1, P1, size, cv.CV_32FC1)
    mapRX, mapRY = cv.initUndistortRectifyMap(calib['camMatrix1'], calib['distCoeff1'], R2, P2, size, cv.CV_32FC1)
    rectL = cv.remap(imgLeft,  mapLX, mapLY, cv.INTER_LINEAR)
    rectR = cv.remap(imgRight, mapRX, mapRY, cv.INTER_LINEAR)

    model, args = load_ffs_model(MODEL_PATH)
    disp = run_ffs(rectL, rectR, model, args)

    vis = vis_disparity(disp, color_map=cv.COLORMAP_TURBO)
    cv.imshow('FFS Disparity', vis[:, :, ::-1])
    cv.waitKey(0)
    cv.destroyAllWindows()

    reconstruct_point_cloud_ffs(disp, rectL, calib['camMatrix0'], abs(float(calib['T'][0][0])))

if __name__ == '__main__':
    main()
