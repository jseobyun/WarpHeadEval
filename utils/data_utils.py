import cv2
import json
import trimesh
import numpy as np
import open3d as o3d
from PIL import Image


def load_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.faces).astype(np.int32)
    return verts, faces

def load_kps(kp_path):
    with open(kp_path, 'r') as f:
        kps = json.load(f)
    kps = np.asarray(kps["kp3ds"]).reshape(-1, 3)
    return kps

def load_img(img_path):
    img = Image.open(img_path)
    img = np.asarray(img).astype(np.float32)
    img = img / 255.0
    return img

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)/255.0
    return mask

def load_depth_gt(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 65535 * 10
    return depth

def load_depth_pred(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 65535
    depth = depth*2 - 1
    return depth

def load_uv(u_path, v_path):
    u = cv2.imread(u_path, cv2.IMREAD_UNCHANGED)
    v = cv2.imread(v_path, cv2.IMREAD_UNCHANGED)

    u = u.astype(np.float32) / 65535
    v = v.astype(np.float32) / 65535

    uv = np.concatenate([u[:,:,None], v[:,:,None]], axis=2)
    return uv

def load_calib(calib_path):
    with open(calib_path, 'r') as json_file:
        calib = json.load(json_file)
    return calib

def backproject_depth(K, depth_rn, T_gk=None):
    img_h, img_w = np.shape(depth_rn)[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h), indexing="xy")
    u = (u - cx) / fx
    v = (v - cy) / fy

    xy1 = np.concatenate([u[:,:,None], v[:,:,None], np.ones_like(u[:,:,None])], axis=2) # HW3
    d = depth_rn.reshape(img_h, img_w, 1)

    xyz = xy1 * d


    xyz = xyz.reshape(-1, 3)
    if T_gk is not None:
        xyz = np.matmul(T_gk[:3, :3], xyz.transpose(1,0)).transpose(1,0) + T_gk[:3,-1].reshape(-1 ,3)

    xyz = xyz.reshape(img_h, img_w, 3).astype(np.float32)

    return xyz


def umeyama_alignment(source, target):
    """
    Umeyama 알고리즘을 사용하여 3D 점 집합 간의 변환 행렬 (R, t, s)을 계산
    source: [N,3] numpy array (변환 전 점들)
    target: [N,3] numpy array (변환 후 점들)
    """
    assert source.shape == target.shape, "두 점 집합의 차원이 일치해야 합니다."

    # 점 개수
    N = source.shape[0]

    # 중심(mean) 계산
    mean_source = np.mean(source, axis=0)
    mean_target = np.mean(target, axis=0)

    # 중심을 원점으로 이동 (Centered data)
    source_centered = source - mean_source
    target_centered = target - mean_target

    # 공분산 행렬 계산
    H = source_centered.T @ target_centered / N

    # SVD 수행
    U, S, Vt = np.linalg.svd(H)

    # 회전 행렬 계산 (반사 문제 방지)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # 만약 R의 행렬식이 음수라면, 반사를 방지하도록 조정
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale 계산
    var_source = np.var(source_centered, axis=0).sum()
    scale = np.sum(S) / var_source

    # Translation 계산
    t = mean_target - scale * (R @ mean_source)

    return R, t, scale

class PinholeRenderer():
    def __init__(self, img_h, img_w, visible=False):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=visible, width=img_w, height=img_h)
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([0, 0, 0])
        self.opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        # self.opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal

        self.ctr = self.vis.get_view_control()
        self.ctr.set_constant_z_near(0.001)
        self.ctr.set_constant_z_far(10.)
        self.w = img_w
        self.h = img_h
        self.curr_geometry = None


    def update_geometry(self, geom):
        if self.curr_geometry is not None:
            self.vis.remove_geometry(self.curr_geometry)
            self.vis.add_geometry(geom)
            self.curr_geometry = geom
        else:
            self.vis.add_geometry(geom)
            self.curr_geometry = geom
        self.vis.poll_events()
        self.vis.update_renderer()

    def render(self, K, T_gk):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.w, self.h,
                                                      K[0,0], K[1,1],
                                                      K[0,2], K[1,2])

        new_cam_param = o3d.camera.PinholeCameraParameters()
        new_cam_param.intrinsic = intrinsic
        new_cam_param.extrinsic = np.linalg.inv(T_gk)

        self.ctr.convert_from_pinhole_camera_parameters(new_cam_param, allow_arbitrary=True)
        self.vis.poll_events()
        self.vis.update_renderer()
        depth_rn = np.asarray(self.vis.capture_depth_float_buffer(True))
        depth_rn = depth_rn.astype(np.float32)

        rgb_rn = np.asarray(self.vis.capture_screen_float_buffer(True))
        rgb_rn = rgb_rn.astype(np.float32)

        M = np.float64([[1, 0, 0], [0, 1, 0]])

        M[0, 2] = K[0,2] - (self.w - 1) / 2
        M[1, 2] = K[1,2] - (self.h - 1) / 2

        img_w, img_h = self.w, self.h
        depth_rn = cv2.warpAffine(depth_rn, M, (img_w, img_h), flags=cv2.INTER_NEAREST)
        rgb_rn = cv2.warpAffine(rgb_rn, M, (img_w, img_h), flags=cv2.INTER_NEAREST)
        return depth_rn, rgb_rn

    def close(self):
        self.vis.clear_geometries()
        self.vis.destroy_window()