import os
import sys
curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(curr_dir, ".."))
import trimesh
import numpy as np
import open3d as o3d
from utils.data_utils import load_mesh, load_depth_pred, load_uv, load_mask

def trimesh2open3d(mesh_path, triangles_uvs, texture):
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces

    # triangles_uvs = []
    # for i in range(3):
    #     triangles_uvs.append(uvs[faces[:, i]].reshape(-1, 1, 2))
    # triangles_uvs = np.concatenate(triangles_uvs, axis=1).reshape(-1, 2)

    vmin = np.min(vertices, axis=0)
    vmax = np.max(vertices, axis=0)

    bbox_len = np.min(vmax-vmin)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
    o3d_mesh.textures = [o3d.geometry.Image(texture)]
    o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0]*len(faces))

    return o3d_mesh

class PredsEMOCAMultiface():
    def __init__(self, root="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/evals/EMOCA/multiface_results"):
        self.root = root

        self.triangles_uvs = np.load(os.path.join(curr_dir, "../assets/flame59315_uvs_per_face.npy"))
        self.triangles_uvs = self.triangles_uvs.reshape(-1, 2)

        self.triangles_uvs[:, 1] = 1 - self.triangles_uvs[:, 1]
        tex_dim = 8192
        self.texture = np.ones([tex_dim, tex_dim, 3], dtype=np.float32)
        r, g = np.meshgrid(np.arange(0, tex_dim), np.arange(0, tex_dim), indexing="xy")
        r = r / tex_dim
        g = g / tex_dim
        self.texture[:,:,0] = r
        self.texture[:,:,1] = g

        self.kp_idx = np.load(os.path.join(curr_dir, "../assets/flame59315_kp_idx.npy")).reshape(-1)
        self.kp_idx = self.kp_idx.astype(np.int32)
        # import cv2
        # texture = cv2.resize(self.texture, dsize=(512, 512))
        # texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
        # cv2.imshow("vis", texture)
        # cv2.waitKey(0)

        self.gather_paths()

    def gather_paths(self):
        subj_names = sorted(os.listdir(self.root))

        pred_dirs = []
        for subj_name in subj_names:
            pred_root = os.path.join(self.root, subj_name)
            expr_names = sorted(os.listdir(pred_root))

            for expr_name in expr_names:
                pred_subroot = os.path.join(pred_root, expr_name)

                pred_dirnames = sorted(os.listdir(pred_subroot))
                for pred_dirname in pred_dirnames:
                    pred_dirs.append(os.path.join(pred_subroot, pred_dirname))



        print(f"Multifiace predictions of EMOCA is ready : {len(pred_dirs)}")
        self.pred_dirs = pred_dirs


    def __getitem__(self, index):

        mesh_path = os.path.join(self.pred_dirs[index], "mesh_coarse_detail.obj")
        o3d_mesh = trimesh2open3d(mesh_path, self.triangles_uvs, self.texture)

        return o3d_mesh

    def search_index(self, sample_id):
        sample_id = sample_id+"00"

        pred_id = None
        index = None
        for pidx, pred_dir in enumerate(self.pred_dirs):
            if sample_id in pred_dir:
                index=  pidx
                pred_id = pred_dir.split("/")[7:]
                pred_id = "/".join(pred_id)
                break
        if pred_id is None or index is None:
            return None, None
        return pred_id[:-2], index

    def __call__(self, sample_id):

        pred_id, index = self.search_index(sample_id)

        if pred_id is None or index is None:
            return None, None, None
        mesh_path = os.path.join(self.pred_dirs[index], "mesh_coarse_detail.obj")
        o3d_mesh = trimesh2open3d(mesh_path, self.triangles_uvs, self.texture)
        verts = np.asarray(o3d_mesh.vertices)
        kps = verts[self.kp_idx]
        del self.pred_dirs[index]


        return o3d_mesh, kps, pred_id


if __name__ == "__main__":
    preds = PredsEMOCAMultiface()

    for i, (data) in enumerate(preds):
        print("")