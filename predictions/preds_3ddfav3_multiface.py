import os
import sys
curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(curr_dir, ".."))
import trimesh
import numpy as np
import open3d as o3d

def trimesh2open3d(mesh_path, triangles_uvs, texture):
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    faces = np.asarray(o3d_mesh.triangles)

    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangles_uvs)
    o3d_mesh.textures = [o3d.geometry.Image(texture)]
    o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0]*len(faces))

    return o3d_mesh

class Preds3DDFAv3Multiface():
    def __init__(self, root):
        self.root = root

        verts53215_uvs = np.load(os.path.join(curr_dir, "../assets/bfm_verts_uvs53215.npy"))
        map53215to35709 = np.load(os.path.join(curr_dir, "../assets/bfm_vidx53215to35709.npy"))
        self.verts35709_uvs = verts53215_uvs[map53215to35709]


        ###
        # mesh = o3d.io.read_triangle_mesh("/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/evals/3DDFA_V2/multiface_results/m--20171024--0000--002757580--GHS/E002_Swallow/cam400002_frame000431/cam400002_frame000431.ply")
        # verts = np.asarray(mesh.vertices)
        # faces = np.asarray(mesh.triangles)
        # ###
        # self.triangles_uvs = []
        # for face in faces:
        #     uvs = self.verts38365_uvs[face].reshape(1, 3, 2)
        #     self.triangles_uvs.append(uvs)
        # self.triangles_uvs = np.concatenate(self.triangles_uvs, axis=0)
        # np.save("../assets/bfm38365_uvs_per_face.npy", self.triangles_uvs)

        self.triangles_uvs = np.load(os.path.join(curr_dir, "../assets/bfm35709_uvs_per_face_of_flame.npy"))
        self.triangles_uvs = self.triangles_uvs.reshape(-1, 2)

        self.triangles_uvs[:, 1] = 1 - self.triangles_uvs[:, 1]
        tex_dim = 8192
        self.texture = np.ones([tex_dim, tex_dim, 3], dtype=np.float32)
        r, g = np.meshgrid(np.arange(0, tex_dim), np.arange(0, tex_dim), indexing="xy")
        r = r / tex_dim
        g = g / tex_dim
        self.texture[:,:,0] = r
        self.texture[:,:,1] = g

        self.kp_idx = np.load(os.path.join(curr_dir, "../assets/bfm_kidx35709.npy")).reshape(-1)
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



        print(f"Multifiace predictions of 3DDFAV3 is ready : {len(pred_dirs)}")
        self.pred_dirs = pred_dirs


    def search_index(self, sample_id):
        sample_id = sample_id

        pred_id = None
        index = None
        for pidx, pred_dir in enumerate(self.pred_dirs):
            if sample_id in pred_dir:
                index=  pidx
                pred_id = pred_dir.split("/")[-3:]
                pred_id = "/".join(pred_id)
                break
        if pred_id is None or index is None:
            return None, None,
        return pred_id, index

    def __call__(self, sample_id):

        pred_id, index = self.search_index(sample_id)

        if pred_id is None or index is None:
            return None, None, None

        subj_id = pred_id.split("/")[-1]
        mesh_path = os.path.join(self.pred_dirs[index], f"{subj_id}_extractTex.obj")
        if not os.path.exists(mesh_path):
            return None, None, None
        o3d_mesh = trimesh2open3d(mesh_path, self.triangles_uvs, self.texture)
        verts = np.asarray(o3d_mesh.vertices)
        kps = verts[self.kp_idx]
        del self.pred_dirs[index]

        return o3d_mesh, kps, pred_id


