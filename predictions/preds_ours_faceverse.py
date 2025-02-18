import os
import numpy as np
from utils.data_utils import load_depth_pred, load_uv, load_mask
class PredsOursFaceverse():
    def __init__(self, root="/media/jseob/db974b7f-3790-49e0-acf4-e8511d26cde9/evals/predictions/faceverse"):
        self.root = root
        self.gather_paths()

    def gather_paths(self):

        subj_names = sorted(os.listdir(self.root))


        pred_dirs = []
        for subj_name in subj_names:
            pred_root = os.path.join(self.root, subj_name)
            pred_dirnames = sorted(os.listdir(pred_root))
            for pred_dirname in pred_dirnames:
                pred_dirs.append(os.path.join(pred_root, pred_dirname))

        print(f"Faceverse predictions of Ours is ready : {len(pred_dirs)}")
        self.pred_dirs = pred_dirs


    def __call__(self, index):

        u_path = os.path.join(self.pred_dirs[index], "u.png")
        v_path = os.path.join(self.pred_dirs[index], "v.png")
        d_path = os.path.join(self.pred_dirs[index], "d.png")
        c_path = os.path.join(self.pred_dirs[index], "c.jpg")
        uv = load_uv(u_path, v_path)
        d = load_depth_pred(d_path)
        c = load_mask(c_path)
        return uv, d, c

    def __len__(self):
        return len(self.pred_dirs)