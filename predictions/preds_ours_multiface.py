import os
import numpy as np
import shutil
from utils.data_utils import load_depth_pred, load_uv, load_mask
class PredsOursMultiface():
    def __init__(self, root="/media/jseob/db974b7f-3790-49e0-acf4-e8511d26cde9/evals/predictions/multiface_nodino"):
        self.root = root
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

        print(f"Multifiace predictions of Ours is ready : {len(pred_dirs)}")
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


if __name__ == "__main__":
    preds = PredsOursMultiface()
