import os
import cv2
import numpy as np
from utils.data_utils import load_img, load_depth_gt, load_uv, load_mask, load_calib, load_kps

class FaceVerseBenchmark():
    def __init__(self, root="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_benchmark", split=-1):
        self.root = root
        self.img_root = os.path.join(root, "images")
        self.depth_root = os.path.join(root, "depths")
        self.calib_root = os.path.join(root, "calibs")
        self.mask_root = os.path.join(root, "masks")
        self.uv_root = os.path.join(root, "uvs")
        self.kp_root = os.path.join(root, "kps")

        self.gahter_paths(split)

    def check_path(self, paths):
        valid = True
        for path in paths:
            if not os.path.exists(path):
                valid = False
        return valid


    def gahter_paths(self, split=-1):
        self.img_paths = []
        self.depth_paths = []
        self.calib_paths = []
        self.mask_paths = []
        self.u_paths = []
        self.v_paths = []
        self.kp_paths = []

        self.aug_ids = []
        subj_names = sorted(os.listdir(self.img_root))

        for subj_name in subj_names: ################################

            ### newly added
            # expr_id = subj_name.split("_")[-1]
            # if int(expr_id) not in [3, 5, 9, 11, 12, 14]:
            #     continue
            ###
            img_dir = os.path.join(self.img_root, subj_name)
            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                img_id = img_name.split(".")[0]

                ### newly added
                # if "elev-45" not in img_id and "elev45" not in img_id:
                #     continue
                ###

                img_path = os.path.join(img_dir, img_id+".jpg")
                depth_path = os.path.join(img_dir.replace(self.img_root, self.depth_root), img_id+".png")
                calib_path = os.path.join(img_dir.replace(self.img_root, self.calib_root), img_id+".json")
                mask_path = os.path.join(img_dir.replace(self.img_root, self.mask_root), img_id + ".jpg")
                u_path = os.path.join(img_dir.replace(self.img_root, self.uv_root), img_id + "_u.png")
                v_path = os.path.join(img_dir.replace(self.img_root, self.uv_root), img_id + "_v.png")
                kp_path = os.path.join(img_dir.replace(self.img_root, self.kp_root), img_id + ".json")

                # if self.check_path([img_path, depth_path, calib_path, mask_path, u_path, v_path, kp_path]):

                aug_id = img_id.split(".")[0]

                self.img_paths.append(img_path)
                self.depth_paths.append(depth_path)
                self.mask_paths.append(mask_path)
                self.calib_paths.append(calib_path)
                self.u_paths.append(u_path)
                self.v_paths.append(v_path)
                self.kp_paths.append(kp_path)
                self.aug_ids.append(aug_id)
        self.aug_ids = list(set(self.aug_ids))

        if split != -1:
            num_samples = len(self.img_paths)
            step = int(num_samples//4)+1
            start = step*split
            end = step*(split+1)
            end = min(end, num_samples)

            self.img_paths = self.img_paths[start:end]
            self.depth_paths = self.depth_paths[start:end]
            self.calib_paths = self.calib_paths[start:end]
            self.mask_paths = self.mask_paths[start:end]
            self.u_paths = self.u_paths[start:end]
            self.v_paths = self.v_paths[start:end]
            self.kp_paths = self.kp_paths[start:end]

        print(f"Faceverse Benchmark with {split} downsample is ready : {len(self.img_paths)} samples")
        print(f"Focals and distances, View points : {len(self.aug_ids)}")


    def __call__(self, index):
        img_path = self.img_paths[index]
        depth_path = self.depth_paths[index]
        calib_path = self.calib_paths[index]
        mask_path = self.mask_paths[index]
        u_path = self.u_paths[index]
        v_path = self.v_paths[index]
        kp_path = self.kp_paths[index]

        face_mask_path = mask_path.replace("masks", "face_masks")
        nonface_mask_path = mask_path.replace("masks", "nonface_masks")

        img = load_img(img_path)
        depth = load_depth_gt(depth_path)
        mask = load_mask(mask_path)
        face_mask = load_mask(face_mask_path)
        nonface_mask = load_mask(nonface_mask_path)
        calib = load_calib(calib_path)
        uv = load_uv(u_path, v_path)
        kps = load_kps(kp_path)
        sample_id = img_path.replace(self.img_root, "")[1:-4]

        return uv, depth, mask, face_mask, nonface_mask, img, calib, kps, sample_id



    def __len__(self):
        return len(self.img_paths)



if __name__ == "__main__":

    faceverse = FaceVerseBenchmark(downsample=1)

    for i, (img, depth, mask, uv, calib, sample_id) in enumerate(faceverse):
        print(sample_id)
        break
        print("")












