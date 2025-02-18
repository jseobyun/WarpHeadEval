import os
import cv2
import numpy as np
from utils.data_utils import load_img, load_depth_gt, load_uv, load_mask, load_calib, load_kps

class MultiFaceBenchmark():
    def __init__(self, root="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/multiface_benchmark", downsample=1):
        self.root = root
        self.img_root = os.path.join(root, "images")
        self.depth_root = os.path.join(root, "depths")
        self.calib_root = os.path.join(root, "calibs")
        self.mask_root = os.path.join(root, "masks")
        self.uv_root = os.path.join(root, "uvs")
        self.kp_root = os.path.join(root, "kps")
        self.subj_ban_list = [
            "m--20190529--1004--5067077--GHS",
            "m--20190529--1300--002421669--GHS",
            "m--20190828--1318--002645310--GHS",
        ]
        self.expr_ban_list = [
            "E002_Swallow",
            "E062_Nostrils_Dilated",
            "E063_Nostrils_Sucked_In",
            "E074_Blink",
            "E047_Tongue_Over_Upper_Lip",
            "E048_Tongue_Out_Lips_Closed",
            "E049_Mouth_Open_Tongue_Out",
            "E050_Bite_Tongue",
            "E051_Tongue_Out_Flat",
            "E052_Tongue_Out_Thick",
            "E053_Tongue_Out_Rolled",
            "E054_Tongue_Out_Right_Teeth_Showing",
            "E055_Tongue_Out_Left_Teeth_Showing",
        ]
        self.gahter_paths(downsample)

    def check_path(self, paths):
        valid = True
        for path in paths:
            if not os.path.exists(path):
                valid = False
        return valid


    def gahter_paths(self, downsample=1):
        self.img_paths = []
        self.depth_paths = []
        self.calib_paths = []
        self.mask_paths = []
        self.u_paths = []
        self.v_paths = []
        self.kp_paths = []

        self.cam_ids = []
        subj_names = sorted(os.listdir(self.img_root))

        for subj_name in subj_names:
            img_dir = os.path.join(self.img_root, subj_name)
            expr_names = sorted(os.listdir(img_dir))

            for expr_name in expr_names:

                img_subdir = os.path.join(img_dir, expr_name)

                img_names = sorted(os.listdir(img_subdir))
                for img_name in img_names:
                    img_id = img_name.split(".")[0]

                    img_path = os.path.join(img_subdir, img_id+".jpg")
                    depth_path = os.path.join(img_subdir.replace(self.img_root, self.depth_root), img_id+".png")
                    calib_path = os.path.join(img_subdir.replace(self.img_root, self.calib_root), img_id+".json")
                    mask_path = os.path.join(img_subdir.replace(self.img_root, self.mask_root), img_id + ".jpg")
                    u_path = os.path.join(img_subdir.replace(self.img_root, self.uv_root), img_id + "_u.png")
                    v_path = os.path.join(img_subdir.replace(self.img_root, self.uv_root), img_id + "_v.png")
                    kp_path = os.path.join(img_subdir.replace(self.img_root, self.kp_root), img_id + ".json")

                    if self.check_path([img_path, depth_path, calib_path, mask_path, u_path, v_path, kp_path]):

                        cam_id = int(img_id.split("_")[0][3:])

                        self.img_paths.append(img_path)
                        self.depth_paths.append(depth_path)
                        self.mask_paths.append(mask_path)
                        self.calib_paths.append(calib_path)
                        self.u_paths.append(u_path)
                        self.v_paths.append(v_path)
                        self.kp_paths.append(kp_path)
                        self.cam_ids.append(cam_id)
        self.cam_ids = list(set(self.cam_ids))

        if downsample !=1:
            self.img_paths = self.img_paths[::downsample]
            self.depth_paths = self.depth_paths[::downsample]
            self.calib_paths = self.calib_paths[::downsample]
            self.mask_paths = self.mask_paths[::downsample]
            self.u_paths = self.u_paths[::downsample]
            self.v_paths = self.v_paths[::downsample]
            self.kp_paths = self.kp_paths[::downsample]

        print(f"Multiface Benchmark with {downsample} downsample is ready : {len(self.img_paths)} samples")
        print(f"Camera view points : {len(self.cam_ids)}")


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


    multiface = MultiFaceBenchmark(downsample=1)
    for i, (uv, depth, mask, img, calib, kps, sample_id) in enumerate(multiface):
        print(i)
        continue














