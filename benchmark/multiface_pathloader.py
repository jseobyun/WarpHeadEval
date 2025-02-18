import os
import argparse

def parse_config():
    parser = argparse.ArgumentParser("Multiface path loader")
    parser.add_argument("--data_root", type=str, default="/path/to/multiface_benchmark")
    parser.add_argument("--save_root", type=str, default="/path/to/multiface_results")
    args = parser.parse_args()
    return args

class MultiFaceBenchmark():
    def __init__(self, data_root, save_root):
        self.data_root = data_root
        self.save_root = save_root
        self.img_root = os.path.join(data_root, "images")
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
        self.gahter_paths()

    def gahter_paths(self):
        self.img_paths = []
        self.depth_paths = []
        self.calib_paths = []
        self.mask_paths = []
        self.u_paths = []
        self.v_paths = []
        self.kp_paths = []

        self.cam_ids = []
        self.expr_ids = []
        subj_names = sorted(os.listdir(self.img_root))

        for subj_name in subj_names:
            if subj_name in self.subj_ban_list:
                continue
            img_dir = os.path.join(self.img_root, subj_name)
            expr_names = sorted(os.listdir(img_dir))

            for expr_name in expr_names:

                if expr_name in self.expr_ban_list:
                    continue


                img_subdir = os.path.join(img_dir, expr_name)

                img_names = sorted(os.listdir(img_subdir))
                for img_name in img_names:
                    img_id = img_name.split(".")[0]

                    img_path = os.path.join(img_subdir, img_id+".jpg")
                    cam_id = int(img_id.split("_")[0][3:])

                    self.img_paths.append(img_path)
                    self.cam_ids.append(cam_id)
                self.expr_ids.append(expr_name)
        self.cam_ids = list(set(self.cam_ids))
        self.expr_ids = list(set(self.expr_ids))

        print(f"Multiface Benchmark is ready : {len(self.img_paths)} samples")
        print(f"Camera view points : {len(self.cam_ids)}")
        print(f"Expression : ", len(self.expr_ids))


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        save_dir = img_path.replace(self.data_root, self.save_root)
        save_dir = save_dir.replace("/images", "")
        save_dir = save_dir.replace(".jpg", "")
        sample_id = img_path.replace(self.img_root, "")[1:-4]
        return img_path, save_dir, sample_id



    def __len__(self):
        return len(self.img_paths)




if __name__ == "__main__":
    args = parse_config()
    args.data_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/multiface_benchmark"
    args.save_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/multiface_results"
    multiface = MultiFaceBenchmark(data_root=args.data_root, save_root=args.save_root)

    for i, (img_path, save_dir, sample_id) in enumerate(multiface):
        '''        
        make sure that you should create save_dir 
        
        os.makedirs(save_dir, exist_ok=True)
        '''
        cam_ids = sorted(multiface.cam_ids)
        for expr_id  in cam_ids:
            print(expr_id)
        print("")












