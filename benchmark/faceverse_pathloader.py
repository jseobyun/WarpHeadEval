import os
import argparse

def parse_config():
    parser = argparse.ArgumentParser("Faceverse path loader")
    parser.add_argument("--data_root", type=str, default="/path/to/faceverse_benchmark")
    parser.add_argument("--save_root", type=str, default="/path/to/faceverse_results")
    args = parser.parse_args()
    return args

class FaceVerseBenchmark():
    def __init__(self, data_root, save_root):
        self.data_root = data_root
        self.save_root = save_root
        self.img_root = os.path.join(data_root, "images")
        self.gahter_paths()

    def gahter_paths(self):
        self.img_paths = []
        self.aug_ids = []
        subj_names = sorted(os.listdir(self.img_root))

        for subj_name in subj_names:
            ### newly added
            expr_id = subj_name.split("_")[-1]
            if int(expr_id) in [3, 5,9, 11, 12, 14]:
                continue
            ###

            img_dir = os.path.join(self.img_root, subj_name)
            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                img_id = img_name.split(".")[0]
                ### newly added
                if "elev-45" in img_id or "elev45" in img_id:
                    continue
                ###

                img_path = os.path.join(img_dir, img_id+".jpg")
                aug_id = img_id.split(".")[0]
                self.img_paths.append(img_path)
                self.aug_ids.append(aug_id)
        self.aug_ids = list(set(self.aug_ids))
        print(f"Faceverse Benchmark is ready : {len(self.img_paths)} samples")
        print(f"Focals and distances, View points : {len(self.aug_ids)}")


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
    args.data_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_benchmark"
    args.save_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_results"
    faceverse = FaceVerseBenchmark(data_root=args.data_root, save_root=args.save_root)

    for i, (img_path, save_dir, sample_id) in enumerate(faceverse):
        '''        
        make sure that you should create save_dir 

        os.makedirs(save_dir, exist_ok=True)
        '''
        print("")













