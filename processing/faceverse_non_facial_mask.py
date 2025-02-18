import os
import cv2
import argparse
import numpy as np
from benchmark.faceverse_pathloader import FaceVerseBenchmark

def parse_config():
    parser = argparse.ArgumentParser("Faceverse path loader")
    parser.add_argument("--data_root", type=str, default="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_benchmark")
    parser.add_argument("--save_root", type=str, default="/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_benchmark")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_config()
    faceverse = FaceVerseBenchmark(args.data_root, args.save_root)

    for i, (img_path, _, sample_id) in enumerate(faceverse):
        print(i, len(faceverse))
        mask_path = img_path.replace("images", "masks")
        face_mask_path = img_path.replace("images", "face_masks")
        save_path = img_path.replace("images", "nonface_masks")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)

        mask_region = mask > 250
        face_region = face_mask > 250
        face_region = mask_region * face_region
        nonface_mask = mask.copy()
        nonface_mask[face_region] = 0
        face_mask[~face_region] = 0
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(face_mask_path, face_mask)
        cv2.imwrite(save_path, nonface_mask)
        # cv2.imshow("mask", mask)
        # cv2.imshow("face", face_mask)
        # cv2.imshow("nonface", nonface_mask)
        # cv2.waitKey(0)