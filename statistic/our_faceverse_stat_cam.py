import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_name(name, selected):

    valid = False
    for s in selected:
        if s in name:
            valid = True
            break
    return valid

if __name__ == "__main__":

    benchmark_name = "faceverse"
    root = "/media/jseob/db974b7f-3790-49e0-acf4-e8511d26cde9/evals/results"
    exp_names = sorted(os.listdir(root))
    exp_names = [exp_name for exp_name in exp_names if exp_name.startswith(benchmark_name)]

    exp_names = [
        "faceverse_ours_before",
        "faceverse_nodino",
        "faceverse_sapiens",
    ]

    selects = ["f1200"]
    for exp_name in exp_names:
        print(exp_name)
        exp_dir = os.path.join(root, exp_name)
        subj_names = sorted(os.listdir(exp_dir))

        mpuve = 0
        mp3de = 0
        count = 0
        mean_aucuv = 0
        mean_auc3d = 0
        area =0

        for subj_name in tqdm(subj_names):
            subj_dir = os.path.join(exp_dir, subj_name)
            result_names = sorted(os.listdir(subj_dir))
            for result_name in result_names:


                if not check_name(result_name, selects):
                    continue

                with open(os.path.join(subj_dir, result_name), 'r') as json_file:
                    result = json.load(json_file)
                    if result["valid_region_area"] == 0:
                        continue
                    mp3de += result["mean_all_distance"] * result["valid_region_area"]
                    mpuve += result["mean_all_uv_errors"] * result["valid_region_area"]
                    aucuv = np.asarray(result["all_uvcount"]) / result["valid_region_area"] * 1/1000
                    aucuv = np.sum(aucuv)
                    mean_aucuv += aucuv
                    auc3d = np.asarray(result["all_count"]) / result["valid_region_area"] * 1 / 1000
                    auc3d = np.sum(auc3d)
                    mean_auc3d += auc3d
                    count += 1
                    area += result["valid_region_area"]

        mpuve = mpuve / area
        mp3de = mp3de / area
        mean_aucuv = mean_aucuv / count
        mean_auc3d = mean_auc3d / count


        print("mAUCuv", mean_aucuv)
        print("mAUC3d", mean_auc3d)
        print("mpuve", mpuve)
        print("mp3de", mp3de)







