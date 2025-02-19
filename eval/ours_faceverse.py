import os
import sys
curr_dir = os.path.dirname(__file__)
root_dir = os.path.join(curr_dir, "..")
sys.path.append(root_dir)
import cv2
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.vis_utils import make_pcd
from utils.data_utils import backproject_depth
from benchmark.faceverse import FaceVerseBenchmark
from predictions.preds_ours_faceverse import PredsOursFaceverse
from pytorch3d.ops import knn_points, knn_gather

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="faceverse_nodino")
    parser.add_argument("--save_root", default="/media/jseob/db974b7f-3790-49e0-acf4-e8511d26cde9/evals/results")
    parser.add_argument("--vis", default=False)
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--save", default=True)

    args = parser.parse_args()
    return args

def measure_distance(points1, points2):
    points1 = points1.reshape(-1, 3)
    points2 = points2.reshape(-1, 3)

    distances = np.sqrt(np.sum((points1 - points2)**2, axis=-1))
    return distances

def measure_pixel(uv_pred, uv_gt):
    uv_pred = uv_pred.reshape(-1, 2)
    uv_gt = uv_gt.reshape(-1, 2)

    uv_error = np.sqrt(np.sum((uv_pred-uv_gt)**2, axis=-1))
    return uv_error


def chamfer_distance(points1, points2):

    points1_t = torch.from_numpy(points1.reshape(-1, 3)).unsqueeze(dim=0).cuda()
    points2_t = torch.from_numpy(points2.reshape(-1, 3)).unsqueeze(dim=0).cuda()
    knn = knn_points(points1_t, points2_t)
    close_points = knn_gather(points2_t, knn.idx)[:, :, 0]

    distance = torch.sqrt(torch.sum((points1_t - close_points)**2, dim=-1))
    distance = torch.mean(distance)

    return distance.item()

if __name__ == "__main__":

    args = parse_config()
    faceverse = FaceVerseBenchmark()
    ours = PredsOursFaceverse(root=os.path.join("/media/jseob/db974b7f-3790-49e0-acf4-e8511d26cde9/evals/predictions", args.exp_name))

    img_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/faceverse_benchmark/images"
    num_predictions = len(ours.pred_dirs)
    img_h, img_w = 448, 448
    for index in tqdm(range(num_predictions)):
        index = index
        start_time = time.time()

        uv_gt, d_gt, c_gt, face_mask, nonface_mask, img, calib, kps, sample_id = faceverse(index)

        uv_pred, d_pred, c_pred = ours(index)

        assert len(ours) == len(faceverse)

        pred_dir = ours.pred_dirs[index]
        pred_id = pred_dir.replace(ours.root, "")[1:]

        assert pred_id == sample_id
        K = np.asarray(calib["K"]).reshape(3,3)
        T_gk = np.asarray(calib["T_gk"]).reshape(4, 4)
        T_kg = np.linalg.inv(T_gk)

        nose_world = kps[30].reshape(3,1)
        reye0, reye1 = kps[36], kps[39]
        leye0, leye1 = kps[42], kps[45]
        rdist = np.sqrt(np.sum((reye0-reye1)**2))
        ldist = np.sqrt(np.sum((leye0-leye1)**2))
        eye_dist = (rdist + ldist) /2

        eye_len = 0.035
        scale_factor = eye_len/eye_dist


        nose_cam = np.matmul(T_kg[:3, :3], nose_world) + T_kg[:3, -1].reshape(3,1)
        nose_z = nose_cam.reshape(-1)[-1]

        uvc_pred = np.concatenate([uv_pred, c_pred[:, :, None]], axis=-1)  # 448 448 3
        uvc_gt = np.concatenate([uv_gt, c_gt[:, :, None]], axis=-1)  # 512 512 3
        f = np.sqrt(K[0, 0] * K[1, 1])
        focal_scale = 224 / f
        d_pred = (d_pred * focal_scale + 1) * nose_z

        '''
        depth_eval
        '''
        mask = (c_gt !=0) * (d_gt !=0)
        face_mask = (face_mask !=0) * (d_gt !=0)
        nonface_mask = (nonface_mask !=0) * (d_gt !=0)

        valid_region_area = np.sum(mask)
        face_region_area = np.sum(face_mask)
        nonface_region_area = np.sum(nonface_mask)

        points_gt = backproject_depth(K, d_gt)
        points_pred = backproject_depth(K, d_pred)

        points_gt *= scale_factor
        points_pred *= scale_factor

        # if args.do_align:
        #     pcd_pred = make_pcd(points_pred[face_mask].reshape(-1, 3))
        #     pcd_gt = make_pcd(points_gt[face_mask].reshape(-1, 3))
        #
        #
        #     ### ICP if you need
        #     reg_p2p = o3d.pipelines.registration.registration_icp(pcd_pred,
        #                                                           pcd_gt,
        #                                                           max_correspondence_distance=0.01,
        #                                                           init=np.eye(4))
        #     T = reg_p2p.transformation
        #     pcd_pred.transform(T)
        #
        #     points_pred = points_pred.reshape(-1, 3)
        #     points_pred = np.matmul(T[:3, :3], points_pred.transpose(1,0)).transpose(1,0) + T[:3, -1].reshape(-1, 3)
        #     points_pred = points_pred.reshape(img_h, img_w, 3).astype(np.float32)

        if args.vis:
            vis_mask = face_mask
            pcd_pred = make_pcd(points_pred[vis_mask])
            pcd_gt = make_pcd(points_gt[vis_mask], img[vis_mask])
            o3d.visualization.draw_geometries([pcd_pred, pcd_gt])



        # pred2gt_distance = chamfer_distance(points_pred, points_gt)
        gt2pred = {}
        face_distances = measure_distance(points_gt[face_mask], points_pred[face_mask])
        nonface_distances = measure_distance(points_gt[nonface_mask], points_pred[nonface_mask])
        all_distances = measure_distance(points_gt[mask], points_pred[mask])


        face_distances = face_distances[face_distances<=1]
        nonface_distances = nonface_distances[nonface_distances <= 1]
        all_distances = all_distances[all_distances <= 1]

        gt2pred["mean_face_distance"] = float(np.mean(face_distances))
        gt2pred["std_face_distances"] = float(np.std(face_distances))
        gt2pred["mean_nonface_distance"] = float(np.mean(nonface_distances))
        gt2pred["std_nonface_distances"] = float(np.std(nonface_distances))
        gt2pred["mean_all_distance"] = float(np.mean(all_distances))
        gt2pred["std_all_distances"] = float(np.std(all_distances))
        gt2pred["face_region_area"] = float(face_region_area)
        gt2pred["nonface_region_area"] = float(nonface_region_area)
        gt2pred["valid_region_area"] = float(valid_region_area)


        face_count = []
        nonface_count = []
        all_count = []
        thresholds = np.linspace(0, 0.05, 1000).astype(np.float32)
        for th in thresholds:
            face_count.append(int(np.sum(face_distances <= th)))
            nonface_count.append(int(np.sum(nonface_distances<= th)))
            all_count.append(int(np.sum(all_distances <= th)))

        gt2pred["face_count"] = face_count
        gt2pred["nonface_count"] = nonface_count
        gt2pred["all_count"] = all_count
        gt2pred["threshold"] = thresholds.tolist()

        save_path = os.path.join(args.save_root, args.exp_name, sample_id+".json")

        if args.vis:
            face_count = np.asarray(face_count)
            nonface_count = np.asarray(nonface_count)
            all_count = np.asarray(all_count)

            face_ratio = face_count / face_region_area
            nonface_ratio = nonface_count / nonface_region_area
            all_ratio = all_count / valid_region_area



            ##
            thresholds_vis = thresholds * 1000
            plt.plot(thresholds_vis.tolist(), face_ratio.tolist(), linestyle="-", color="g", label="face region")
            plt.plot(thresholds_vis.tolist(), nonface_ratio.tolist(), linestyle="-", color="b", label="nonface reigion")
            plt.plot(thresholds_vis.tolist(), all_ratio.tolist(), linestyle="-", color="r", label="all region")
            plt.xlabel("Distance error(mm)")
            plt.ylabel("Accumulated ratio")
            plt.title("3D Distance Error")
            plt.legend()

            plt.show()

        '''
        UV Eval 
        '''
        face_uv_errors = measure_pixel(uv_pred[face_mask], uv_gt[face_mask])
        nonface_uv_errors = measure_pixel(uv_pred[nonface_mask], uv_gt[nonface_mask])
        all_uv_errors = measure_pixel(uv_pred[mask], uv_gt[mask])

        gt2pred["mean_face_uv_errors"] = float(np.mean(face_uv_errors))
        gt2pred["std_face_uv_errors"] = float(np.std(face_uv_errors))
        gt2pred["mean_nonface_uv_errors"] = float(np.mean(nonface_uv_errors))
        gt2pred["std_nonface_uv_errors"] = float(np.std(nonface_uv_errors))
        gt2pred["mean_all_uv_errors"] = float(np.mean(all_uv_errors))
        gt2pred["std_all_uv_errors"] = float(np.std(all_uv_errors))

        face_uvcount = []
        nonface_uvcount = []
        all_uvcount = []
        uv_thresholds = np.linspace(0, 0.1, 1000).astype(np.float32)
        for uv_th in uv_thresholds:
            face_uvcount.append(int(np.sum(face_uv_errors <= uv_th)))
            nonface_uvcount.append(int(np.sum(nonface_uv_errors <= uv_th)))
            all_uvcount.append(int(np.sum(all_uv_errors <= uv_th)))

        gt2pred["face_uvcount"] = face_uvcount
        gt2pred["nonface_uvcount"] = nonface_uvcount
        gt2pred["all_uvcount"] = all_uvcount
        gt2pred["uv_threshold"] = uv_thresholds.tolist()

        if False: #args.vis:
            face_uvcount = np.asarray(face_uvcount)
            nonface_uvcount = np.asarray(nonface_uvcount)
            all_uvcount = np.asarray(all_uvcount)

            face_uvratio = face_uvcount / face_region_area
            nonface_uvratio = nonface_uvcount / nonface_region_area
            all_uvratio = all_uvcount / valid_region_area

            ###
            uv_thresholds_vis = uv_thresholds
            plt.plot(uv_thresholds_vis.tolist(), face_uvratio.tolist(), linestyle="-", color="g", label="face region")
            plt.plot(uv_thresholds_vis.tolist(), nonface_uvratio.tolist(), linestyle="-", color="b", label="nonface reigion")
            plt.plot(uv_thresholds_vis.tolist(), all_uvratio.tolist(), linestyle="-", color="r", label="all region")
            plt.xlabel("Distance error(px)")
            plt.ylabel("Accumulated ratio")
            plt.title("2D Pixel Error")
            plt.legend()

            plt.show()
            end_time = time.time()


        if args.save:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'w') as json_file:
                json.dump(gt2pred, json_file)
