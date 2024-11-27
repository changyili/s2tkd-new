import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import torchsummary

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from eval import evaluate
from model.s2tkd import S2TKD
from model.losses import cosine_similarity_loss, focal_loss, l1_loss
import pandas as pd
warnings.filterwarnings("ignore")

def get_parameter(category):
    # fixed_lr = [
    #         # "capsule",
    #         "metal_nut",
    #         # "pill",
    #         # "toothbrush",
    #         # "transistor",
    #         # "wood",
    #         "zipper",
    #         "cable",
    #         "bottle",
    #         "grid",
    #         "hazelnut",
    #         "leather",
    #         "tile",
    #         "carpet",
    #         # "screw",
    #     ]
    # if category == "metal_nut":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "zipper":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "bottle":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "grid":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "hazelnut":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "leather":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "tile":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "carpet":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 9000
    #     steps = 15000
    # elif category == "capsule":
    #     lr1 = 0.001
    #     lr2 = 0.01
    #     de_st_steps = 11000
    #     steps = 17000
    # elif category == "wood":
    #     lr1 = 0.0005
    #     lr2 = 0.005
    #     de_st_steps = 10000
    #     steps = 16000 
    # elif category == "pill":
    #     lr1 = 0.0005
    #     lr2 = 0.01
    #     de_st_steps = 11000
    #     steps = 17000   
    # elif category == "screw":
    #     lr1 = 0.005
    #     lr2 = 0.05
    #     de_st_steps = 10000
    #     steps = 16000   
    # elif category == "transistor":
    #     lr1 = 0.001
    #     lr2 = 0.5
    #     de_st_steps = 11000
    #     steps = 17000 
    # elif category == "cable":
    #     lr1 = 0.001
    #     lr2 = 0.05
    #     de_st_steps = 10000
    #     steps = 16000 
    # elif category == "toothbrush":
    #     lr1 = 0.0005
    #     lr2 = 0.001
    #     de_st_steps = 10000
    #     steps = 17000
    if category == "metal_nut":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    elif category == "zipper":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    elif category == "bottle":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    elif category == "grid":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    elif category == "hazelnut":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3500
        steps = 11500
    elif category == "leather":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 4000
        steps = 12000
    elif category == "tile":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 4500
        steps = 12500
    elif category == "carpet":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 4000
        steps = 12000
    elif category == "capsule":
        lr1 = 0.001
        lr2 = 0.01
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    elif category == "wood":
        lr1 = 0.001
        lr2 = 0.005
        # lr3 = 0.001
        de_st_steps = 4000
        steps = 12000
    elif category == "pill":
        lr1 = 0.001
        lr2 = 0.01
        # lr3 = 0.001
        de_st_steps = 6000
        steps = 16000  
    elif category == "screw":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000  
    elif category == "transistor":
        lr1 = 0.001
        lr2 = 0.5
        # lr3 = 0.001
        de_st_steps = 2000
        steps = 10000
    elif category == "cable":
        lr1 = 0.001
        lr2 = 0.05
        # lr3 = 0.001
        de_st_steps = 4000
        steps = 12000
    elif category == "toothbrush":
        lr1 = 0.001
        lr2 = 0.001
        # lr3 = 0.001
        de_st_steps = 3000
        steps = 11000
    return lr1, lr2, de_st_steps, steps

def train(args, category, lr1, lr2, rotate_90=False, random_rotate=0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    lr1 ,lr2, args.de_st_steps, args.steps = get_parameter(category)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    category_path = './save/' + category
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    # if os.path.exists(os.path.join(args.log_path, run_name + "/")):
    #     shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = S2TKD(dest=True, ed=True).to(device)
    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": lr2},
            {"params": model.segmentation_net.head.parameters(), "lr": lr2},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.feed.parameters(), "lr": lr1},
            {"params": model.student_net.back.bn1.parameters(), "lr": lr1},
            {"params": model.student_net.back.decoder1.parameters(), "lr": lr1},
            {"params": model.student_net.fusionblock.parameters(), "lr": lr1},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    print(args.mvtec_path + category + "/train/good/")
    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0
    best_auc_de_st = 0
    best_aupro_de_st = 0
    best_auc_detect_de_st = 0
    best_auc_seg = 0
    best_aupro_seg = 0
    best_auc_detect_seg = 0
    best_list = []
    best_seg_list = []
    flag = True
    metrics1 = {'class': [], 'pixel_AUC': [], 'pixel_AP': [], 'PRO': [], 'image_AUC': [], 'IAP': [], 'IAP90': [], 'seg_pixel_AUC': [], 'seg_pixel_AP': [], 'seg_PRO': [], 'seg_image_AUC': [], 'seg_IAP': [], 'seg_IAP90': [], 'lr1': [], 'lr2':[], 'epoch':[],'seg_epoch':[]}
    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].to(device)
            img_aug = sample_batched["img_aug"].to(device)
            mask = sample_batched["mask"].to(device)

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation, output_de_st, output_de_st_list = model(
                img_aug, img_origin
            )
            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()
                de_st_optimizer.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)
            if global_step < args.de_st_steps:
                args.eval_per_steps = 1000
            else:
                args.eval_per_steps = 1000
            if global_step > args.de_st_steps and global_step % args.eval_per_steps == 0:
                auc_de_st, ap_de_st, aupro_de_st, auc_detect_de_st, iap_de_st, iap90_de_st, auc_seg, ap_seg, aupro_seg, auc_detect_seg, iap_seg, iap90_seg = evaluate(args, category, model, visualizer, global_step)
                if auc_de_st +aupro_de_st + auc_detect_de_st > best_auc_de_st + best_aupro_de_st + best_auc_detect_de_st:
                    best_auc_de_st = auc_de_st
                    best_aupro_de_st = aupro_de_st
                    best_auc_detect_de_st = auc_detect_de_st
                    best_list = [auc_de_st, ap_de_st, aupro_de_st, auc_detect_de_st, iap_de_st, iap90_de_st, global_step]
                    torch.save(
                        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
                    )
                if auc_seg + aupro_seg + auc_detect_seg > best_auc_seg + best_aupro_seg + best_auc_detect_seg:
                    best_auc_seg = auc_seg
                    best_aupro_seg = aupro_seg
                    best_auc_detect_seg = auc_detect_seg
                    best_seg_list = [auc_seg, ap_seg, aupro_seg, auc_detect_seg, iap_seg, iap90_seg, global_step-args.de_st_steps]
                    torch.save(
                        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
                    )
                metrics1['class'].append(category)
                metrics1['pixel_AUC'].append(auc_de_st)
                metrics1['pixel_AP'].append(ap_de_st)
                metrics1['PRO'].append(aupro_de_st)
                metrics1['image_AUC'].append(auc_detect_de_st)
                metrics1['IAP'].append(iap_de_st)
                metrics1['IAP90'].append(iap90_de_st)
                metrics1['epoch'].append(global_step)
                metrics1['seg_pixel_AUC'].append(auc_seg)
                metrics1['seg_pixel_AP'].append(ap_seg)
                metrics1['seg_PRO'].append(aupro_seg)
                metrics1['seg_image_AUC'].append(auc_detect_seg)
                metrics1['seg_IAP'].append(iap_seg)
                metrics1['seg_IAP90'].append(iap90_seg)
                metrics1['seg_epoch'].append(global_step-args.de_st_steps)
                metrics1['lr1'].append(lr1)
                metrics1['lr2'].append(lr2)
                pd.DataFrame(metrics1).to_csv(category_path+'/metrics_results4.csv', index=False)
            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}, total loss: {round(float(l1_loss_val)+float(focal_loss_val), 4)}"
                    )

            if global_step >= args.steps:
                flag = False
                break
    return best_list, best_seg_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="../RegAD/MVTec/")
    parser.add_argument("--dtd_path", type=str, default="../dtd/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model4/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    # parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument(
        # "--de_st_steps", type=int, default=13000
        "--de_st_steps", type=int, default=9000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]

    # with torch.cuda.device(args.gpu_id):
    #     # lr1 = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    #     lr1 = [0.5, 0.1]
    #     lr2 = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    #     for p in lr1:
    #         for q in lr2:
    #             for obj in no_rotation_category:
    #                 print(obj, "lr1:", p, "lr2:", q)
    #                 train(args, obj, p, q)
    #
    #     # for obj in slight_rotation_category:
    #     #     print(obj)
    #     #     train(args, obj, rotate_90=False, random_rotate=5)
    #     #
    #     # for obj in rotation_category:
    #     #     print(obj)
    #     #     train(args, obj, rotate_90=True, random_rotate=5)


    metrics = {'class': [], 'pixel_AUC': [], 'pixel_AP': [], 'PRO': [], 'image_AUC': [], 'IAP': [], 'IAP90': [], 'seg_pixel_AUC': [], 'seg_pixel_AP': [], 'seg_PRO': [], 'seg_image_AUC': [], 'seg_IAP': [], 'seg_IAP90': [], 'lr1': [], 'lr2':[], 'epoch':[],'seg_epoch':[]}
    # lr1 = 0.005
    # lr2 = 0.01
    # lr_list1 = [0.001, 0.005, 0.01, 0.05, 0.1]
    # lr_list2 = [0.05]
    # lr_list1 = [0.05]
    # lr_list2 = [0.0005, 0.0001]
    # lr_list1 = [0.01, 0.005, 0.001, 0.0005]
    # lr_list2 = [0.01, 0.005]
    lr_list1 = [0.001]
    lr_list2 = [0.05]   
    with torch.cuda.device(args.gpu_id):
        for lr2 in lr_list2:
            for lr1 in lr_list1:
                for obj in no_rotation_category:
                    print(obj, 'lr1:', lr1, 'lr2:', lr2)
                    best_list, best_seg_list = train(args, obj, lr1, lr2)
                    print("================================")
                    print("Denoising Student-Teacher (DeST)")
                    print("pixel_AUC:", best_list[0])
                    print("pixel_AP:",  best_list[1])
                    print("PRO:",  best_list[2])
                    print("image_AUC:",  best_list[3])
                    print("IAP:",  best_list[4])
                    print("IAP90:",  best_list[5])
                    print("Epoch:", best_list[6])
                    print()
                    print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
                    print("pixel_AUC:", best_seg_list[0])
                    print("pixel_AP:", best_seg_list[1])
                    print("PRO:", best_seg_list[2])
                    print("image_AUC:", best_seg_list[3])
                    print("IAP:", best_seg_list[4])
                    print("IAP90:", best_seg_list[5])
                    print("Seg_epoch:", best_seg_list[6])
                    print()
                    metrics['class'].append(obj)
                    metrics['pixel_AUC'].append(best_list[0])
                    metrics['pixel_AP'].append(best_list[1])
                    metrics['PRO'].append(best_list[2])
                    metrics['image_AUC'].append(best_list[3])
                    metrics['IAP'].append(best_list[4])
                    metrics['IAP90'].append(best_list[5])
                    metrics['epoch'].append(best_list[6])
                    metrics['seg_pixel_AUC'].append(best_seg_list[0])
                    metrics['seg_pixel_AP'].append(best_seg_list[1])
                    metrics['seg_PRO'].append(best_seg_list[2])
                    metrics['seg_image_AUC'].append(best_seg_list[3])
                    metrics['seg_IAP'].append(best_seg_list[4])
                    metrics['seg_IAP90'].append(best_seg_list[5])
                    metrics['seg_epoch'].append(best_seg_list[6])
                    metrics['lr1'].append(lr1)
                    metrics['lr2'].append(lr2)
                    pd.DataFrame(metrics).to_csv('./save/metrics_results4.csv', index=False)
        for lr2 in lr_list2:
            for lr1 in lr_list1:
                for obj in slight_rotation_category:
                    print(obj, 'lr1:', lr1, 'lr2:', lr2)
                    best_list, best_seg_list = train(args, obj, lr1, lr2, rotate_90=False, random_rotate=5)
                    print("================================")
                    print("Denoising Student-Teacher (DeST)")
                    print("pixel_AUC:", best_list[0])
                    print("pixel_AP:",  best_list[1])
                    print("PRO:",  best_list[2])
                    print("image_AUC:",  best_list[3])
                    print("IAP:",  best_list[4])
                    print("IAP90:",  best_list[5])
                    print("Epoch:", best_list[6])
                    print()
                    print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
                    print("pixel_AUC:", best_seg_list[0])
                    print("pixel_AP:", best_seg_list[1])
                    print("PRO:", best_seg_list[2])
                    print("image_AUC:", best_seg_list[3])
                    print("IAP:", best_seg_list[4])
                    print("IAP90:", best_seg_list[5])
                    print("Seg_epoch:", best_seg_list[6])
                    print()
                    metrics['class'].append(obj)
                    metrics['pixel_AUC'].append(best_list[0])
                    metrics['pixel_AP'].append(best_list[1])
                    metrics['PRO'].append(best_list[2])
                    metrics['image_AUC'].append(best_list[3])
                    metrics['IAP'].append(best_list[4])
                    metrics['IAP90'].append(best_list[5])
                    metrics['epoch'].append(best_list[6])
                    metrics['seg_pixel_AUC'].append(best_seg_list[0])
                    metrics['seg_pixel_AP'].append(best_seg_list[1])
                    metrics['seg_PRO'].append(best_seg_list[2])
                    metrics['seg_image_AUC'].append(best_seg_list[3])
                    metrics['seg_IAP'].append(best_seg_list[4])
                    metrics['seg_IAP90'].append(best_seg_list[5])
                    metrics['seg_epoch'].append(best_seg_list[6])
                    metrics['lr1'].append(lr1)
                    metrics['lr2'].append(lr2)
                    pd.DataFrame(metrics).to_csv('./save/metrics_results4.csv', index=False)
        for lr2 in lr_list2:
            for lr1 in lr_list1:
                for obj in rotation_category:
                    print(obj, 'lr1:', lr1, 'lr2:', lr2)
                    best_list, best_seg_list = train(args, obj, lr1, lr2, rotate_90=True, random_rotate=5)
                    print("================================")
                    print("Denoising Student-Teacher (DeST)")
                    print("pixel_AUC:", best_list[0])
                    print("pixel_AP:",  best_list[1])
                    print("PRO:",  best_list[2])
                    print("image_AUC:",  best_list[3])
                    print("IAP:",  best_list[4])
                    print("IAP90:",  best_list[5])
                    print("Epoch:", best_list[6])
                    print()
                    print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
                    print("pixel_AUC:", best_seg_list[0])
                    print("pixel_AP:", best_seg_list[1])
                    print("PRO:", best_seg_list[2])
                    print("image_AUC:", best_seg_list[3])
                    print("IAP:", best_seg_list[4])
                    print("IAP90:", best_seg_list[5])
                    print("Seg_epoch:", best_seg_list[6])
                    print()
                    metrics['class'].append(obj)
                    metrics['pixel_AUC'].append(best_list[0])
                    metrics['pixel_AP'].append(best_list[1])
                    metrics['PRO'].append(best_list[2])
                    metrics['image_AUC'].append(best_list[3])
                    metrics['IAP'].append(best_list[4])
                    metrics['IAP90'].append(best_list[5])
                    metrics['epoch'].append(best_list[6])
                    metrics['seg_pixel_AUC'].append(best_seg_list[0])
                    metrics['seg_pixel_AP'].append(best_seg_list[1])
                    metrics['seg_PRO'].append(best_seg_list[2])
                    metrics['seg_image_AUC'].append(best_seg_list[3])
                    metrics['seg_IAP'].append(best_seg_list[4])
                    metrics['seg_IAP90'].append(best_seg_list[5])
                    metrics['seg_epoch'].append(best_seg_list[6])
                    metrics['lr1'].append(lr1)
                    metrics['lr2'].append(lr2)
                    pd.DataFrame(metrics).to_csv('./save/metrics_results4.csv', index=False)