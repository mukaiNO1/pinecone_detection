import os
from collections import defaultdict

import six
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from data_loader import get_dataloader, get_transforms
from mask_keypoint_rcnn import maskkeypointrcnn_resnet34_fpn, maskkeypointrcnn_resnet50_fpn, maskkeypointrcnn_resnet101_fpn
from mask_keypoint_rcnn_mobnet import maskkeypointrcnn_mobilenet
from test import predict
from tqdm import tqdm
import os
import cv2
import time
from engine import train_one_epoch, evaluate
import itertools
import pylab
import numpy as np

import csv
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

class WarmUpLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warm_up_epochs, warm_up_factor):
        def f(epoch):
            if epoch >= warm_up_epochs:
                return 1
            alpha = epoch / warm_up_epochs
            return warm_up_factor * (1-alpha) + 1.0 * alpha
        super(WarmUpLR, self).__init__(optimizer, f)


def train_and_eval(model,
                   dataloaders,
                   epochs, batches_show=10,
                   lr=0.0025,
                   keypoint_weight=1.0, mask_weight=1.0,
                   save_dir=None, save_interval=10,
                   load_from=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is unavailable, using CPU instead.")

    dataloader_train, dataloader_val = dataloaders
    val = False if dataloader_val is None else True

    model = model.to(device)
    if load_from is not None:
        model.load_state_dict(torch.load(load_from))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)


    for epoch in range(1, epochs+1):
        model.train()
        print("----------------------  TRAINING  ---------------------- ")
        running_losses = {}
        for i, data in enumerate(dataloader_train, start=1):
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss_dict["loss_keypoint"] = loss_dict["loss_keypoint"] * keypoint_weight
            loss_dict["loss_mask"] = loss_dict["loss_mask"] * mask_weight
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                if k in running_losses:
                    running_losses[k] += v.item()
                else:
                    running_losses[k] = v.item()

            #if i % batches_show == 0:
        running_losses = {k: v / batches_show for k, v in running_losses.items()}
        running_loss = sum(v for v in running_losses.values())
        summary = f'[epoch: {epoch}] [loss: {running_loss:.3f}] ' + " ".join([f"[{k}: {v:.3f}]" for k, v in running_losses.items()])
        print(summary)
        running_losses = {}
        lr_scheduler.step(loss)

        # if val:
        #     model.eval()
        #     print("----------------------  Val  ---------------------- ")
        #     running_losses = {}
        #     for i, data in enumerate(dataloader_val, start=1):
        #         images, targets = data
        #         images = [image.to(device) for image in images]
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #
        #         result = model(images)[0]

                # x = compute_recall(result['boxes'].detach().cpu().numpy(), targets[0]['boxes'].detach().cpu().numpy(), 0.5)
                # # y = compute_overlaps(targets[0]['boxes'].detach().cpu().numpy(), result['boxes'].detach().cpu().numpy())
                # y = compute_ap(
                #     targets[0]['boxes'].detach().cpu().numpy(),
                #     targets[0]['labels'].detach().cpu().numpy(),
                #     targets[0]['masks'].detach().cpu().numpy(),
                #     result['boxes'].detach().cpu().numpy(),
                #     result['labels'].detach().cpu().numpy(),
                #     result['scores'].detach().cpu().numpy(),
                #     result['masks'].detach().cpu().numpy())
                # print(x, y)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if epoch % save_interval == 0:
                print("----------------------   SAVING   ---------------------- ")
                # torch.save(model, os.path.join(save_dir, "epoch_{}.pth".format(epoch)))
                torch.save(model.state_dict(), os.path.join(save_dir, "epoch_{}.state_dict.pth".format(epoch)))


def eval_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2)
    model.load_state_dict(torch.load("work_dir_res50.test/epoch_13.state_dict.pth"))
    dataloaders=get_dataloader(img_dir="data/tmp_database/final_dataset/img_test/", lab_path="data/tmp_database/final_dataset/annotation_test.json",
                                   kp_names=("l", "r"),
                                   train=True, batch_size=1, val_split=0.0 , shuffle=True,  num_workers=0)
    model = model.to(device)
    model.eval()
    dataloader_train, dataloader_val = dataloaders
    print("----------------------  Val  ---------------------- ")
    running_losses = {}
    e1=evaluate(model, dataloader_train, device=device)

    model = maskkeypointrcnn_resnet101_fpn(num_classes=2, num_keypoints=2)
    model.load_state_dict(torch.load("work_dir_res101.test/epoch_13.state_dict.pth"))
    dataloaders=get_dataloader(img_dir="data/tmp_database/final_dataset/img_test/", lab_path="data/tmp_database/final_dataset/annotation_test.json",
                                   kp_names=("l", "r"),
                                   train=True, batch_size=1, val_split=0.0 , shuffle=True,  num_workers=0)
    model = model.to(device)
    model.eval()
    dataloader_train, dataloader_val = dataloaders
    print("----------------------  Val  ---------------------- ")
    running_losses = {}
    e2= evaluate(model, dataloader_train, device=device)

    model = maskkeypointrcnn_mobilenet(num_classes=2, num_keypoints=2)
    model.load_state_dict(torch.load("work_dir_mobilenetv3.test/epoch_13.state_dict.pth"))
    dataloaders=get_dataloader(img_dir="data/tmp_database/final_dataset/img_test/", lab_path="data/tmp_database/final_dataset/annotation_test.json",
                                   kp_names=("l", "r"),
                                   train=True, batch_size=1, val_split=0.0 , shuffle=True,  num_workers=0)
    model = model.to(device)
    model.eval()
    dataloader_train, dataloader_val = dataloaders
    print("----------------------  Val  ---------------------- ")
    running_losses = {}

    e3= evaluate(model, dataloader_train, device=device)
    model = maskkeypointrcnn_mobilenet(num_classes=2, num_keypoints=2)
    model.load_state_dict(torch.load("work_dir_mobilenetv3_awg.test/epoch_13.state_dict.pth"))
    dataloaders=get_dataloader(img_dir="data/tmp_database/final_dataset/img_test/", lab_path="data/tmp_database/final_dataset/annotation_test.json",
                                   kp_names=("l", "r"),
                                   train=True, batch_size=1, val_split=0.0 , shuffle=True,  num_workers=0)
    model = model.to(device)
    model.eval()
    dataloader_train, dataloader_val = dataloaders
    print("----------------------  Val  ---------------------- ")
    running_losses = {}
    e4= evaluate(model, dataloader_train, device=device)


    return e1, e2, e3 ,e4



def eval():
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2)
    # model.load_state_dict(torch.load("work_dir_res101.test/epoch_13.state_dict.pth"))
    # dataloaders=get_dataloader(img_dir="data/tmp_database/final_dataset/img_test/", lab_path="data/tmp_database/final_dataset/annotation_test.json",
    #                                kp_names=("l", "r"),
    #                                train=True, batch_size=1, val_split=0.0 , shuffle=True,  num_workers=0)
    # model = model.to(device)
    # model.eval()
    # dataloader_train, dataloader_val = dataloaders
    # print("----------------------  Val  ---------------------- ")
    # running_losses = {}
    # e= evaluate(model, dataloader_train, device=device)
    #
    # print("val", e.coco_eval["bbox"].eval["precision"].mean(), e.coco_eval["segm"].eval["precision"].mean(),
    #       e.coco_eval["keypoints"].eval["precision"].mean())
    e1, e2, e3, e4  = eval_model()

    pr_array1 = e1.coco_eval["keypoints"].eval["precision"][0, :, 0, 0, 0]
    print(np.average(pr_array1))
    pr_array2 = e2.coco_eval["keypoints"].eval["precision"][0, :, 0, 0, 0]
    print(np.average(pr_array2))
    pr_array3 = e3.coco_eval["keypoints"].eval["precision"][0, :, 0, 0, 0]
    print(np.average(pr_array3))
    pr_array4 = e4.coco_eval["keypoints"].eval["precision"][0, :, 0, 0, 0]
    print(np.average(pr_array4))

    # pr_array2 = e2.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    # pr_array2 = e2.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    # pr_array2 = e2.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    # pr_array3 = e.coco_eval["keypoints"].eval["precision"][0, :, 0, 0, 2]
    bbox_pr_array1 = e1.coco_eval["bbox"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array1))
    bbox_pr_array2 = e2.coco_eval["bbox"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array2))
    bbox_pr_array3= e3.coco_eval["bbox"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array3))
    bbox_pr_array4 = e4.coco_eval["bbox"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array4))

    segm_pr_array1 = e1.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array1))
    segm_pr_array2 = e2.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array2))
    segm_pr_array3= e3.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array3))
    segm_pr_array4 = e4.coco_eval["segm"].eval["precision"][0, :, 0, 0, 2]
    print(np.average(pr_array4))

    x = np.arange(0.0, 1.01, 0.01)
    # x_1 = np.arange(0, 1.01, 0.111)
    plt.xlabel('Recal')
    plt.ylabel('Precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)

    for i in range(len(pr_array1)):
        with open("plot_bbox.csv", 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([x[i], bbox_pr_array1[i], bbox_pr_array2[i], bbox_pr_array3[i], bbox_pr_array4[i]])

        with open("plot_segm.csv", 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([x[i], segm_pr_array1[i], segm_pr_array2[i], segm_pr_array3[i], segm_pr_array4[i]])

        with open("plot_kp.csv", 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([x[i], pr_array1[i], pr_array2[i], pr_array3[i], pr_array4[i]])


    # plt.plot(x, pr_array1, 'b-', label='res50')
    # plt.plot(x, pr_array2, 'g-', label='res101')
    # plt.plot(x, pr_array3, 'r-', label='mobilenet')
    # plt.plot(x, pr_array4, 'y-', label='mobilenet_awg')
    # # plt.plot(x, pr_array3, 'b-', label='keypoints')
    # #plt.title("iou=0.5 catid=person maxdet=100")
    #
    # plt.legend(loc="lower left")
    # plt.show()

    # for i, data in enumerate(dataloader_train, start=1):
    #     images, targets = data
    #     images = [image.to(device) for image in images]
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     loss_dict = model(images)[0]
    #     result = model(images)[0]


@ torch.no_grad()
def test_model(model, candidates_dir,
               kp_names=("l", "r"),
               box_score_thre=0.5, kp_score_thre=0, mask_thre=0.5,
               save_image_out=True, show_image_out=False):
    out_dir = candidates_dir.rstrip("/") + ".detected"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()


    transforms = get_transforms(train=False)
    for p in tqdm(os.listdir(candidates_dir)):
        image_path = os.path.join(candidates_dir, p)
        image_cv2 = cv2.imread(image_path)

        canvas, results = predict(image_cv2, model,
                                  transforms=transforms, device=device, kp_names=kp_names,
                                  box_score_thre=box_score_thre, kp_score_thre=kp_score_thre, mask_thre=mask_thre,
                                  show=False)

        if save_image_out:
            cv2.imwrite(os.path.join(out_dir, p), canvas)
        if show_image_out:
            cv2.imshow("res", canvas)
            cv2.waitKey(0)
            cv2.destroyWindow("res")


if __name__ == '__main__':
    # train_and_eval(
    #     model=maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2, pretrained=True),
    #     dataloaders=get_dataloader(img_dir="data/tmp_database/img/", lab_path="data/tmp_database/annotation.json",
    #                                kp_names=("l", "r"),
    #                                train=True, batch_size=2, val_split=0.2 , shuffle=True, num_workers=0),
    #     epochs=500, batches_show=1, save_dir="work_dir.test", save_interval=99,
    #     lr=0.001,
    #     keypoint_weight=0.1, mask_weight=1.0,
    #     load_from=None)
    #
    # model = maskkeypointrcnn_resnet50_fpn(num_classes=2, num_keypoints=2)
    # model.load_state_dict(torch.load("work_dir.test/epoch_6.state_dict.pth"))
    # test_model(model, "data/tmp_database/img3",
    #            box_score_thre=0.5, kp_score_thre=0.5, mask_thre=0.5,
    #            save_image_out=True, show_image_out=False)

    eval()
