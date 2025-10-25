#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn, move_to_device
from typing import Callable
from torch.optim.swa_utils import AveragedModel
import os
import sys
import h5py
import numpy as np
# %%
def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    logger: LoggingConfig,
):
    # Initialize EMA for model weights using PyTorch's AveragedModel
    swa_start = 1000 # Start using SWA after 1000 iterations

    model = model.to(config.device)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
    ema_model = ema_model.to(config.device)
    state = logger.load_checkpoint()
    if state is not None:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        ema_model.load_state_dict(state['ema_model_state_dict'])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
    else:
        global_step = 0
        start_epoch = 0
    
    logger.global_step = global_step
    avg_loss = OnlineMovingAverage(size=5000)

    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        for images, targets in pb:
            # print(type(images), type(targets))
            model.train()

            # Move batch to device
            images, targets = move_to_device(images, targets, config.device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                continue

            optimizer.zero_grad()
            losses.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()

            if logger.global_step > swa_start and logger.global_step % 5 == 0:
                ema_model.update_parameters(model)
            
            avg_loss.update(losses.item())
            pb.set_description(f"Avg_loss: {avg_loss.mean:.3e}")
            
            if ((logger.global_step + 1) % logger.log_loss_freq == 0) or (logger.global_step == 0):
                loss_test = 0
                num_sample_test = 0
                for images_test, targets_test in val_loader:
                    images_test, targets_test = move_to_device(images_test, targets_test, config.device)

                    loss_test_dict = model(images_test, targets_test)

                    num_sample_test += images_test.shape[0]
                    loss_test += sum(loss for loss in loss_test_dict.values())
                
                metrics = {
                    "avg_loss": loss_test.item()/num_sample_test,
                    "avg_loss": avg_loss.mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max()
                }
                logger.log_metrics(metrics, logger.global_step)
                logger.log_histogram(grad_norm, "grad_norm", logger.global_step)
            if ((logger.global_step+1) % logger.log_image_freq == 0) or (logger.global_step == 0):
                model.eval()
                num_log_images = logger.num_log_images

                # Log images of training set
                images_train = images[:num_log_images]
                targets_pred_train = model(images_train)
                drawn_gt_train = []
                drawn_pred_train = []
                for idx in range(num_log_images):
                    img_with_bb_pred = F.interpolate(
                                            draw_bounding_boxes(images_train[idx], targets_pred_train[idx]['boxes'], colors='red').unsqueeze(0),
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False)
                    drawn_pred_train.append(img_with_bb_pred.to("cpu"))

                    img_with_bb_gt = F.interpolate(
                                            draw_bounding_boxes(images_train[idx], targets[idx]['boxes'], colors='red').unsqueeze(0),
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False)
                    drawn_gt_train.append(img_with_bb_gt.to("cpu"))

                drawn_pred_train = torch.cat(drawn_pred_train, dim=0)
                drawn_gt_train = torch.cat(drawn_gt_train, dim=0)
                # Log images of validation set
                images_val, targets_val = next(iter(val_loader))
                images_val = images_val[:num_log_images]
                targets_val = targets_val[:num_log_images]
                images_val, targets_val = move_to_device(images_val, targets_val, config.device)
                targets_pred_val = model(images_val)
                drawn_gt_val = []
                drawn_pred_val = []
                for idx in range(num_log_images):
                    img_with_bb_pred = F.interpolate(
                                            draw_bounding_boxes(images_val[idx], targets_pred_val[idx]['boxes'], colors='red').unsqueeze(0),
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False
                                            )
                    drawn_pred_val.append(img_with_bb_pred.to("cpu"))

                    img_with_bb_gt = F.interpolate(
                                            draw_bounding_boxes(images_val[idx], targets_val[idx]['boxes'], colors='red').unsqueeze(0),
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False)
                    drawn_gt_val.append(img_with_bb_gt.to("cpu"))
                drawn_pred_val = torch.cat(drawn_pred_val, dim=0)
                drawn_gt_val = torch.cat(drawn_gt_val, dim=0)

                logger.log_images({
                    'x_train': images_train,
                    'train_gt': drawn_gt_train,
                    'train_pred': drawn_pred_train,
                    'x_val': images_val,
                    'val_gt': drawn_gt_val,
                    'val_pred': drawn_pred_val
                }, logger.global_step)

            logger.global_step += 1

        # Save checkpoint
        if ((epoch + 1) == config.num_epochs) or (epoch % logger.save_freq == 0):
            state = {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "global_step": logger.global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }

            logger.save_checkpoint(state, epoch, metric_value=avg_loss.mean)
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()

                

if __name__ == "__main__":
    import sys
    sys.path.append("./../")
    from dataset import VOCDataset, collate_fn
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models import ResNet50_Weights, ResNet18_Weights, MobileNet_V3_Large_Weights
    import torch.utils.data as data
    import argparse  


    # model = fasterrcnn_resnet50_fpn_v2(weights=None,
    #                                 progress=True,
    #                                 num_classes = 21,
    #                                 weights_backbone= ResNet50_Weights.DEFAULT,
    #                                 trainable_backbone_layers=1)
    model_kwargs = dict(
        weights=None,
        progress=True,
        num_classes = 21,
        weights_backbone= MobileNet_V3_Large_Weights.DEFAULT,
        trainable_backbone_layers=1
    )

    model = fasterrcnn_mobilenet_v3_large_320_fpn(**model_kwargs)
    
    print("Number of trainable parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

    parser = argparse.ArgumentParser(description="Training script for fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--images_train", type=str, default='data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', help="Images to use for training")
    parser.add_argument("--annotations_train", type=str, default='data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations', help="Annotations to use for training")
    parser.add_argument("--images_val", type=str, default='data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', help="Dataset to use for validation")
    parser.add_argument("--annotations_val", type=str, default='data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations', help="Dataset to use for validation")
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of images used for training")
    parser.add_argument("--num_epochs", type=int, default=300, help="Numeber of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, help="Saved directory",
                        default='/home/qbao/School/5A/research_project/bacteria_detection_app/exp/default')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create training configuration
    training_config = TrainingConfig()
    training_config.update(**vars(args))

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Create logging configuration
    logger = LoggingConfig(project_dir=os.path.join(abs_path,'exp/object_detection'),
                           exp_name=f"VOC_fasterrcnn_resnet50_fpn_v2_{args.dataset_size}")
    logger.monitor_metric = "avg_loss"
    logger.monitor_mode = "min"
    logger.initialize()
    logger.log_hyperparameters(vars(args), main_key="training_config")

    


    # Create dataset and loader
    # train_dataset = VOCDataset(os.path.join(abs_path, args.images_train),
    #                             os.path.join(abs_path,args.annotations_train))
    # val_dataset = VOCDataset(os.path.join(abs_path, args.images_val),
    #                             os.path.join(abs_path,args.annotations_val))
    # print(len(train_dataset), len(val_dataset))
    train_dataset = VOCDataset(args.images_train, args.annotations_train)
    val_dataset = VOCDataset(args.images_val, args.annotations_val)

    if args.dataset_size is not None:
        print(f"Using only the first {min(args.dataset_size, len(train_dataset))} images from the training set")
        train_dataset = data.Subset(train_dataset, range(min(args.dataset_size, len(train_dataset))))
    
    if len(train_dataset) < args.batch_size:
        sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=args.batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                   shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=False,
                                   num_workers=8)
    val_loader = data.DataLoader(val_dataset, batch_size=logger.num_log_images, shuffle=True, collate_fn=collate_fn, drop_last=False)

    # Save checkpoint every 200 steps
    num_step_per_epoch = max(len(train_loader), 1)
    freq = max(1, int(200 // num_step_per_epoch))
    logger.save_freq = freq
    logger.val_epoch_freq = freq
    logger.log_loss_freq = 5
    logger.log_image_freq = 200

    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)
    
    training_loop(model, optimizer, lr_scheduler, train_loader, val_loader, training_config, logger)




# %%
