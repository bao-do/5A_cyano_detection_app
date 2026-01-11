#%%
import sys, os
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(abs_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn, move_to_device
from typing import Callable
from torch.optim.swa_utils import AveragedModel
import os
import sys
import numpy as np
import deepinv as dinv


# %%
def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    logger: LoggingConfig,
    num_step_to_accumulate: int = 1,
):
    # Initialize EMA for model weights using PyTorch's AveragedModel
    swa_start = 1000 # Start using SWA after 1000 iterations

    # Move model to device FIRST before loading any states
    model = model.to(config.device)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
    ema_model = ema_model.to(config.device)
    
    state = logger.load_latest_checkpoint()
    if state is not None:
        # Load model state
        model.load_state_dict(state['model_state_dict'])
        ema_model.load_state_dict(state['ema_model_state_dict'])
        
        # Load optimizer state with device mapping
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # Move optimizer states to correct device
        for state_param in optimizer.state.values():
            if isinstance(state_param, dict):
                for k, v in state_param.items():
                    if torch.is_tensor(v):
                        state_param[k] = v.to(config.device)
        
        # Load scheduler state
        if state['scheduler_state_dict'] is not None:
            lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        
        print(f"Resumed from global_step={global_step}, epoch={start_epoch}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")
    else:
        global_step = 0
        start_epoch = 0
    
    logger.global_step = global_step
    train_avg_loss = OnlineMovingAverage(size=5000)
    train_avg_map = OnlineMovingAverage(size=1000)

    if test_loader is not None:
        test_avg_loss = OnlineMovingAverage(size=1000)
        test_avg_map = OnlineMovingAverage(size=1000)
    
    test_map = MeanAveragePrecision()
    train_map = MeanAveragePrecision()

    print(f"Training with {config.num_epochs} epochs")

    is_first_iteration = True

    for epoch in range(start_epoch, config.num_epochs):
        # Reset metrics at start of each epoch
        test_map.reset()
        train_map.reset()
        
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        accumulation_step = 0
        
        for images, targets in pb:

            model.train()
            images, targets = move_to_device(images, targets, config.device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                continue

            # Scale loss by accumulation steps
            scaled_loss = losses / num_step_to_accumulate
            
            # Zero gradients at start of accumulation cycle
            if accumulation_step == 0:
                optimizer.zero_grad()
            
            scaled_loss.backward()
            accumulation_step += 1

            # Optimizer step only after accumulation
            if accumulation_step == num_step_to_accumulate:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                accumulation_step = 0
            elif is_first_iteration:
                    grad_norm = torch.tensor(0.0)  # Placeholder for non-update steps

            if logger.global_step > swa_start and logger.global_step % 5 == 0:
                ema_model.update_parameters(model)            
            train_avg_loss.update(losses.item()/len(images))
            pb.set_description(f"Avg_loss: {train_avg_loss.mean:.3e}")
            
            ################## Calcul and Log metrics ##########################
            if ((logger.global_step + 1) % logger.log_loss_freq == 0) or (logger.global_step == 0):
                
                if test_loader is not None:
                    # update loss and map on test set
                    images_test, targets_test = next(iter(test_loader))
                    images_test, targets_test = move_to_device(images_test, targets_test, config.device)
                    with torch.no_grad():
                        # loss
                        model.train()
                        loss_test_dict = model(images_test, targets_test)
                        # map
                        model.eval()
                        preds = model(images_test)

                    loss_test = sum(loss.detach().item() for loss in loss_test_dict.values())
                    test_avg_loss.update(loss_test/len(images_test))

                    test_map.update(preds, targets_test)
                    test_avg_map.update(test_map.compute()['map'].item())

                    del preds,  loss_test_dict, images_test, targets_test
                    torch.cuda.empty_cache()    
                
                # update map on train set
                images_train, targets_train = next(iter(train_loader))
                images_train, targets_train = move_to_device(images_train, targets_train, config.device)
                
                with torch.no_grad():
                    model.eval()
                    preds_train = model(images_train)

                train_map.update(preds_train, targets_train)
                train_avg_map.update(train_map.compute()['map'].item())

                del preds_train, images_train, targets_train
                torch.cuda.empty_cache()
                
                # log metric
                metrics = {
                    "validation_loss": None if test_loader is None else test_avg_loss.mean,
                    "map_val": None if test_loader is None else test_avg_map.mean ,
                    "train_loss": train_avg_loss.mean,
                    "map_train": train_avg_map.mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max()
                }
                logger.log_metrics(metrics, logger.global_step)
                logger.log_histogram(grad_norm, "grad_norm", logger.global_step)

            ############################ LOG IMAGES ####################################3
            if ((logger.global_step+1) % logger.log_image_freq == 0) or (logger.global_step == 0):
                model.eval()
                num_log_images = logger.num_log_images

                # Log images of training set
                images_train = images[:num_log_images]
                with torch.no_grad():
                    targets_pred_train = model(images_train)
                drawn_gt_train = []
                drawn_pred_train = []
                images_train = [(img*255).clamp(0,255).to(torch.uint8) for img in images_train]
                for idx in range(num_log_images):
                    img_with_bb_pred = F.interpolate(
                                            draw_bounding_boxes(images_train[idx],targets_pred_train[idx]['boxes'], colors='red').unsqueeze(0).float()/255.0
                                            if targets_pred_train[idx]['boxes'].shape[0] != 0
                                            else images_train[idx].unsqueeze(0).float()/255.0,
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False)
                    drawn_pred_train.append(img_with_bb_pred.to("cpu"))

                    img_with_bb_gt = F.interpolate(
                                            draw_bounding_boxes(images_train[idx], targets[idx]['boxes'], colors='red').unsqueeze(0).float()/255.0
                                            if targets[idx]['boxes'].shape[0] != 0
                                            else images_train[idx].unsqueeze(0).float()/255.0,
                                            size = logger.image_size,
                                            mode='bilinear',
                                            align_corners=False)
                    drawn_gt_train.append(img_with_bb_gt.to("cpu"))

                drawn_pred_train = torch.cat(drawn_pred_train, dim=0)
                drawn_gt_train = torch.cat(drawn_gt_train, dim=0)

                # Log images of test set
                if test_loader is not None:
                    images_test, targets_test = next(iter(test_loader))
                    images_test = images_test[:num_log_images]
                    targets_test = targets_test[:num_log_images]
                    images_test, targets_test = move_to_device(images_test, targets_test, config.device)
                    with torch.no_grad():   
                        targets_pred_test = model(images_test)
                    drawn_gt_test = []
                    drawn_pred_test = []
                    images_test = [(img*255).clamp(0,255).to(torch.uint8) for img in images_test]
                    for idx in range(num_log_images):
                        img_with_bb_pred = F.interpolate(
                                                draw_bounding_boxes(images_test[idx], targets_pred_test[idx]['boxes'], colors='red').unsqueeze(0).float()/255.0
                                                if targets_pred_test[idx]['boxes'].shape[0] != 0
                                                else images_test[idx].unsqueeze(0).float()/255.0,
                                                size = logger.image_size,
                                                mode='bilinear',
                                                align_corners=False
                                                )
                        drawn_pred_test.append(img_with_bb_pred.to("cpu"))

                        img_with_bb_gt = F.interpolate(
                                                draw_bounding_boxes(images_test[idx], targets_test[idx]['boxes'], colors='red').unsqueeze(0).float()/255.0
                                                if targets_test[idx]['boxes'].shape[0] != 0
                                                else images_test[idx].unsqueeze(0).float()/255.0 ,
                                                size = logger.image_size,
                                                mode='bilinear',
                                                align_corners=False)
                        drawn_gt_test.append(img_with_bb_gt.to("cpu"))
                    drawn_pred_test = torch.cat(drawn_pred_test, dim=0)
                    drawn_gt_test = torch.cat(drawn_gt_test, dim=0)
                
                fig = dinv.utils.plot([drawn_pred_train,
                                       drawn_gt_train,
                                       drawn_pred_test if test_loader is not None else None,
                                       drawn_gt_test if test_loader is not None else None],
                                       titles=['train pred', 'train gt',
                                               'test pred' if test_loader is not None else None,
                                               'test gt' if test_loader is not None else None],
                                        return_fig=True,   
                                        show=False)
                logger.log_figure(figure=fig,name="samples from train and test sets",step=logger.global_step)


            ##################### UPDATE GLOBAL STEP #########################3
            logger.global_step += 1
            is_first_iteration = False

        ######################### SAVE CHECKPOINT #########################
        if ((epoch + 1) == config.num_epochs) or (epoch % logger.save_freq == 0):
            state = {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "global_step": logger.global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }

            if test_loader is not None:
                logger.save_checkpoint(state, epoch, metric_value=test_avg_map.mean)
            else:
                logger.save_checkpoint(state, epoch, metric_value=train_avg_map.mean)
                
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()
        
    logger.clean_old_tensorboard_events()

                


#%%
from dataset import VOCDataset, collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch.utils.data as data
import argparse  



args = dict(images_train = 'data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
            annotations_train = 'data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
            images_val = 'data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
            annotations_val = 'data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
            train_dataset_size = 1000,
            test_dataset_size = 200,
            num_epochs = 40,
            batch_size = 10,
            save_dir = 'exp/object_detection',
            exp_name = f"VOC_fasterrcnn_resnet50_fpn_v2_test",)


abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


#################################### DEFINE MODEL #########################
    
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze box predictor (detection head)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

# # Unfreeze specific backbone layers (layer3, layer4)
# resnet = model.backbone.body
# for layer_name in ["layer3", "layer4"]:
#     layer = getattr(resnet, layer_name)
#     for param in layer.parameters():
#         param.requires_grad = True

# # Unfreeze RPN head
# for param in model.rpn.head.parameters():
#     param.requires_grad = True

# Unfreeze ROI box head
for param in model.roi_heads.box_head.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable params: {trainable / 1e6:.2f}M")
print(f"Total params: {total / 1e6:.2f}M")


############################## DEFINE DATASET ###########################################
transform_test = v2.Compose([
    v2.ToDtype(dtype=torch.float32, scale=True)
])

transform_train = v2.Compose([
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandomGrayscale(p=0.1),
    v2.GaussianNoise(),
    v2.ColorJitter(),
])
train_dataset = VOCDataset(os.path.join(abs_path, args['images_train']), os.path.join(abs_path, args['annotations_train']), transform=transform_train)

#%%
if args['train_dataset_size'] is not None:
    print(f"Using the first {min(args['train_dataset_size'], len(train_dataset))} images from the training set")
    train_dataset = data.Subset(train_dataset, range(min(args['train_dataset_size'], len(train_dataset))))

if len(train_dataset) < args['batch_size']:
    sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=args['batch_size'])
    shuffle = False
else:
    sampler = None
    shuffle = True

train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(42)
train_loader = data.DataLoader(train_dataset, batch_size=args['batch_size'], collate_fn=collate_fn,
                                shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=False,
                                num_workers=8, generator=train_loader_generator)

val_dataset = VOCDataset(os.path.join(abs_path, args['images_val']), os.path.join(abs_path, args['annotations_val']), transform=transform_test)
if len(val_dataset) == 0:
    test_loader=None
else:
    if args['test_dataset_size'] is not None:
        print(f"Using the first {min(args['test_dataset_size'], len(val_dataset))} images as validation set")
        val_dataset = data.Subset(val_dataset, range(min(args['test_dataset_size'], len(val_dataset))))
    test_loader = data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=False)


################################ Training and Saving Configuration #####################
training_config = TrainingConfig()
training_config.update(**args)


# Save checkpoint every 200 steps
monitor_metric = "val_avg_map" if test_loader is not None else "train_avg_map"
monitor_mode = "max"
num_step_per_epoch = max(len(train_loader), 1)
freq = max(1, int(200 // num_step_per_epoch))
save_freq = freq
val_epoch_freq = freq
log_loss_freq = 5
log_image_freq = 200
num_log_images = 2
logger_args = dict(monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode,
                    save_freq=save_freq,
                    val_epoch_freq=val_epoch_freq,
                    log_loss_freq=log_loss_freq,
                    log_image_freq=log_image_freq,
                    num_log_images=num_log_images)

logger = LoggingConfig(project_dir=os.path.join(abs_path,args['save_dir']),
                        exp_name=args['exp_name'],
                        **logger_args
                        )
batch_size_to_calculate_grad = 100
num_step_to_accumulate = max(1, batch_size_to_calculate_grad // args['batch_size'])

logger.initialize()
logger.log_hyperparameters(args, main_key="training_config")

optim_config = OptimizationConfig()
optimizer = optim_config.get_optimizer(model)
lr_scheduler = optim_config.get_scheduler(optimizer)

########################### LANCE TRAINING LOOP ##############################################
training_loop(model, optimizer, lr_scheduler, train_loader, test_loader, training_config, logger, num_step_to_accumulate)


# %%
sample = val_dataset[0][0].unsqueeze(0).to(training_config.device)
model.eval()

with torch.no_grad():
    output = model(sample)

model_from_ckpt = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model_from_ckpt.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

state = logger.load_best_checkpoint()
model_from_ckpt.load_state_dict(state['model_state_dict'])
model_from_ckpt = model_from_ckpt.to(training_config.device)
model_from_ckpt.eval()

with torch.no_grad():
    output_from_ckpt = model_from_ckpt(sample)

draw_img = draw_bounding_boxes(sample[0], output[0]['boxes'], colors='red').float()/255.0
draw_img_ckpt = draw_bounding_boxes(sample[0], output_from_ckpt[0]['boxes'], colors='red').float()/255.0

from deepinv.utils import plot

plot([draw_img.unsqueeze(0), draw_img_ckpt.unsqueeze(0)], titles=['Original', 'loaded cpkt'], figsize=(8,4))

# model state dicts match

# %% check optimizer state dict
opt_state_ckpt = state['optimizer_state_dict']
optimizer_state = optimizer.state_dict()
def compare_optimizer_states(state1, state2):
    """Compare two optimizer state dicts."""
    if state1['param_groups'] != state2['param_groups']:
        print("param_groups differ")
        return False
    
    if set(state1['state'].keys()) != set(state2['state'].keys()):
        print("state keys differ")
        return False
    
    for key in state1['state']:
        for param_name in state1['state'][key]:
            val1 = state1['state'][key][param_name]
            val2 = state2['state'][key][param_name]
            
            if torch.is_tensor(val1) and torch.is_tensor(val2):
                if not torch.allclose(val1, val2):
                    print(f"Tensor mismatch at state[{key}][{param_name}]")
                    return False
            elif val1 != val2:
                print(f"Value mismatch at state[{key}][{param_name}]: {val1} vs {val2}")
                return False
    
    print("Optimizers are the same!")
    return True

compare_optimizer_states(opt_state_ckpt, optimizer_state)

# Optimizers match




# %% check learning rate scheduler state

sched_state_ckpt = state['scheduler_state_dict']
sched_state = lr_scheduler.state_dict()

def compare_scheduler_states(state1, state2):
    """Compare two scheduler state dicts."""
    for key in state1:
        val1 = state1[key]
        val2 = state2[key]
        
        if val1 != val2:
            print(f"Value mismatch at {key}: {val1} vs {val2}")
            return False
    print("Schedulers are the same!")
    return True
compare_scheduler_states(sched_state_ckpt, sched_state)

# learning rate schedulers match


