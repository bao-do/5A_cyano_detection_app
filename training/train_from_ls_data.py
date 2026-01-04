#%%
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from dataset import LSDetectionDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, move_to_device
import json
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

    model = model.to(config.device)       
    global_step = 0
    start_epoch = 0
    
    logger.global_step = global_step
    train_avg_loss = OnlineMovingAverage(size=5000)
    train_avg_map = OnlineMovingAverage(size=1000)

    if test_loader is not None:
        test_avg_loss = OnlineMovingAverage(size=1000)
        test_avg_map = OnlineMovingAverage(size=1000)
    
    print(f"Training with {config.num_epochs} epochs")

    is_first_iteration = True  

    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        accumulation_step = 0
        
        for images, targets in pb:

            model.train()
            images, targets = move_to_device(images, targets, config.device)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                continue

            scaled_loss = losses / num_step_to_accumulate
            
            if accumulation_step == 0:
                optimizer.zero_grad()
            
            scaled_loss.backward()
            accumulation_step += 1

            if accumulation_step == num_step_to_accumulate:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                accumulation_step = 0

            elif is_first_iteration:
                    grad_norm = torch.tensor(0.0) 
            
            train_avg_loss.update(losses.item()/len(images))
            pb.set_description(f"Avg_loss: {train_avg_loss.mean:.3e} | Accum: {accumulation_step}/{num_step_to_accumulate}")
            
            ################## Calcul and Log metrics ##########################
            if ((logger.global_step + 1) % logger.log_loss_freq == 0) or (logger.global_step == 0):
                
                # update loss and map on test set
                test_map = MeanAveragePrecision()
                images_test, targets_test = next(iter(test_loader))
                images_test, targets_test = move_to_device(images_test, targets_test, config.device)

                # loss
                model.train()
                with torch.no_grad():
                    loss_test_dict = model(images_test, targets_test)

                loss_test = sum(loss.detach().item() for loss in loss_test_dict.values())
                test_avg_loss.update(loss_test/len(images_test))

                # map
                model.eval()
                with torch.no_grad():
                    preds = model(images_test)
                test_map.update(preds, targets_test)
                test_avg_map.update(test_map.compute()['map'].item())

                del preds,  loss_test_dict, images_test, targets_test
                torch.cuda.empty_cache()    
                
                # update map on train set
                train_map = MeanAveragePrecision()
                model.eval()
                images_train, targets_train = next(iter(train_loader))
                images_train, targets_train = move_to_device(images_train, targets_train, config.device)

                with torch.no_grad():
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
                    targets_pred_train = model(images_train).detach()
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
                images_test, targets_test = next(iter(test_loader))
                images_test = images_test[:num_log_images]
                targets_test = targets_test[:num_log_images]
                images_test, targets_test = move_to_device(images_test, targets_test, config.device)
                with torch.no_grad():
                    targets_pred_test = model(images_test).detach()
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
                                       drawn_pred_test,
                                       drawn_gt_test],
                                       titles=['train pred', 'train gt',
                                               'test pred',
                                               'test gt'],
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
                "global_step": logger.global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }

            logger.save_checkpoint(state, epoch, metric_value=test_avg_map.mean)

                
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()
        
    logger.clean_old_tensorboard_events()

                

if __name__ == "__main__":
    import sys
    import random
    from dataset import VOCDataset, collate_fn
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import torch.utils.data as data
    from pathlib import Path
    from torchvision.transforms import v2
    

    # get training payload
    payload_path = Path(os.getenv("PAYLOAD_DIR", "/tmp/train_payloads.json"))
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(payload_path, "r") as f:
        payload = json.load(f)

    # remove payload to avoid re-using it
    os.remove(payload_path)

    raw_dataset = payload.get("raw_train_data", [])
    class_str = payload.get("class_str", "")

    if len(class_str) == 0:
        raise ValueError("No class provided in the training payload.")

    # initiate logging parameters
    save_dir = os.getenv("SAVE_DIR", "exp/object_detection/")
    save_dir = os.path.join("/app/", save_dir)

    exp_name = os.getenv("EXP_NAME", "VOC_fasterrcnn_resnet50_fpn_v2")
    freq = int(os.getenv("SAVE_FREQ", "200"))
    monitor_metric = os.getenv("MONITOR_METRIC", "val_avg_map")
    monitor_mode = os.getenv("MONITOR_MODE", "max")
    val_epoch_freq = int(os.getenv("VAL_EPOCH_FREQ", freq))
    log_loss_freq = int(os.getenv("LOG_LOSS_FREQ", "5"))
    log_image_freq = int(os.getenv("LOG_IMAGE_FREQ", "200"))


    # initiate training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    train_test_ratio = float(os.getenv("TRAIN_TEST_RATIO", "0.8")) 
    num_step = int(os.getenv("NUM_STEP", "1000"))
    batch_size_to_accumulate = int(os.getenv("BATCH_SIZE_TO_ACCUMULATE", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "10"))
    num_step_to_accumulate = max(1, batch_size_to_accumulate // batch_size) 

    # create datasets

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

    random.shuffle(raw_dataset)
    split_idx = min(int((1 - train_test_ratio) * len(raw_dataset)), 200)

    raw_dataset_train = raw_dataset[:-split_idx]
    raw_dataset_test = raw_dataset[-split_idx:]

    train_ds = LSDetectionDataset(raw_dataset_train, classes=class_str, transform=transform_train)
    test_ds = LSDetectionDataset(raw_dataset_test, classes=class_str, transform=transform_test)

    
    # Sampler logic
    if len(train_ds) < batch_size:
        sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Dataloaders
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, 
                              sampler=sampler, pin_memory=True, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), 
                             shuffle=True, pin_memory=True, collate_fn=collate_fn)

    # initiate logger
    logger_args = dict(monitor_metric=monitor_metric,
                        monitor_mode=monitor_mode,
                        save_freq=freq,
                        val_epoch_freq=val_epoch_freq,
                        log_loss_freq=log_loss_freq,
                        log_image_freq=log_image_freq)

    logging_config = LoggingConfig(project_dir=save_dir,
                                  exp_name=exp_name,
                                  load_old_ckpt=False,
                                  **logger_args)
    logging_config.initialize()
    
    # Training config
    num_steps_per_epoch = len(raw_dataset)//batch_size + 1
    num_epoch = max(1, num_step // num_steps_per_epoch) + logging_config.epoch


    train_config_params = dict(
        device=device,
        dtype=dtype,
        num_epochs=num_epoch,
        batch_size=batch_size,
    )
    training_config = TrainingConfig()
    training_config.update(**train_config_params)
    
    # Initiate model
    model_kwargs = dict(
        weights=None,
        progress=True,
        num_classes = len(class_str),
        weights_backbone= FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=1
    )

    model = fasterrcnn_resnet50_fpn_v2(**model_kwargs)
    model.to(training_config.device)


    # Optimizer and LR scheduler
    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)

    # Start training loop
    print("Worker: Starting Loop...")
    training_loop(model, optimizer, lr_scheduler, train_loader, test_loader, training_config, logging_config, num_step_to_accumulate)
    
    print("Worker: Training Finished successfully.")

    # Save class_str used for training
    with open(payload_path,'w') as f:
        json.dump({'class_str': class_str}, f)
    

    



# %%
