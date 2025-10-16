import torch.optim as optim
import warnings
from torch.utils.tensorboard import SummaryWriter
import psutil
import json
import glob
import re
import datetime
import os


class OptimizationConfig:
    # Optimization setting
    optimizer = "AdamW"  # Options: adamw, adam, sgd, rmsprop
    initial_lr = 1e-3
    weight_decay = 1e-4
    gen_setting_kwargs = dict(lr=initial_lr, weight_decay=weight_decay)

    # adam and adamw specific parameters
    betas = (0.9, 0.999)
    eps = 1e-8

    # SGD specific parameters
    momentum = 0.9
    neserov = True

    # RMSprop specific parameters
    alpha = 0.99
    centered =False

    optim_kwargs = {"AdamW": dict(betas=betas, eps=eps),
                    "adam": dict(betas=betas, eps=eps),
                    "SGD": dict(momentum=momentum, neserov=neserov),
                    "RMSprop": dict(alpha=alpha, centered=centered, eps=eps, momentum=momentum)}

    # learning rate scheduler setting
    lr_scheduler = "CosineAnnealingLR"
    min_lr = 1e-6

    # CosineAnnealingLR specific parameters
    epochs = 60  #T_max for cosing scheduler

    # StepLR
    step_size = 30
    gamma = 0.1

    # ReduceLROnPlateau specific parameters
    factor = 0.1
    patience = 10
    mode = "min"

    # OneCycleLR specific parameters
    max_lr = 1e-2
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4
    
    scheduler_kwargs = {
        "CosineAnnealingLR": dict(eta_min=min_lr,T_max=epochs),
        "StepLR": dict(step_size=step_size, gamma=gamma),
        "ReduceLROnPlateau": dict(min_lr=min_lr, factor=factor, patience=patience, mode=mode),
        "OneCycleLR": dict(min_lr=min_lr, max_lr=max_lr, pct_start=pct_start,
                             div_factor=div_factor, final_div_factor= final_div_factor)
    }

    def get_optimizer(self, model, **kwargs):
        """
        Get the optimizer based on the configuration.
        Args:
            model: The model whose parameters need to be optimized
            **kwargs: Additional argument to the optimizer
        """
        params = model.parameters()
        if self.optim_kwargs[self.optimizer] is not None:
            optimizer_class = getattr(optim, self.optimizer)
            optimizer = optimizer_class(params, **self.gen_setting_kwargs, **self.optim_kwargs[self.optimizer], **kwargs)
            return optimizer
        else:
            warnings.warn(f"Optimizer {self.optimizer} not found, using Adamw instead")
            return optim.AdamW(params, lr=self.initial_lr, **kwargs)

    def get_scheduler(self, optimizer, **kwargs):
        if self.scheduler_kwargs[self.lr_scheduler] is not None:
            scheduler_class = getattr(optim.lr_scheduler, self.lr_scheduler)
            lr_scheduler = scheduler_class(optimizer, **self.scheduler_kwargs[self.lr_scheduler], **kwargs)
            return lr_scheduler
        else:
            warnings.warn(f"Learning rate scheduler {self.lr_scheduler} not found, Using constant lr instead")
            return optim.lr_scheduler.ConstantLR(optimizer, **kwargs)

import torch
class TrainingConfig:
    num_epochs: int=10
    batch_size: int=32
    device: str="cuda" if torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        """
        Update the confiuration with new values
        Args:
        ***kwargs: New values for the configuration parameter
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                warnings.warn(f"New argument {key}")
            setattr(self,key, val)

from collections import deque
class OnlineMovingAverage:
    def __init__(self, size=5000):
        self.size=5000
        self.queue = deque(maxlen=size)
        self.sum = 0.0
        self.mean = 1.0
    def update(self, value):
        if len(self.queue) == self.size:
            self.sum -= self.queue[0]
        self.queue.append(value)
        self.sum += value
        self.mean = self.sum/len(self.queue)

def ema_avg_fn(averaged_model_parameter, model_parameter, n_averaged):
    decay = 0.99
    return decay * averaged_model_parameter + (1 - decay) * model_parameter

class LoggingConfig:
    # Logging frequencies
    log_freq = 10
    save_freq = 1

    # Tensorboard settings
    tensorboard = True
    tensorboard_dir = "runs" # Directory for tensorboard logs
    exp_name = "default" # Experiment name for the run

    # Checkpoint settings
    checkpoint_dir = "checkpoint"
    save_best_only = True
    save_last = True
    max_checkpoint = 5

    # Metrics to monitor
    monitor_metric = "val_loss"  # Metric to monitor
    monitor_mode = "min"        # 'min' for loss, 'max' for metrics like PSNR

    # Logging medias setting
    image_size = (256,256) # resize image in case the images do not have the same size
    image_logging = True
    log_image_freq = 200
    num_log_images= 4 
    log_loss_freq: int=10
    val_epoch_freq: int=5

    # System monitoring
    log_gpu_stats = False  # Log GPU utilization
    log_memory_stats = False    # Log memory usage

    def __init__(self, project_dir: str=None, exp_name: str="default", **kwargs ):
        self.best_metric = float("inf") if self.monitor_mode=="min" else -float("inf")
        self.global_step = 0
        self.epoch = 0
        self.writer = None
        self.project_dir = "" if project_dir is None else project_dir
        self.exp_dir = os.path.join(self.project_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.exp_dir, "runs")

        for key, val in kwargs.items():
            if hasattr(self, key, val):
                warnings.warn(f"Unknown argment {key}")
            setattr(self, key, val)
        
    def initialize(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.writer = None
        
        self.metadata_file = os.path.join(self.exp_dir,"metadata.json")
        self._save_metadata()

    def _save_metadata(self):
        """Save metadata about the training process"""

        metadata = {
            "exp_name": self.exp_name,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "tensorboard_logdir": self.tensorboard_dir if self.tensorboard else None
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
    
    def get_checkpoint_path(self, epoch, metric_value=None):
        """
        Return checkpoint path based on epoch and metric value
        """
        if metric_value is not None:
            return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}_{self.monitor_metric}_{metric_value:.4f}.pth")
        else:
            return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        
    def save_checkpoint(self, state, epoch, metric_value=None):
        """
        Save a checkpoint with training state.

        Args:
            state (dict): State dictionary containing model, optimizwer state etc.
            epoch (int): Current epoch
            metric_value (float): Current value of the monitored metric
        """

        # Update best metric if applicable
        if metric_value is not None:
            is_best = (self.monitor_mode=='min' and metric_value < self.best_metric
                       ) or (self.monitor_mode=='max' and metric_value > self.best_metric)
            if is_best:
                self.best_metric = metric_value
        
        checkpoint_path = self.get_checkpoint_path(epoch, metric_value)
        state.update({
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric
        })

        torch.save(state, checkpoint_path)
        self._save_metadata()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None, verbose=True):
        """Load a checkpoint and return the training state. if no checkpoint_path is provide,
        load the most recent checkpoint. If no checkpoint exists, return None.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. If None, load the latest checkpoint
            verbose (bool): whetther to print the checkpoint_path 
        
        Returns:
            dict: The loaded state dictionary, or None if no checkpoint exists
        """
        if checkpoint_path is None:
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "epoch_*.pth"))
            
            if not checkpoints:
                warnings.warn("No checkpoint found in the experiment directory.")
                return None
            else:
                def get_epoch_num(checkpoint_path):
                    match = re.search("epoch_(\d+)", checkpoint_path)
                    return int(match.group(1)) if match else -1
                
                checkpoint_path = max(checkpoints, key=get_epoch_num)

        else:
            if not os.path.exists(checkpoint_path):
                warnings.warn(f"Checkpoint {checkpoint_path} does not exist.")
                return None
            
        state = torch.load(checkpoint_path)
        self.global_step = state.get("global_step",0) + 1 
        self.epoch = state.get("epoch", 0) + 1
        self.best_metric = state.get("best_metric", self.best_metric)

        if verbose:
            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"Resuming from epoch {self.epoch}")
        
        return state
    
    def log_metrics(self, metrics, step="None", prefix="train"):
        """
        Log metrics to TensorBoard
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int, optional): Current step within the epoch
            prefix (str): Prefix for metrix name (e.g., `train` or `val`)
        """
        if self.writer is None:
            return
        else:
            step = self.global_step if step is None else step

            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{name}", value, step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.writer.add_scalar(f"{prefix}/{name}", value.item(), step)
    
    def log_histogram(self, values, name, step=None):
        """Log a histogram of values to TensorBoard
        Args:
            values (torch.Tensor): Tensor of values to log
            name (str): Name for the histogram
            step (int, optional): Current step within the epoch
        """
        if self.writer is None:
            return
        else:
            step = self.global_step if step is None else step
            self.writer.add_histogram(name, values, step)
    
    def clean_old_checkpoint(self):
        """
        Keep only the top checkpoints based on the monitored metric
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir,"epoch*.pth"))

        if len(checkpoints) >= self.max_checkpoint:
            checkpoints_metrics = []
            metric_pattern = re.compile(rf"{self.monitor_metric}_([0-9.]+)(?=\.pth)")
            for checkpoint in checkpoints:
                match = metric_pattern.search(checkpoint)
                if match:
                    metric_value = float(match.group(1))
                    checkpoints_metrics.append((checkpoint, metric_value))
                else:
                    # Use creation time as fallback
                    creation_time = os.path.getctime(checkpoint)
                    checkpoints_metrics.append((checkpoint, creation_time))
                
            reverse = True if self.monitor_mode == 'max' else False 
            checkpoints_metrics.sort(key=lambda x: x[1], reverse=reverse)
            for checkpoint, _ in checkpoints_metrics[self.max_checkpoint:]:
                os.remove(checkpoint)
    
    def log_images(self, images_dict, step):
        """
        Log images to Tensorboard
        Args:
            images_dict (dictionary): Dictionary of image names and tensors,
            step (int, optional): Current step 
        """

        if (self.writer is None) or (not self.image_logging):
            return
        else:
            step = self.global_step if step is None else step
            for name, images in images_dict.items():
                if isinstance(images, torch.Tensor):
                    if images.ndim  == 3:
                        images = images.unsqueeze(0)
                    
                    images = images[:self.num_log_images]
                    self.writer.add_images(name, images, step)
                self.writer.flush()
    
    def log_figure(self, figure, name, step):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure object
            name (str): Name for the figure
            step (int): Current step/iteration
        """
        if self.writer is None:
            return
        step = self.global_step if step is None else step
        self.writer.add_figure(name, figure, global_step=step)   

    
    def log_system_stats(self, step):
        """
        Log system statistics to TensorBoard.

        Args:
            writer: TensorBoard writer instance
            step (int): Current step/iteration
        """
        if self.writer is None:
            return

        step = self.global_step if step is None else step
        if self.log_gpu_stats and torch.cuda.is_available():
            self.writer.add_scalar(
                "system/gpu_utilization", torch.cuda.utilization(), step
            )
            self.writer.add_scalar(
                "system/gpu_memory_allocated", torch.cuda.memory_allocated(), step
            )

        if self.log_memory_stats:
            self.writer.add_scalar(
                "system/ram_usage_percent", psutil.virtual_memory().percent, step
            )

        self.writer.flush()

    def log_hyperparameters(self, hparams, main_key: str = "hyperparameters"):
        """
        Log hyperparameters to TensorBoard.

        Args:
            hparams (dict): Dictionary of hyperparameters
        """
        if self.writer is None:
            return

        # Log hyperparameters as text
        for key, value in hparams.items():
            self.writer.add_text(f"{main_key}/{key}", str(value))

        self.writer.flush()
            



def move_to_device(images: list, targets: list, device="cpu"):
    images = [img.to(device) for img in images]
    new_targets = []
    for t in targets:
        new_targets.append({
            'boxes': t['boxes'].to(device),
            'labels': torch.tensor(t['labels'], dtype=torch.int64, device=device),
            'scores': torch.tensor(t['scores'], dtype=torch.float32, device=device)
        })
    targets = new_targets
    return images, new_targets

        


        




