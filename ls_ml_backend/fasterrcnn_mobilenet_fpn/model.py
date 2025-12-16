#%%
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os, sys
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
from utils import make_json_safe
from training import (LoggingConfig,
                      OptimizationConfig,
                      OnlineMovingAverage,
                      TrainingConfig,
                      training_loop
                      )
from dataset import VOCDataset, collate_fn
from NNModels import FasterRCNNMobile
from uuid import uuid4
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision.transforms as T
from torchvision import tv_tensors
from torchvision.transforms import v2
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk import LabelStudio
from PIL import Image

import redis
from rq import Queue

#%%
# --------- REDIS SETUP -------------
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_url = f'redis://{redis_host}:{redis_port}'
redis_conn = redis.from_url(redis_url)
q = Queue(connection=redis_conn)
logger = logging.getLogger(__name__)

def get_local_path(url):
    if url.startswith("upload") or url.startswith("/upload"):
        url = "/data" + ("" if url.startswith("/") else "/") + url

    is_uploaded_file = url.startswith("/data/upload")

    project_id = url.split("/")[2]
    filename = os.path.basename(url)


    if is_uploaded_file:
        project_id = url.split("/")[-2]
        filename = os.path.basename(url)
        filepath = os.path.join("/data/ls_data/media/upload",
                                project_id,
                                filename)
        if os.path.exists(filepath):
            return filepath
        else:
            raise FileNotFoundError(f"Uploaded file not found at {filepath}.")
    else:
        filepath = os.path.join("/data/ls_data", url.split("?d=")[1])
        if os.path.exists(filepath):
            return filepath
        else:
            raise FileNotFoundError(f"Local storage file not found at {filepath}.")
        
        
def parse_ls_detection_tasks(tasks: list[dict], value: str="image" ):
    dataset = []
    for task in tasks:
        if len(task.annotations) == 0:
            continue
        image_url = task.data[value]
        image_path = get_local_path(image_url)

        result = task.annotations[0]["result"]
        image_width = result[0]["original_width"]
        image_height = result[0]["original_height"]
        boxes = []
        labels = []
        for ann in result:
            x1, y1 = ann["value"]["x"], ann["value"]["y"]
            width, height = ann["value"]["width"], ann["value"]["height"]
            x2, y2 = min(100, x1 + width), min(100,y1 + height)

            x1, y1 = x1*image_width/100, y1*image_height/100
            x2, y2 = x2*image_width/100, y2*image_height/100


            boxes.append([int(x1), int(y1), int(x2), int(y2)])

            label_id = ann["value"]["rectanglelabels"][0]
            labels.append(label_id)
        dataset.append({
            "original_width": image_width,
            "original_height": image_height,
            "image_path": image_path,
            "annotation": {
                "boxes": boxes,
                "labels": labels
            }
        })
    return dataset


def run_training_task(ls_url, ls_api_key, main_project_id, val_project_id):
    print("=== WORKER: Starting Training Task ===")
    
    # Config setup
    freq = 200
    monitor_metric = "val_avg_map"
    monitor_mode = "max"
    save_freq = freq
    val_epoch_freq = freq
    log_loss_freq = 5
    log_image_freq = 200

    logger_args = dict(monitor_metric=monitor_metric,
                       monitor_mode=monitor_mode,
                       save_freq=save_freq,
                       val_epoch_freq=val_epoch_freq,
                       log_loss_freq=log_loss_freq,
                       log_image_freq=log_image_freq)

    logger_config = LoggingConfig(project_dir='/data/model/exp/object_detection',
                           exp_name=f"VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000",
                           **logger_args)
    logger_config.initialize()



    train_config = TrainingConfig() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    num_step = 1000
    batch_size = 10
    num_epochs = num_step//batch_size + 1
    num_epochs = logger_config.epoch + num_epochs
    
    train_config_params = {'device': device,
                     'dtype': dtype,
                     'num_epochs': num_epochs,
                     'batch_size': batch_size}
    train_config.update(**train_config_params)
    
    # Initialize Model HERE (inside worker), not at module level
    model = FasterRCNNMobile(score_threshold=0.8)
    model.to(train_config.device)
    
    # Load data
    ls = LabelStudio(base_url=ls_url, api_key=ls_api_key)
    print(f"Worker: Fetching tasks for Project {main_project_id}...")
    
    tasks_train = list(ls.tasks.list(project=main_project_id))
    tasks_test = list(ls.tasks.list(project=val_project_id))
    
    raw_dataset_train = parse_ls_detection_tasks(tasks_train, value='image')
    raw_dataset_test = parse_ls_detection_tasks(tasks_test, value='image')
    
    print(f"Worker: Found {len(raw_dataset_train)} train samples, {len(raw_dataset_test)} test samples")

    # Create Datasets
    transform = model.transform
    train_ds = LSDetectionDataset(raw_dataset_train, classes=VOCDataset.voc_cls, transform=transform)
    test_ds = LSDetectionDataset(raw_dataset_test, classes=VOCDataset.voc_cls, transform=transform)

    # Dataloaders
    
    # Sampler logic
    if len(train_ds) < batch_size:
        sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, 
                              sampler=sampler, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), 
                             shuffle=True, pin_memory=True, collate_fn=collate_fn)


    
    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model.model)
    lr_scheduler = optim_config.get_scheduler(optimizer)

    print("Worker: Starting Loop...")
    training_loop(model.model, optimizer, lr_scheduler, train_loader, test_loader, train_config, logger_config)
    
    print("Worker: Training Finished successfully.")
    return True


# ----------- MODEL PREDICTION AND TRAINING SETUP  ----------------------
MAIN_PROJECT_ID = int(os.getenv("MAIN_PROJECT_ID", 1))
VAL_PROJECT_ID = int(os.getenv("VAL_PROJECT_ID", 2))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOGGER_CONFIG = LoggingConfig(project_dir='/data/model/exp/object_detection',
                        exp_name=f"VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000",
                        monitor_metric = "val_avg_map",
                        monitor_mode = "max")

MODEL = FasterRCNNMobile(score_threshold=0.8)
MODEL.to(DEVICE)
TRANSFORM = MODEL.transform

state = LOGGER_CONFIG.load_best_checkpoint()
MODEL.model.load_state_dict(state['model_state_dict'])
MODEL.eval()

class LSDetectionDataset(Dataset):
    def __init__(self, raw_data: list[dict], classes: list[str]=VOCDataset.voc_cls, transform: v2=None):
        cls_to_id = {name: i + 1 for i,name in enumerate(classes)}
        self.classes = classes
        self.cls_to_id = cls_to_id
        self.raw_data = raw_data

        self.transform = transform

    def __len__(self):
        return len(self.raw_data)


    def __getitem__(self, index: int):
        chosen_data = self.raw_data[index]

        img_path = chosen_data["image_path"]
        img = Image.open(img_path).convert("RGB")

        boxes = chosen_data["annotation"]["boxes"]
        boxes = tv_tensors.BoundingBoxes(boxes,
                                         format="XYXY",
                                         canvas_size=(chosen_data['original_width'],
                                                      chosen_data['original_height']))
        
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)
        else:
            img = v2.ToTensor()(img)
        
        labels = [self.cls_to_id[label] for label in chosen_data["annotation"]["labels"]]
        
        return img, {'boxes':torch.tensor(boxes), 'labels': torch.tensor(labels)}


class FasterRCNN(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL','http://label-studio:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
        self.START_TRAINING_EACH_N_UPDATES = 100
        self.set("model_version", f"{self.__class__.__name__}")
        # self.device = TRAIN_CONFIG.device
        self.device = DEVICE

    
    def get_local_path(self, url, project_dir=None, ls_host=None, ls_access_token=None, task_id=None, *args, **kwargs):
        
        return get_local_path(url)
        


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
        image_url = tasks[0]["data"][value]
        local_path = self.get_local_path(image_url)
        image_tensor = TRANSFORM(Image.open(local_path).convert("RGB")).to(self.device)
        with torch.no_grad():
            targets_pred = MODEL.predict(image_tensor)[0]
        results = []
        original_h, original_w = image_tensor.shape[-2:]
        print(f"Image of size: {original_h}x{original_w}")
        total_score = 0
                
        for i in range(len(targets_pred['boxes'])):
            x1, y1, x2, y2 = targets_pred['boxes'][i].tolist()
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"Top-left edge point {(x1,y1)} and bottom-right edge point {(x2,y2)}")
            label = int(targets_pred['labels'][i])
            cls = VOCDataset.voc_cls[label-1]
            score = float(targets_pred['scores'][i])
            label_id = str(uuid4())[:9]
            results.append({
                'id':label_id,
                'from_name': from_name,
                'to_name': to_name,
                'source': '$image',
                'type': 'rectanglelabels',
                'value': {
                    'rotation':0,
                    'width': (x2 - x1) * 100.0 / original_w,
                    'height': (y2 - y1) * 100.0/ original_h,
                    'x': x1 * 100.0 / original_w,
                    'y': y1 * 100.0 / original_h,
                    "rectanglelabels": [cls]
                },
                'score': score,
            })

            total_score += score
        
        total_score /= max(len(targets_pred['boxes']),1)

        predictions = [{
            'result': make_json_safe(results),
            'score': make_json_safe(total_score),
            'model_version': self.get('model_version')
        }]
        
        return ModelResponse(predictions=predictions)

    def parse_ls_detection_tasks(self, tasks: list[dict], value: str="image" ):
            dataset = []
            for task in tasks:
                if len(task.annotations) == 0:
                    continue
                image_url = task.data[value]
                image_path = self.get_local_path(image_url)

                result = task.annotations[0]["result"]
                image_width = result[0]["original_width"]
                image_height = result[0]["original_height"]
                boxes = []
                labels = []
                for ann in result:
                    x1, y1 = ann["value"]["x"], ann["value"]["y"]
                    width, height = ann["value"]["width"], ann["value"]["height"]
                    x2, y2 = min(100, x1 + width), min(100,y1 + height)

                    x1, y1 = x1*image_width/100, y1*image_height/100
                    x2, y2 = x2*image_width/100, y2*image_height/100


                    boxes.append([int(x1), int(y1), int(x2), int(y2)])

                    label_id = ann["value"]["rectanglelabels"][0]
                    labels.append(label_id)
                dataset.append({
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_path": image_path,
                    "annotation": {
                        "boxes": boxes,
                        "labels": labels
                    }
                })
            return dataset
    

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"skip training: event {event} is not supported")
            return

        project = (
            data.get("project") or
            data.get("annotation", {}).get("project") or
            data.get("task", {}).get("project")
        )
        project_id = project['id']
        if project_id != MAIN_PROJECT_ID:
            logger.info("Skip training: fit method is only supported for the main project.")
            return

        logger.debug(f"host:{self.LABEL_STUDIO_HOST}")
        logger.debug(f"API:{self.LABEL_STUDIO_API_KEY}")

        print(f"Server: Received fit request for event {event}. Enqueueing...")

        q.enqueue(run_training_task,
                  args=(self.LABEL_STUDIO_HOST,
                        self.LABEL_STUDIO_API_KEY,
                        MAIN_PROJECT_ID,
                        VAL_PROJECT_ID),
                  job_timeout='2h')

        # ls = LabelStudio(base_url=self.LABEL_STUDIO_HOST, api_key=self.LABEL_STUDIO_API_KEY)
        # tasks_train = list(ls.tasks.list(project=MAIN_PROJECT_ID))
        # tasks_test = list(ls.tasks.list(project=VAL_PROJECT_ID))
        
        
        # # if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
        # if event != 'START_TRAINING':
        #     logger.debug(f"skip training: the number of tasks is not divisible by {self.START_TRAINING_EACH_N_UPDATES}")
        #     return
        

        # train_config = {'device': DEVICE,
        #              'dtype': DTYPE,
        #              'num_epochs': NUM_EPOCHS,
        #              'batch_size': BATCH_SIZE}
        
        # TRAIN_CONFIG.update(**train_config)
        # logger.debug(f"Training configuration: {TRAIN_CONFIG}")
        # LOGGER_CONFIG.log_hyperparameters(train_config, main_key='training_config')

        
        # _, _, value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        # raw_dataset_train = self.parse_ls_detection_tasks(tasks_train, value=value)
        # train_ds = LSDetectionDataset(raw_dataset_train, classes=VOCDataset.voc_cls, transform=TRANSFORM)
        # logger.debug(f"Train set size: {len(train_ds)}")


        # if len(train_ds) < BATCH_SIZE:
        #     sampler = data.RandomSampler(train_ds, replacement=True, num_samples=BATCH_SIZE)
        #     shuffle = False
        # else:
        #     sampler = None
        #     shuffle = True
        # train_loader = DataLoader(train_ds,
        #                       batch_size=BATCH_SIZE,
        #                       shuffle=shuffle,
        #                       sampler=sampler,
        #                       pin_memory=True,
        #                       drop_last=False,
        #                       collate_fn=collate_fn)
        
        # raw_dataset_test = self.parse_ls_detection_tasks(tasks_test, value=value)
        # test_ds = LSDetectionDataset(raw_dataset_test, classes=VOCDataset.voc_cls, transform=TRANSFORM)
        # logger.debug(f"Test set size: {len(test_ds)}")
        
        # test_loader = DataLoader(test_ds,
        #                         batch_size=min(BATCH_SIZE, len(test_ds)),
        #                         shuffle=True,
        #                         pin_memory=True,
        #                         drop_last=False,
        #                         collate_fn=collate_fn)
        
        # print(f"Train loader size: {len(train_loader)}")
        # print(f"Test loader size: {len(test_loader)}")

        # optim_config = OptimizationConfig()
        # optimizer = optim_config.get_optimizer(MODEL.model)
        # lr_scheduler = optim_config.get_scheduler(optimizer)

        # LOGGER_CONFIG.initialize()

        # metadata = LOGGER_CONFIG.load_metadata()

        # TRAIN_CONFIG.num_epochs = metadata.get("epoch", 0) + NUM_EPOCHS

        # print("Starting training loop...")
        # training_loop(MODEL.model, optimizer, lr_scheduler, train_loader, test_loader, TRAIN_CONFIG, LOGGER_CONFIG)
        # print('fit() completed successfully.')
        # state = LOGGER_CONFIG.load_best_checkpoint()
        # # if state is not None
        # MODEL.model.load_state_dict(state['model_state_dict'])

