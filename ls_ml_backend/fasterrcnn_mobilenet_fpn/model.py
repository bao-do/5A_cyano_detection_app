#%%
import os, sys
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
from utils import make_json_safe
from training import (LoggingConfig,
                      OptimizationConfig,
                      OnlineMovingAverage,
                      TrainingConfig,
                      training_loop,
                      move_to_device)
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
from torchvision.io import decode_image
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk import LabelStudio, Client
from PIL import Image



#%%
MAIN_PROJECT_ID = 1

VAL_PROJECT_ID = 2
VAL_PROJECT_API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDk2NTE4MiwiaWF0IjoxNzYzNzY1MTgyLCJqdGkiOiIyMjc1ZDhjZjE5MGU0Y2M0YmMzYWJiN2VkYjRhMDEyMSIsInVzZXJfaWQiOjF9.pAMcDVKI7yCDvkYvP6mxJtoCN8GCeOAPGzd_i2fb-tc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
NUM_STEPS = 1000
BATCH_SIZE = 100
NUM_EPOCHS = NUM_STEPS//BATCH_SIZE + 1
FREQ = 200



TRAIN_CONFIG = TrainingConfig()


CONFIG = LoggingConfig(project_dir='/data/model/exp/object_detection',
                        exp_name=f"VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000")



logger = logging.getLogger(__name__)
CONFIG.monitor_metric = "avg_loss"
CONFIG.monitor_mode = "min"
CONFIG.save_freq = FREQ
CONFIG.val_epoch_freq = FREQ
CONFIG.log_loss_freq = 5
CONFIG.log_image_freq = 200

# model_kwargs= dict(
#         weights=None,
#         progress=True,
#         num_classes = 21,
#         weights_backbone= MobileNet_V3_Large_Weights.DEFAULT,
#         trainable_backbone_layers=1
#     )

# MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(**model_kwargs)

MODEL = FasterRCNNMobile(score_threshold=0.8)
TRANSFORM = MODEL.transform

state = CONFIG.load_checkpoint()
# if state is not None:
MODEL.model.load_state_dict(state['model_state_dict'])

MODEL.eval()

class LSDetectionDataset(Dataset):
    def __init__(self, raw_data: list[dict], classes: list[str]=VOCDataset.voc_cls, transform: v2=None):
        cls_to_id = {name: i + 1 for i,name in enumerate(classes)}
        self.classes = classes
        self.cls_to_id = cls_to_id
        self.raw_data = raw_data
        # if transform is None:
        #     transform= v2.Compose([
        #         v2.ToImage(),
        #         v2.ToDtype(torch.float32, scale=True),
        #     ])
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
        
        labels = [self.cls_to_id(label) for label in chosen_data["annotation"]["labels"]]
        
        return img, {'boxes':torch.tensor(boxes), 'labels': torch.tensor(labels)}




class FasterRCNN(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL','http://localhost:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
        self.START_TRAINING_EACH_N_UPDATES = 100
        self.set("model_version", f"{self.__class__.__name__}")

    
    def get_local_path(self, url, project_dir=None, ls_host=None, ls_access_token=None, task_id=None, *args, **kwargs):
        print(f"------------------Find local path of {url}" )
        

        if url.startswith("upload") or url.startswith("/upload"):
            url = "/data" + ("" if url.startswith("/") else "/") + url

        is_uploaded_file = url.startswith("/data/upload")

        project_id = url.split("/")[2]
        filename = os.path.basename(url)

        print(f"is_uploaded_file? {is_uploaded_file}")

        # /data/upload/2/7377a21f-000012.jpg
        if is_uploaded_file:
            print("UPLOADED FILE")
            project_id = url.split("/")[-2]
            filename = os.path.basename(url)
            filepath = os.path.join("/label-studio/data/media/upload",
                                    project_id,
                                    filename)
            print(f"file path: {filepath}")
            if os.path.exists(filepath):
                print(f"Uploaded file: Path exists in image_dir: {filepath}")
                return filepath
            else:
                raise FileNotFoundError(f"Uploaded file not found at {filepath}.")
        # "/data/local-files/?d=local_source/000018.jpg"
        else:
            print("LOCAL STORAGE FILE")
            filepath = os.path.join("/label-studio/data", url.split("?d=")[1])
            print(f"file path: {filepath}")
            if os.path.exists(filepath):
                print(
                    f"Local Storage file path exists locally, use it as a local file: {filepath}"
                )

                return filepath
            else:
                raise FileNotFoundError(f"Local storage file not found at {filepath}.")
        


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_CONFIG}
        # Parsed JSON Label config: {self.parsed_label_CONFIG}
        # Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
        image_url = tasks[0]["data"][value]
        local_path = self.get_local_path(image_url)
        image_tensor = TRANSFORM(Image.open(local_path).convert("RGB"))
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
                
                image_url = task["data"][value]
                image_path = self.get_local_path(image_url)

                result = task["annotations"][0]["result"]
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

        # use cache to retrieve the data from the previous fit() runs
        model_version = self.get('model_version')
        print(f'Old model version: {model_version}')

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"skip training: event {event} is not supported")
            return
        
        project_id = (
            data.get("project") or
            data.get("annotation", {}).get("project") or
            data.get("task", {}).get("project")
        )

        logger.debug(f"Project {project_id}")

        ls = Client(url=self.LABEL_STUDIO_HOST, api_key=self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()

        logger.debug(f"Downloaded {len(tasks)} labeled tasks from Label Studio")
        logger.debug(f"First task example format: {tasks[0]}")
        
        # if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
        if event != 'START_TRAINING':
            logger.debug(f"skip training: the number of tasks is not divisible by {self.START_TRAINING_EACH_N_UPDATES}")
            return
        
        train_config = {'device': DEVICE,
                     'dtype': DTYPE,
                     'num_epochs': NUM_EPOCHS,
                     'batch_size': BATCH_SIZE}
        
        TRAIN_CONFIG.update(**train_config)
        logger.debug(f"Training configuration: {TRAIN_CONFIG}")
        CONFIG.log_hyperparameters(train_config, main_key='training_config')

        
        from_name, to_name, value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        raw_dataset = self.parse_ls_detection_tasks(tasks, value=value)

        train_ds = LSDetectionDataset(raw_dataset, classes=VOCDataset.voc_cls, transform=TRANSFORM)
        if len(train_ds) < BATCH_SIZE:
            sampler = data.RandomSampler(train_ds, replacement=True, num_samples=BATCH_SIZE)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              shuffle=shuffle,
                              sampler=sampler,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn)
        
        val_ds = VOCDataset(images_dir='/data/validation_data/JPEGImages',
                            annotation_dir='/data/validation_data/Annotations')
        
        val_loader = DataLoader(val_ds,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn)

        optim_config = OptimizationConfig()
        optimizer = optim_config.get_optimizer(MODEL.model)
        lr_scheduler = optim_config.get_scheduler(optimizer)

        training_loop(MODEL.model, optimizer, lr_scheduler, train_loader, val_loader, TRAIN_CONFIG, CONFIG)
        
        


        # store new data to the cache
        self.set('model_version', 'my_new_model_version')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

