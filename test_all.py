#%%
import torch

def parse_ls_detection_tasks(tasks: list[dict], value: str="image" ):
        dataset = []
        for task in tasks:
            
            image_url = task["data"][value]

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
                "image_path": image_url,
                "annotation": {
                    "boxes": boxes,
                    "labels": labels
                }
            })
        return dataset

task1={
  "id": 3,
  "data": {
    "image": "data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg"
  },
  "annotations": [
    {
      "id": 5,
      "result": [
        {
          "original_width": 500,
          "original_height": 375,
          "image_rotation": 0,
          "value": {
            "x": 31.39022216796875,
            "y": 38.658740234375,
            "width": 13.878878784179687,
            "height": 37.2009765625,
            "rotation": 0,
            "rectanglelabels": [
              "person"
            ]
          },
          "id": "e0bbc042-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.9999938011169434
        },
        {
          "original_width": 500,
          "original_height": 375,
          "image_rotation": 0,
          "value": {
            "x": 17.99237060546875,
            "y": 48.25021158854167,
            "width": 35.13955078125,
            "height": 38.360026041666664,
            "rotation": 0,
            "rectanglelabels": [
              "horse"
            ]
          },
          "id": "3a29afc0-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.999207079410553
        },
        {
          "original_width": 500,
          "original_height": 375,
          "image_rotation": 0,
          "value": {
            "x": 55.86973266601562,
            "y": 53.15025634765625,
            "width": 9.72041015625,
            "height": 32.51424560546875,
            "rotation": 0,
            "rectanglelabels": [
              "person"
            ]
          },
          "id": "201a4230-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.9971592426300049
        },
        {
          "original_width": 500,
          "original_height": 375,
          "image_rotation": 0,
          "value": {
            "x": 50.966061401367185,
            "y": 51.85169677734375,
            "width": 8.638021850585938,
            "height": 34.68187255859375,
            "rotation": 0,
            "rectanglelabels": [
              "person"
            ]
          },
          "id": "5ab66ace-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.990407407283783
        }
      ],
      "created_username": " baolqd123@gmail.com, 1",
      "created_ago": "0 minutes",
      "completed_by": {
        "id": 1,
        "first_name": "",
        "last_name": "",
        "avatar": None,
        "email": "baolqd123@gmail.com",
        "initials": "ba"
      },
      "was_cancelled": False,
      "ground_truth": False,
      "created_at": "2025-11-05T08:23:07.103031Z",
      "updated_at": "2025-11-05T08:23:07.103051Z",
      "draft_created_at": None,
      "lead_time": 4.94,
      "import_id": None,
      "last_action": None,
      "bulk_created": False,
      "task": 3,
      "project": 1,
      "updated_by": 1,
      "parent_prediction": 63,
      "parent_annotation": None,
      "last_created_by": None
    }
  ],
  "predictions": [
    {
      "id": 63,
      "result": [
        {
          "from_name": "label",
          "id": "e0bbc042-",
          "score": 0.9999938011169434,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 37.2009765625,
            "rectanglelabels": [
              "person"
            ],
            "rotation": 0,
            "width": 13.878878784179687,
            "x": 31.39022216796875,
            "y": 38.658740234375
          }
        },
        {
          "from_name": "label",
          "id": "3a29afc0-",
          "score": 0.999207079410553,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 38.360026041666664,
            "rectanglelabels": [
              "horse"
            ],
            "rotation": 0,
            "width": 35.13955078125,
            "x": 17.99237060546875,
            "y": 48.25021158854167
          }
        },
        {
          "from_name": "label",
          "id": "201a4230-",
          "score": 0.9971592426300049,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 32.51424560546875,
            "rectanglelabels": [
              "person"
            ],
            "rotation": 0,
            "width": 9.72041015625,
            "x": 55.86973266601562,
            "y": 53.15025634765625
          }
        },
        {
          "from_name": "label",
          "id": "5ab66ace-",
          "score": 0.990407407283783,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 34.68187255859375,
            "rectanglelabels": [
              "person"
            ],
            "rotation": 0,
            "width": 8.638021850585938,
            "x": 50.966061401367185,
            "y": 51.85169677734375
          }
        }
      ],
      "model_version": "FasterRCNN-v0.0.1",
      "created_ago": "22 hours, 56 minutes",
      "score": 0.996691882610321,
      "cluster": None,
      "neighbors": None,
      "mislabeling": 0,
      "created_at": "2025-11-04T09:27:07.481475Z",
      "updated_at": "2025-11-04T09:27:07.481501Z",
      "model": None,
      "model_run": None,
      "task": 3,
      "project": 1
    }
  ]
}
task2={
  "id": 6,
  "data": {
    "image": "data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000007.jpg"
  },
  "annotations": [
    {
      "id": 4,
      "result": [
        {
          "original_width": 480,
          "original_height": 364,
          "image_rotation": 0,
          "value": {
            "x": 37.25006103515625,
            "y": 17.40701381976788,
            "width": 20.75921376546224,
            "height": 36.806189358889405,
            "rotation": 0,
            "rectanglelabels": [
              "person"
            ]
          },
          "id": "0b1fa6a0-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.9999653100967407
        },
        {
          "original_width": 480,
          "original_height": 364,
          "image_rotation": 0,
          "value": {
            "x": 20.472591718037922,
            "y": 20.85869548084972,
            "width": 64.2207384109497,
            "height": 69.54161465822996,
            "rotation": 0,
            "rectanglelabels": [
              "horse"
            ]
          },
          "id": "42a3a987-",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "prediction",
          "score": 0.9997016787528992
        }
      ],
      "created_username": " baolqd123@gmail.com, 1",
      "created_ago": "19 hours, 53 minutes",
      "completed_by": {
        "id": 1,
        "first_name": "",
        "last_name": "",
        "avatar": None,
        "email": "baolqd123@gmail.com",
        "initials": "ba"
      },
      "was_cancelled": False,
      "ground_truth": False,
      "created_at": "2025-11-04T12:24:39.420188Z",
      "updated_at": "2025-11-04T12:24:39.420203Z",
      "draft_created_at": None,
      "lead_time": 4.886,
      "import_id": None,
      "last_action": None,
      "bulk_created": False,
      "task": 6,
      "project": 1,
      "updated_by": 1,
      "parent_prediction": 66,
      "parent_annotation": None,
      "last_created_by": None
    }
  ],
  "predictions": [
    {
      "id": 66,
      "result": [
        {
          "from_name": "label",
          "id": "0b1fa6a0-",
          "score": 0.9999653100967407,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 36.806189358889405,
            "rectanglelabels": [
              "person"
            ],
            "rotation": 0,
            "width": 20.75921376546224,
            "x": 37.25006103515625,
            "y": 17.40701381976788
          }
        },
        {
          "from_name": "label",
          "id": "42a3a987-",
          "score": 0.9997016787528992,
          "source": "$image",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "height": 69.54161465822996,
            "rectanglelabels": [
              "horse"
            ],
            "rotation": 0,
            "width": 64.2207384109497,
            "x": 20.472591718037922,
            "y": 20.85869548084972
          }
        }
      ],
      "model_version": "FasterRCNN-v0.0.1",
      "created_ago": "22 hours, 50 minutes",
      "score": 0.99983349442482,
      "cluster": None,
      "neighbors": None,
      "mislabeling": 0,
      "created_at": "2025-11-04T09:27:16.309871Z",
      "updated_at": "2025-11-04T09:27:16.309901Z",
      "model": None,
      "model_run": None,
      "task": 6,
      "project": 1
    }
  ]
}
tasks=[task1, task2]

#%%
dataset = parse_ls_detection_tasks(tasks)
# %%
from torchvision.transforms import v2
from dataset import VOCDataset
from torch.utils.data import Dataset
from torchvision import tv_tensors
from PIL import Image


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomGrayscale(p=0.5)
])

class LSDetectionDataset(Dataset):
    def __init__(self, raw_data: list[dict], classes: list[str]=VOCDataset.voc_cls, transform: v2=None):
        super().__init__()
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
        img = v2.ToTensor()(Image.open(img_path).convert("RGB"))
        boxes = chosen_data["annotation"]["boxes"]
        boxes = tv_tensors.BoundingBoxes(boxes,
                                         format="XYXY",
                                         canvas_size=img.shape[-2:])
        
        labels = torch.tensor([self.cls_to_id[label] for label in chosen_data["annotation"]["labels"]])
        if self.transform is not None:
          img, boxes, labels = self.transform(img, boxes, labels)
        
        return img, {'boxes':boxes, 'labels': labels}
#%%
dataset = parse_ls_detection_tasks(tasks)
ls_ds = LSDetectionDataset(raw_data=dataset, transform=transform)

img1, pred1 = ls_ds[0]
img2, pred2 = ls_ds[1]
#%%
imgs = [img1, img2]
preds = [pred1, pred2]

from utils import images_with_bboxes
images_with_bboxes(imgs, preds, max_pixel=1.0, label_str=VOCDataset.voc_cls)

# %%
save_dir = 'model_backend/label-studio-ml-backend/my_ml_backend/data/validation_data'
target_image_dir = os.path.join(save_dir, 'JPEGImages')
target_annot_dir = os.path.join(save_dir, 'Annotations')
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_annot_dir, exist_ok=True)

source_image_dir = 'data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
source_annot_dir = 'data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations'

size = 200
import os
import shutil
os.makedirs(save_dir, exist_ok=True)
for i in range(size):
    img_filename = os.listdir(source_image_dir)[i]
    annot_filename = img_filename.replace('.jpg', '.xml')
    shutil.copy(os.path.join(source_image_dir, img_filename),
                os.path.join(target_image_dir, img_filename))
    shutil.copy(os.path.join(source_annot_dir, annot_filename),
                os.path.join(target_annot_dir, annot_filename))

# %%
from label_studio_sdk import LabelStudio, Client
MAIN_PROJECT_ID = 1


LABEL_STUDIO_HOST = 'http://0.0.0.0:8080'
LABEL_STUDIO_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDk2NTE4MiwiaWF0IjoxNzYzNzY1MTgyLCJqdGkiOiIyMjc1ZDhjZjE5MGU0Y2M0YmMzYWJiN2VkYjRhMDEyMSIsInVzZXJfaWQiOjF9.pAMcDVKI7yCDvkYvP6mxJtoCN8GCeOAPGzd_i2fb-tc"
VAL_PROJECT_ID = 2

ls = LabelStudio(base_url=LABEL_STUDIO_HOST, api_key=LABEL_STUDIO_API_KEY)
# ls = Client(url=LABEL_STUDIO_HOST, api_key=LABEL_STUDIO_API_KEY)
main_project = ls.projects.get(id=MAIN_PROJECT_ID)
# tasks_train = main_project.tasks



# %%
from utils import make_json_safe
# %%
