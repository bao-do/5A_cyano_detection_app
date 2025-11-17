#%%
from label_studio_sdk import LabelStudio
import numpy as np
from PIL import Image
from io import BytesIO
import uuid
import requests
import json

# Connect to your local Label Studio instance
LS_URL="http://localhost:8080"
API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDUxNzQyNywiaWF0IjoxNzYzMzE3NDI3LCJqdGkiOiI4ZTE0MWI1ZjAxZjA0NTE4ODcxMjgxZDE1YjFlYmJiNSIsInVzZXJfaWQiOjF9.398QTAIU5nMK21YiHxbASw5R9_MTtSCM9Bqe1IoTrDU"
PROJECT_ID = 1
ls = LabelStudio(
    base_url=LS_URL,
    api_key=API_KEY
)
# # projects = ls.projects.list()
# # print(projects)



# task = ls.tasks.create(
#     project=1,
#     data={
#         "image": "/data/local-files?d=VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000002.jpg"
#         # "image": "/data/VOC?d=/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000002.jpg"
#     }
# )
# print(task)
#%%
pil_image = Image.open("/home/bao/School/5A/research_project/5A_cyano_detection_app/data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000131.jpg")
img_name = "test_image5.png"
pil_image.save(f"./../data/ls_data/source_storage/{PROJECT_ID}/{img_name}")
# ls.tasks.create()
# %%
task = ls.tasks.create(
    project=PROJECT_ID,
    data={
        "image": f"/data/local-files/?d=source_storage/{PROJECT_ID}/{img_name}"
    }
)
print(task)
# %%
task = ls.tasks.create(
    project=PROJECT_ID,
    data={
        "image": f"/data/local-files/?d=test_folder/{img_name}"
    }
)
print(task)
# %%
ls.predictions.create(
            model_version="user_correction",
            result=[
                {
                    "from_name": "label",
                    "source": "$image",
                    # "original_height": 300,
                    # "original_width": 500,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    'score': float(0.9),
                    "value": {
                        "height": 60,
                        "rotation": 0,
                        "rectanglelabels": ["car"],
                        "width": 50,
                        "x": 20,
                        "y": 30,
                    },
                }
            ],
            score=0.95,
            task=8,
        )
# %%


# {
#   "id": 8,
#   "data": {
#     "image": "/data/local-files/?d=test_folder/test_image5.png"
#   },
#   "annotations": [
#     {
#       "id": 2,
#       "result": [
#         {
#           "original_width": 500,
#           "original_height": 333,
#           "image_rotation": 0,
#           "value": {
#             "x": 0,
#             "y": 25.643258725965488,
#             "width": 63.90977443609023,
#             "height": 54.350806230505476,
#             "rotation": 0,
#             "rectanglelabels": [
#               "Car"
#             ]
#           },
#           "id": "tULOWCGSiz",
#           "from_name": "label",
#           "to_name": "image",
#           "type": "rectanglelabels",
#           "origin": "manual"
#         }
#       ],
#       "created_username": " baolqd123@gmail.com, 1",
#       "created_ago": "0 minutes",
#       "completed_by": {
#         "id": 1,
#         "first_name": "",
#         "last_name": "",
#         "avatar": null,
#         "email": "baolqd123@gmail.com",
#         "initials": "ba"
#       },
#       "was_cancelled": false,
#       "ground_truth": false,
#       "created_at": "2025-11-16T17:59:25.548367Z",
#       "updated_at": "2025-11-16T17:59:25.548382Z",
#       "draft_created_at": null,
#       "lead_time": 19.05,
#       "import_id": null,
#       "last_action": null,
#       "bulk_created": false,
#       "task": 8,
#       "project": 1,
#       "updated_by": 1,
#       "parent_prediction": null,
#       "parent_annotation": null,
#       "last_created_by": null
#     }
#   ],
#   "predictions": []
# }