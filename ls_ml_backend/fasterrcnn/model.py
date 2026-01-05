#%%
import os, sys

from uuid import uuid4
import logging
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image
import requests
import io
import json


#%%


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
        


# ----------- MODEL PREDICTION AND TRAINING SETUP  ----------------------
MAIN_PROJECT_ID = int(os.getenv("MAIN_PROJECT_ID", 1))
VAL_PROJECT_ID = int(os.getenv("VAL_PROJECT_ID", 2))
API_URL = os.getenv("API_URL", "http://model-api:5075/predict")
TRAIN_API_URL = os.getenv("TRAIN_API_URL", API_URL.replace("/predict", "/train"))



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
        # self.device = DEVICE

    
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
        
        img_pil = Image.open(local_path).convert("RGB")
        img_binary = io.BytesIO()
        img_pil.save(img_binary, format="PNG")

        response = requests.post(API_URL,
                               files={"file": img_binary.getvalue()})
        targets_pred = response.json()
        results = []
        original_w, original_h = img_pil.size
        print(f"Image of size: {original_h}x{original_w}")
        total_score = 0
                
        for i in range(len(targets_pred['boxes'])):
            x1, y1, x2, y2 = targets_pred['boxes'][i]
            cls = targets_pred['classes'][i]
            score = float(targets_pred['scores'][i])
            label_id = str(uuid4())[:9]
            results.append({
                'id':label_id,
                'from_name': from_name,
                'to_name': to_name,
                'source': '$image',
                'type': 'rectanglelabels',
                'original_width': original_w,
                'original_height': original_h,
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
            'result': results,
            'score': total_score,
            'model_version': self.get('model_version')
        }]
        
        return ModelResponse(predictions=predictions)

    

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        # Minimal trigger: only respond to explicit START_TRAINING events
        if event != 'START_TRAINING':
            return


        project = (
            data.get("project") or
            data.get("annotation", {}).get("project") or
            data.get("task", {}).get("project")
        )

        project_id = project['id']
        if (project_id != MAIN_PROJECT_ID):
            logger.info("Skip training: fit method is not supported for this project")
            return
        
        try:
            resp = requests.post(TRAIN_API_URL, timeout=5)
            print(f"[fit] Triggered training at {TRAIN_API_URL}, status={resp.status_code}, response={resp.json()}")
        except Exception as e:
            print(f"[fit] Failed to trigger training: {e}")
        return


