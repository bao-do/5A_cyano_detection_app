import sys, os
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(abs_path)
from NNModels import FasterRcnnPredictor
from utils import LoggingConfig
from training import training_loop
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import glob
import re
import threading
import subprocess
import shlex
import json
from label_studio_sdk import LabelStudio
import time
import httpx


LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "http://label-studio:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
MAIN_PROJECT_ID = int(os.getenv("MAIN_PROJECT_ID", 1))

MONITOR_METRIC = os.getenv("MONITOR_METRIC","val_avg_map")
MONITOR_MODE = os.getenv("MONITOR_MODE", "max")
SAVE_WEIGHTS_DIR = os.getenv("SAVE_WEIGHTS_DIR", "exp/object_detection/")
EXP_NAME = os.getenv("EXP_NAME", "VOC_fasterrcnn_resnet50_fpn_v2")

ls = LabelStudio(base_url= LABEL_STUDIO_HOST, api_key=LABEL_STUDIO_API_KEY)


max_retries = 10
for i in range(max_retries):
    try:
        print(f"Attempting to connect to Label Studio (Try {i+1}/{max_retries})...")
        class_str = ls.projects.get(MAIN_PROJECT_ID).parsed_label_config['label']['labels']
        print("Connected successfully!")
        break
        
    except (httpx.ConnectError, Exception) as e:
        # If connection fails, wait and try again
        print(f"Connection failed: {e}")
        if i < max_retries - 1:
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
        else:
            print("Max retries reached. Exiting.")
            raise e # Crash if it still fails after 50 seconds
        
# class_str = ls.projects.get(MAIN_PROJECT_ID).parsed_label_config['label']['labels']

logger_config = LoggingConfig(project_dir=SAVE_WEIGHTS_DIR,
                              exp_name=EXP_NAME,
                              monitor_metric=MONITOR_METRIC,
                              monitor_mode=MONITOR_MODE)



############# Define Model ##############

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.9))
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.5))

model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, max(len(class_str),1)+1)
state = logger_config.load_best_checkpoint()

if state is not None:
    model.load_state_dict(state['model_state_dict'])

MODEL = FasterRcnnPredictor(model=model,
                         score_threshold=DEFAULT_SCORE_THRESHOLD,
                         iou_threshold=DEFAULT_IOU_THRESHOLD,
                         device=DEVICE)

MODEL.eval()

# PIL image to tensor transform
transform = transforms.ToTensor()

# payload file path
PAYLOAD_DIR = os.getenv("PAYLOAD_DIR", "/tmp/train_payloads.json")

# ---------------- Training trigger (background) ----------------
# We run training inside the API container (has torch installed).
_train_lock = threading.Lock()
_train_thread = None

# Model reload lock to prevent concurrent reloads
_model_reload_lock = threading.Lock()


def load_model_and_classes():
    """Load the best model checkpoint and update class list."""
    global MODEL, class_str
    
    with _model_reload_lock:
        try:
            print("[model] Reloading model and classes...")
            
            # Load new class list from Label Studio
            with open(PAYLOAD_DIR, "r") as f:
                payload = json.load(f)
            new_class_str = payload.get("class_str", [])
            
            # Load model with new number of classes
            new_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            in_features = new_model.roi_heads.box_predictor.cls_score.in_features
            new_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, max(len(new_class_str), 1) + 1)
            
            # Load best checkpoint
            state = logger_config.load_best_checkpoint()
            if state is not None:
                new_model.load_state_dict(state['model_state_dict'])
                print(f"[model] Loaded checkpoint from epoch {state.get('epoch', 'unknown')}")
            
            # Update global MODEL
            MODEL.model = new_model
            MODEL.eval()
            
            # Update class list
            class_str = new_class_str
            
            print(f"[model] Model and classes reloaded successfully. Classes: {class_str}")
            return True
        except Exception as e:
            print(f"[model] Error reloading model: {e}", flush=True)
            return False


def _run_training(cmd: list[str]):
    """Run the training process in a background thread."""
    print(f"[train] Starting training with command: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, check=True)
        print(f"[train] Training finished with return code {proc.returncode}", flush=True)
        
        # Reload model and classes after successful training
        print("[train] Reloading model with new weights...", flush=True)
        if load_model_and_classes():
            print("[train] Model reload successful!", flush=True)
        else:
            print("[train] Warning: Model reload failed, using previous model", flush=True)
            
    except subprocess.CalledProcessError as e:
        print(f"[train] Training failed: {e}", flush=True)
    except Exception as e:
        print(f"[train] Unexpected training error: {e}", flush=True)
    finally:
        global _train_thread
        with _train_lock:
            _train_thread = None


def start_training():
    """Start a training job if none is running; returns status and message."""
    global _train_thread
    with _train_lock:
        if _train_thread is not None and _train_thread.is_alive():
            return False, "Training already running"

        cmd = [
            "python", "training/train_from_ls_data.py"
        ]

        _train_thread = threading.Thread(target=_run_training, args=(cmd,), daemon=True)
        _train_thread.start()
        return True, "Training started"


def parse_ls_detection_tasks(tasks: list[dict], value: str="image" ):
    dataset = []
    for task in tasks:
        if len(task.annotations) == 0 or len(task.annotations[0]["result"]) == 0:
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
        

############### api ####################
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    iou_threshold = float(request.form.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    score_threshold = float(request.form.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.predict(img_tensor, iou_threshold, score_threshold)[0]
    

    return jsonify({
        "boxes": outputs['boxes'].numpy().tolist(),
        "labels": [label for label in outputs['labels'].numpy().tolist()],
        "scores": outputs['scores'].numpy().tolist(),
        "classes": [class_str[label-1] for label in outputs['labels'].numpy().tolist()]
    })


@app.route("/train", methods=["POST"])
def train():

    # Load data
    ls = LabelStudio(base_url= LABEL_STUDIO_HOST, api_key=LABEL_STUDIO_API_KEY)

    print(f"Worker: Fetching tasks for Project {MAIN_PROJECT_ID}...")
    
    tasks_train = list(ls.tasks.list(project=MAIN_PROJECT_ID))

    raw_dataset_train = parse_ls_detection_tasks(tasks_train)
    class_str = ls.projects.get(MAIN_PROJECT_ID).parsed_label_config['label']['labels']

    train_payload = {
        "raw_train_data": raw_dataset_train,
        "class_str": class_str
    }
    # save training payload to temp file for debugging
    with open(PAYLOAD_DIR, "w") as f:
        json.dump(train_payload, f)

    ok, msg = start_training()
    status = 202 if ok else 409
    return jsonify({"status": "ok" if ok else "busy", "message": msg}), status


@app.route("/reload-model", methods=["POST"])
def reload_model():
    """Manually reload the model and classes (useful for debugging)."""
    success = load_model_and_classes()
    return jsonify({
        "status": "success" if success else "error",
        "message": "Model and classes reloaded" if success else "Failed to reload model"
    }), 200 if success else 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075, debug=True)