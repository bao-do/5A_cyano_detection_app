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


# Load checkpoint function
MONITOR_METRIC = os.getenv("MONITOR_METRIC","val_avg_map")
MONITOR_MODE = os.getenv("MONITOR_MODE", "max")
SAVE_WEIGHTS_DIR = os.getenv("SAVE_WEIGHTS_DIR", "exp/object_detection/")
EXP_NAME = os.getenv("EXP_NAME", "VOC_fasterrcnn_resnet50_fpn_v2_3000")


logger_config = LoggingConfig(project_dir=SAVE_WEIGHTS_DIR,
                              exp_name=EXP_NAME,
                              monitor_metric=MONITOR_METRIC,
                              monitor_mode=MONITOR_MODE)



############# Define Model ##############

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.8))
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.5))

model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
state = logger_config.load_best_checkpoint()
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


def _run_training(cmd: list[str]):
    """Run the training process in a background thread."""
    print(f"[train] Starting training with command: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, check=True)
        print(f"[train] Training finished with return code {proc.returncode}", flush=True)
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



############### api ####################
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

@app.route("/predict", methods=["POST"])
def predict():
    # if there is no payload, meaning the model has not been trained yet
    if not os.path.exists(PAYLOAD_DIR):
        return jsonify({"error": "Model not trained yet."}), 400
    with open(PAYLOAD_DIR, "r") as f:
        payload = json.load(f)
    class_str = payload.get("class_str", "")
    file = request.files["file"]
    img_bytes = file.read()
    iou_threshold = float(request.form.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    score_threshold = float(request.form.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.predict(img_tensor, iou_threshold, score_threshold)[0]

    return jsonify({
        "boxes": outputs['boxes'].cpu().numpy().tolist(),
        "labels": [label for label in outputs['labels'].cpu().numpy().tolist()],
        "scores": outputs['scores'].cpu().numpy().tolist(),
        "classes": [voc_cls[label-1] for label in outputs['labels'].cpu().numpy().tolist()]
    })


@app.route("/train", methods=["POST"])
def train():
    train_payload = request.get_json(silent=True) or {}
    # save training payload to temp file for debugging
    with open(PAYLOAD_DIR, "w") as f:
        json.dump(train_payload, f)

    ok, msg = start_training()
    status = 202 if ok else 409
    return jsonify({"status": "ok" if ok else "busy", "message": msg}), status

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075, debug=True)