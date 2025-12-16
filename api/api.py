import sys, os
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(abs_path)
from NNModels import FasterRCNNMobile
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import glob
import re

# VOC classes and mapping
voc_cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
cls_to_id = {name: i+1 for i, name in enumerate(voc_cls)}

# Load checkpoint function
DEFAULT_MONITOR_METRIC = os.getenv("DEFAULT_MONITOR_METRIC","val_avg_map")
DEFAULT_MONITOR_MODE = os.getenv("DEFAULT_MONITOR_MODE", "max")
DEFAULT_CHECKPOINT_DIR = os.getenv("DEFAULT_CHECKPOINT_DIR",
                                   os.path.join(abs_path,
                                       "exp/object_detection/VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000/checkpoints"))
print(DEFAULT_CHECKPOINT_DIR)
def load_checkpoint():
    checkpoints = glob.glob(os.path.join(DEFAULT_CHECKPOINT_DIR, f"*{DEFAULT_MONITOR_METRIC}_*.pth"))
    if len(checkpoints) == 0:
        print("No checkpoints found.")
        return None
    
    def get_metric_value(checkpoint_path):
        match = re.search(rf"{DEFAULT_MONITOR_METRIC}_([0-9]+(?:\.[0-9]+)?)", checkpoint_path)
        if match:
            return float(match.group(1))
        else:
            return float("inf") if DEFAULT_MONITOR_MODE=="min" else float("-inf")
        
    if DEFAULT_MONITOR_MODE == "min":
        checkpoint_path = min(checkpoints, key=get_metric_value)
    else:
        checkpoint_path = max(checkpoints, key=get_metric_value)
            
    state = torch.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return state


############# Define Model ##############

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.8))
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.5))

MODEL = FasterRCNNMobile(score_threshold=DEFAULT_SCORE_THRESHOLD,
                         iou_threshold=DEFAULT_IOU_THRESHOLD,
                         device=DEVICE)

# Load checkpoint
state = load_checkpoint()
MODEL.model.load_state_dict(state['model_state_dict'])


MODEL.eval()


# PIL image to tensor transform
transform = transforms.ToTensor()



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

    print(type(outputs['boxes']), type(outputs['labels']), type(outputs['scores']))
    return jsonify({
        "boxes": outputs['boxes'].cpu().numpy().tolist(),
        "labels": [label for label in outputs['labels'].cpu().numpy().tolist()],
        "scores": outputs['scores'].cpu().numpy().tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5075)