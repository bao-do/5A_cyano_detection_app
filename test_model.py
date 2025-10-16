#%%
import sys
from training import move_to_device
from dataset import VOCDataset
import gradio as gr
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes

# Define dataset



# Define device on which model will process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy exemplar database
abs_path = os.path.abspath(os.path.dirname(__file__))


# Define the same model as training
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models import MobileNet_V3_Large_Weights

ck_dir = os.path.join(abs_path, 'exp/fasterrcnn_fpn/checkpoints/epoch_499_avg_loss_0.3784.pth')
state = torch.load(ck_dir)

model_kwargs = dict(
    weights=None,
    progress=True,
    num_classes = 21,
    weights_backbone= MobileNet_V3_Large_Weights.DEFAULT,
    trainable_backbone_layers=1
)
model = fasterrcnn_mobilenet_v3_large_320_fpn(**model_kwargs)
model.load_state_dict(state['model_state_dict'])
model = model.to(device)
model.eval()

transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
])
to_pil = transforms.ToPILImage()

#%%

test_img_path ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
image = decode_image(test_img_path).to(device)
image = transform(image.unsqueeze(0))
# Extract features & proposals
with torch.no_grad():
    images_list, _ = model.transform(image)
    features = model.backbone(images_list.tensors)
    proposals, _ = model.rpn(images_list, features)

    # pick one feature map level (depends on FPN output)
    box_features = model.roi_heads.box_roi_pool(features, proposals, images_list.image_sizes)
    
    # Pass through the box head (two fc layers)
    box_features = model.roi_heads.box_head(box_features)
    
    # Now we can manually get class logits
    class_logits = model.roi_heads.box_predictor.cls_score(box_features)
    box_regression = model.roi_heads.box_predictor.bbox_pred(box_features)

# Compute softmax probabilities
import torch.nn.functional as F
probs = F.softmax(class_logits, dim=1)
print(probs.shape)  # [num_proposals, num_classes]

# %%
from IPython.display import display
test_img_path ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
image = decode_image(test_img_path).to(device)
image = transform(image.unsqueeze(0))
targets_pred = model(image)

model.train()
for i in range(len(targets_pred[0]['boxes'])):

    x1, y1, x2, y2 = targets_pred[0]['boxes'][i].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    zoomed_obj = to_pil(image[0][:, y1:y2, x1:x2])
    size = (x2 - x1)/10.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except:
        font = ImageFont.load_default(size=size)
    label = targets_pred[0]['labels'][i]
    cls = VOCDataset.voc_cls[label]

    score = targets_pred[0]['scores'][i]

    draw = ImageDraw.Draw(zoomed_obj)

    pos = tuple([x / 15.0 for x in zoomed_obj.size])
    draw.text(pos, f"{cls}: {score:.2f}", fill="red", font=font)
    # display(zoomed_obj)