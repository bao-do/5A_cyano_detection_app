#%%
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

def get_fasterrcnn_resnet50_fpn_v2(num_classes, pretrained=True):
    # Load a model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = get_fasterrcnn_resnet50_fpn_v2(num_classes=21, pretrained=True)
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# %%
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")
# %%
for name, param in model.named_parameters():
    if name.startswith("backbone"):
        param.requires_grad = True
    else:
        param.requires_grad = False
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %%
from label_studio_sdk import LabelStudio

base_url = "http://localhost:8080"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MjEyOTQyNiwiaWF0IjoxNzY0OTI5NDI2LCJqdGkiOiIxZmJlNDczNDljNzM0MzkzOTc5OTNkZWMxMmUwNmQzNSIsInVzZXJfaWQiOjF9.AsGLtLl9VehwHK1VeFPHfCOoYuUZ6oEG2DmRRG9uP6E"

ls = LabelStudio(base_url=base_url, api_key=api_key)
tasks = list(ls.tasks.list(project=2))
tasks = list(map(lambda x: list(x), tasks))
# %%
