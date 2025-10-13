#%%
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet50_Weights

# model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights,
#                                    progress=True,
#                                    weights_backbone = ResNet50_Weights,
#                                    trainable_backbone_layers=3)
model = fasterrcnn_resnet50_fpn_v2()
for name, param in model.backbone.named_parameters():
    print((name))
# print(model.state_dict())
for name, param in model.backbone.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False
print("Number of trainable parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))


# %%
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)

# %%
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import draw_bounding_boxes

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
for idx, img in enumerate(x):
    drawn_boxes = draw_bounding_boxes(img, predictions[idx]['boxes'])
    print(drawn_boxes.shape)
    show(drawn_boxes)
    plt.show()
# %%
