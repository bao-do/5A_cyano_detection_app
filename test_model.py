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
#%%
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models import MobileNet_V3_Large_Weights

ck_dir = os.path.join(abs_path, 'exp/object_detection/VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000/checkpoints/epoch_1199_avg_loss_0.2755.pth')
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


print("Number of trainable parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))


#%%
transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
])
to_pil = transforms.ToPILImage()

#%%

test_img_path1 ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
test_img_path2 ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000007.jpg'

image1 = decode_image(test_img_path1).to(device)
image2 = decode_image(test_img_path2).to(device)
images = [image1, image2]
images_transformed = [transform(image) for image in images]
targets_pred = model(images_transformed)

print(type(targets_pred['boxes']), type(targets_pred['labels']), type(targets_pred['scores']))
#%%
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

#%%
from NNModels import FasterRCNNMobile
from training import LoggingConfig

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
ckpath = '/home/qbao/School/5A/research_project/5A_cyano_detection_app/exp/object_detection/VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000/checkpoints/epoch_1199_avg_loss_0.2755.pth'
# model = FasterRCNNMobile(ckpath=ckpath, device=device)
model = FasterRCNNMobile(device=device)
logger = LoggingConfig(project_dir="exp/object_detection", exp_name='VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000')
logger.monitor_metric = "avg_loss"
logger.monitor_mode = "min"

state = logger.load_checkpoint()
model.model.load_state_dict(state_dict=state['model_state_dict'])

test_img_path1 ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
test_img_path2 ='/home/qbao/Work/self_learning/deep_learning/object_dectection/yolo/data/pascal_voc/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/000007.jpg'

image1 = decode_image(test_img_path1).to(device)
image2 = decode_image(test_img_path2).to(device)

images = [image1, image2]
targets_pred = model.predict(images)

# %%
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = fm.findfont("DejaVuSans")  # common default on most systems
def images_with_bboxes(images: list[torch.Tensor],
                       predictions: list[dict],
                       label_str: list[str],
                       max_pixel: int=255,
                       image_size = 3,
                       ncol: int=None):

    if ncol is None:
        ncol = len(images)
        nrow = 1
    nrow = int(len(images) *1.0/ ncol)
    fig, ax = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*image_size, nrow*image_size))
    ax = ax.flatten()

    for i in range(len(images)):
        if label_str is not None:
            labels_idx = torch.tensor(predictions[i]['labels'])
            labels_idx = (labels_idx-1).tolist()
            labels = np.array(label_str)[labels_idx]
            
            
        resize_image = F.interpolate(
                                draw_bounding_boxes(
                                    images[i].detach().cpu()/max_pixel,
                                    predictions[i]['boxes'].detach().cpu(),
                                    labels=labels if labels is not None else None,
                                    width=3,
                                    colors='red',
                                    font = font_path,
                                    font_size=10
                                    ).unsqueeze(0),
                                size=(256,256),
                                mode='bilinear',
                                align_corners=False)
        ax[i].imshow(resize_image.squeeze(0).permute(1,2,0).numpy())
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

# %%
from dataset import VOCDataset
images_train_dir ='data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
annotations_train_dir='data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
images_val_dir='data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
annotations_val_dir='data/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
train_dataset = VOCDataset(images_val_dir, annotations_val_dir)

#%%
indices = np.random.randint(0,100,(2,))
train_img1, pred1 = train_dataset[indices[0]]
train_img2, pred2 = train_dataset[indices[1]]

train_images = [train_img1, train_img2]
pred_gt = [pred1, pred2]
pred = model.predict([image.to(device) for image in train_images])

images_with_bboxes(train_images, pred, max_pixel=1, label_str=VOCDataset.voc_cls, image_size=5)
images_with_bboxes(train_images, pred_gt, max_pixel=1, label_str=VOCDataset.voc_cls, image_size=5)

# %%
