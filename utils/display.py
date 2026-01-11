import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

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
                                    # font = font_path,
                                    font_size=10
                                    ).unsqueeze(0),
                                size=(256,256),
                                mode='bilinear',
                                align_corners=False)
        ax[i].imshow(resize_image.squeeze(0).permute(1,2,0).numpy())
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()
    

