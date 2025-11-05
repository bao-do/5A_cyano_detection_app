import os, sys
sys.path.append("./../")
from utils import iou, non_max_suppression, get_bboxes
from training import LoggingConfig
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import v2
import numpy as np


class FasterRCNNMobile(torch.nn.Module):
    model_kwargs = dict(
                        weights=None,
                        progress=True,
                        num_classes = 21,
                        trainable_backbone_layers=1
                    )
    
    model = fasterrcnn_mobilenet_v3_large_320_fpn(**model_kwargs)

    def __init__(self,ckpath: str=None,
                 transform: v2=None,
                 score_threshold: float=0.5,
                 iou_threshold: float=0.5,
                 device: str="cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model = self.model.to(device)
        if ckpath is not None:
            state = torch.load(ckpath)
            self.model.load_state_dict(state['model_state_dict'])

        if transform is not None:
            self.transform = transform
        else :
            self.transform = v2.Compose([
                                    v2.ToImage(),
                                    v2.ToDtype(torch.float32, scale=True)
                                ])
    def train_forward(self, images, labels, is_transform_needed=False):
        self.model.train()
        if images is torch.Tensor and input.ndim == 3:
            images = images.unsqueeze(0)
        if is_transform_needed:
            images = self.transform(images)
        return self.model(images, labels)

    def forward(self, input):
        self.model.eval()
        if isinstance(input, torch.Tensor) and input.ndim == 3:
            input = input.unsqueeze(0)
        input_transformed = [self.transform(img.to(self.device)) for img in input]
        return self.model(self.transform(input_transformed))
    
    def predict(self, input):
        raw_predictions = self.forward(input)
        predictions = []
        for raw_prediction in raw_predictions:
            score_mask = raw_prediction['scores'] >= self.score_threshold
            boxes = raw_prediction['boxes'][score_mask]
            labels = raw_prediction['labels'][score_mask]
            scores = raw_prediction['scores'][score_mask]

            _, indices = torch.sort(scores, descending=True)
            boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
            
            boxes_after_nms = []
            labels_after_nms = []
            scores_after_nms = []
            while boxes.size(0):
                chosen_box = boxes[0:1,:]
                chosen_label = labels[0]
                boxes_after_nms.append(chosen_box.detach().cpu())
                labels_after_nms.append(chosen_label.detach().cpu())
                scores_after_nms.append(scores[0].detach().cpu())

                if boxes.size(0) == 1:
                    break
                    
                ious = iou(chosen_box, boxes[1:], box_format='corners')
                mask = (labels[1:] != chosen_label) | (ious < self.iou_threshold)
                boxes, labels, scores = boxes[1:][mask], labels[1:][mask], scores[1:][mask]
            
            predictions.append({
                'boxes': torch.cat(boxes_after_nms, dim=0),
                'labels': torch.tensor(labels_after_nms),
                'scores': torch.tensor(scores_after_nms)
            })
        return predictions
            

        


