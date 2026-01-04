import os, sys
sys.path.append("./../")
from utils import iou
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import v2


class FasterRcnnPredictor(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 transform: v2=None,
                 score_threshold: float=0.5,
                 iou_threshold: float=0.5,
                 device: str="cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model = model.to(device)

        if transform is not None:
            self.transform = transform
        else :
            self.transform = v2.Compose([
                                    v2.ToImage(),
                                    v2.ToDtype(torch.float32, scale=True)
                                ])
    def train_forward(self, images, labels):
        self.model.train()
        if images is torch.Tensor and input.ndim == 3:
            images = images.unsqueeze(0)
        return self.model(images, labels)

    def forward(self, input):
        self.model.eval()
        if isinstance(input, torch.Tensor) and input.ndim == 3:
            input = input.unsqueeze(0)
        return self.model(input)
    
    def predict(self, input, iou_threshold=None, score_threshold=None):
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        if score_threshold is None:
            score_threshold = self.score_threshold

        raw_predictions = self.forward(input)
        predictions = []
        for raw_prediction in raw_predictions:
            score_mask = raw_prediction['scores'] >= score_threshold
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
                mask = (labels[1:] != chosen_label) | (ious < iou_threshold)
                boxes, labels, scores = boxes[1:][mask], labels[1:][mask], scores[1:][mask]
            

            predictions.append({
                'boxes': torch.cat(boxes_after_nms, dim=0) if len(boxes_after_nms) != 0 else boxes_after_nms,
                'labels': torch.tensor(labels_after_nms) if len(labels_after_nms) != 0 else labels_after_nms,
                'scores': torch.tensor(scores_after_nms) if len(scores_after_nms) != 0 else scores_after_nms
            })
        return predictions
            

        


