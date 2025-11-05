import torch
        
def iou(boxes_pred: torch.Tensor, boxes_labels: torch.Tensor, box_format: str='midpoint') -> torch.Tensor:
    """
    Calculates intersection over union.

    Parameters:
        boxes_pred: Prediction of bounded boxes (*, 4)
        boxes_labels: Ground true of bounded boxes (*, 4)
        box_format: format of bounded boxes, "mindpoint" (x_c,y_c,w,h)"/"corners"(x1,y1,x2,y2)
    Returns:
        Intersection over union (B,)
    """
    assert (boxes_labels.shape[-1] == 4) and (boxes_pred.shape[-1] == 4), "The last dimension should have 4 elements"
    if box_format == 'midpoint':
        b1_x1 = boxes_pred[...,0] - boxes_pred[...,2]/2 # (*,)
        b1_x2 = boxes_pred[...,0] + boxes_pred[...,2]/2 # (*,)
        b1_y1 = boxes_pred[...,1] - boxes_pred[...,3]/2 # (*,)
        b1_y2 = boxes_pred[...,1] + boxes_pred[...,3]/2 # (*,)

        b2_x1 = boxes_labels[...,0] - boxes_labels[...,2]/2 # (*,)
        b2_x2 = boxes_labels[...,0] + boxes_labels[...,2]/2 # (*,)
        b2_y1 = boxes_labels[...,1] - boxes_labels[...,3]/2 # (*,)
        b2_y2 = boxes_labels[...,1] + boxes_labels[...,3]/2 # (*,)
    elif box_format == 'corners':
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes_pred[...,0], boxes_pred[...,1], boxes_pred[...,2], boxes_pred[...,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes_labels[...,0], boxes_labels[...,1], boxes_labels[...,2], boxes_labels[...,3]
    else:
        raise ValueError("box_format should be 'midpoint' or 'corners'")
    
    x1, y1 = torch.max(b1_x1, b2_x1), torch.max(b1_y1, b2_y1) # (*,)
    x2, y2 = torch.min(b1_x2, b2_x2), torch.min(b1_y2, b2_y2) # (*,)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) # (*,)

    b1_area = ((b1_x2-b1_x1)*(b1_y2-b1_y1)).abs() # (*,)
    b2_area = ((b2_x2-b2_x1)*(b2_y2-b2_y1)).abs() # (*,)

    return inter/(b1_area + b2_area - inter + 1e-7) # (*,)

def non_max_suppression(bboxes: torch.Tensor, iou_threshold: float=0.5,
                        prob_threshold: float=0.5, box_format: str="corners") -> torch.Tensor:
    """
    Perform Non Max Suppression on given `bboxes`, note that all the bboxes must predict the same class. 
    Parameters:
        bbox: tensor containing all boxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2], (*, 6 )
        iou_threshold: threshold to decide whether two boxes bound a same object.
        prob_threshold: threshold to decide whether object exist in the bbox
        box_format: 'midpoint' or 'corner'
    Returns:
        bboxes after performing NMS. 
    """
    print("Inside the function")
    assert bboxes.ndim == 2, "The input should be a 2d tensor"
    bboxes = bboxes[bboxes[..., 1] >= prob_threshold]

    _, indices= torch.sort(bboxes[...,1], descending=True) 
    bboxes = bboxes[indices]

    bboxes_after_nms = []

    while bboxes.size(0):
        chosen_box = bboxes[0:1,:]
        bboxes_after_nms.append(chosen_box)

        if bboxes.size(0) == 1:
            break
        
        ious = iou(chosen_box[:,-4:], bboxes[1:,-4:], box_format)
        mask = (bboxes[1:,0] != chosen_box[0,0]) | (ious < iou_threshold)
        bboxes = bboxes[1:][mask]
    
    return torch.cat(bboxes_after_nms, dim=0)



def mean_average_precision(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_threshold: float=0.5, box_format: str='corners', num_classes: int=20):
    """
    Calculate mean average precision
    Parameters:
        pred_boxes: list of lists containing all bboxes with each bboxes specified as [image_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes: list of lists containing all bboxes from groud truth
        iou_threshold: intersection over union threshold
        box_format: 'midpoint' or 'corner'
        num_classes: number of classes
    Returns:
        float: mAP value across all classes 
    """

    avg_precision = []
    eps = 1e-7
    for c in range(num_classes):
        pred_bb_cls = pred_boxes[pred_boxes[...,1] == c]  
        gt_bb_cls = true_boxes[true_boxes[...,1] == c]  
        unique, counts = gt_bb_cls[:, 0].unique(return_counts=True)
        num_bb_per_img = dict(zip(unique.tolist(), counts.tolist()))
        for key, val in num_bb_per_img.items():
            num_bb_per_img[key] = torch.zeros(val)

        _,indices = torch.sort(pred_bb_cls[...,2], descending=True)
        pred_bb_cls = pred_bb_cls[indices]

        TP, FP, num_gt_bb_cls = torch.zeros(pred_bb_cls.shape[0]), torch.zero(pred_bb_cls.shape[0]), gt_bb_cls.shape[0]

        if num_gt_bb_cls != 0:
            for pred_idx, pred_bb in enumerate(pred_bb_cls):
                img_idx = int(pred_bb[0].item())
                gt_bbs = gt_bb_cls[gt_bb_cls[...,0] ==  img_idx]
                iou_best = 0
                iou_score = iou(pred_bb, gt_bbs, box_format)
                iou_best, idx_gt_best = torch.max(iou_score)
                if iou_best >= iou_threshold:
                    if num_bb_per_img[img_idx][int(idx_gt_best.item())] == 0:
                        num_bb_per_img[img_idx][int(idx_gt_best.item())] = 1
                        TP[pred_idx] = 1
                    else:
                        FP[pred_idx] = 1
                else:
                    FP[pred_idx] = 1
            
            TP_cumsum, FP_cumsum = torch.cumsum(TP), torch.cumsum(FP)
            recalls = torch.cat((torch.tensor([0]),
                                 TP_cumsum/(num_gt_bb_cls + eps)))
            precisions = torch.cat((torch.tensor([1]),
                                   TP_cumsum/(TP_cumsum + FP_cumsum +eps)))
            avg_precision.append(torch.trapezoid(precisions, recalls))
    
    return sum(avg_precision)/ (len(avg_precision) + eps)
        


      
def get_bboxes(loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               S: int=7,
               num_classes: int=20,
               iou_threshold: float=0.5,
               prob_threshold: float=0.5,
               box_format: str='midpoint',
               device: str="cuda"):
    
    """
    Extract predicted and ground-truth bounding boxes from a dataset loader
    using a YOLO-style model, converting cell-based predictions into full-image
    coordinates and applying Non-Maximum Suppression (NMS).
    Parameters
    ----------
    loader : Dataloader providing input images and corresponding ground-truth labels.
    model : YOLO-style model.
    S : Number of grid cells the image is divided into along each dimension (default: 7).
    num_classes : Number of object classes in the dataset (default: 20).
    iou_threshold : Minimum Intersection-over-Union (IoU) threshold for NMS (default: 0.5).
    prob_threshold : Confidence threshold to filter out low-confidence boxes (default: 0.5).
    box_format : Format of the bounding boxes — usually "midpoint" (`x, y, w, h`) (default: "midpoint").
    device : Device on which to perform inference (e.g. "cuda" or "cpu") (default: "cuda").
    Returns
    -------
    all_pred_boxes : List of all predicted bounding boxes across the dataset. Each element has the form:
        `[image_idx, class_id, confidence, x, y, w, h]`
        where `(x, y, w, h)` are normalized to image dimensions.
    all_gt_boxes : List of all ground-truth bounding boxes across the dataset. Each element has the same format as above.
    """
    all_pred_boxes = []
    all_gt_boxes = []

    model.eval()
    train_idx = torch.tensor([0])

    for x, label in loader:
        x = x.to(device)
        label = label.to(device)

        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        gt_boxes = convert_cellboxes(label, S, num_classes).flatten(-3,-2) # (B, S*S, 6)
        pred_boxes = convert_cellboxes(predictions, S, num_classes).flatten(-3,-2) # (B, S*S, 6)
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(pred_boxes[idx], iou_threshold, prob_threshold,box_format)
            for nms_box in nms_boxes:
                all_pred_boxes.append(torch.cat((train_idx,nms_box)))
            
            for box in gt_boxes[idx]:
                if box[1] > prob_threshold:
                    all_gt_boxes.append(torch.cat((train_idx, box)))
            train_idx += 1
    
    model.train()
    return all_pred_boxes, all_gt_boxes