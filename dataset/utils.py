import torch
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2
from torchvision.io import decode_image
import pandas as pd
from PIL import Image
import xml.etree.cElementTree as ET
import numpy as np
import os
import warnings

def create_dir_df(img_dir: str, annot_dir: str, size: int=None):
    """
    Create a DataFrame that maps images to their corresponding annotation files.

    This function assumes that:
        - Images are stored as `.jpg` files inside `img_dir`.
        - Annotations are stored as `.xml` files inside `annot_dir`.
        - Each annotation file has the same base filename as its corresponding image.

    Parameters
    ----------
    img_dir : str
        Path to the directory containing image files (`.jpg`).
    annot_dir : str
        Path to the directory containing annotation files (`.xml`).

    Returns
    -------
    pandas.DataFrame with following columns: `image`, `annotation`, `image_path`, `annotation_path`
    """
    try:
        imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    except FileNotFoundError:
        print("There is no such directory")
        imgs = None
    

    if (imgs is None) or len(imgs) == 0:
        warnings.warn("The respository does not cointain any images.")
        return None
    else:
        annots = [f.replace('.jpg', '.xml') for f in imgs]
        
        if size is not None:
            imgs = imgs[:size]
            annots = annots[:size]
        img_paths = [os.path.join(img_dir,f) for f in imgs]
        annot_paths = [os.path.join(annot_dir,f) for f in annots]

        df = pd.DataFrame({
            'image': imgs,
            'annotatation': annots,
            'image_path': img_paths,
            'annotation_path': annot_paths
        }) 
        return df

def parse_annotation(df: pd.DataFrame):
    """
    Extract the objects and size of images in `df`.
    Return a bounding boxes dataframe with following columns `class`, `box`, `width`, `height`, and a torch.tensor containig size of images.
    """
    sizes = []
    all_objects = []

    for xml_path in df['annotation_path']:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')

            if size is not None:
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                sizes.append(torch.tensor([h,w]))
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                all_objects.append({
                    'class': name,
                    'box': (xmin, xmax, ymin, ymax),
                    'width': xmax-xmin,
                    'height':ymax-ymin
                })
        except Exception as e:
            print(f"Error at {xml_path}: e")
    return all_objects, torch.stack(sizes)

def collate_fn(batch):
    # batch is a list of tuples (img, target)
    return tuple(zip(*batch))  # returns (images, targets) as tuples

class VOCDataset(torch.utils.data.Dataset):
    voc_cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# aeroplane
# bicycle
# bird
# boat
# bottle
# bus
# car
# cat
# chair
# cow
# diningtable
# dog
# horse
# motorbike
# person
# pottedplant
# sheep
# sofa
# train
# tvmonitor
    

    cls_to_id = {name: i+1 for i, name in enumerate(voc_cls)}

    pixel_max = 255.0

    to_tensor = transforms.ToTensor()

    def __init__(self, images_dir: str, annotation_dir: str, transform: transforms=None):
        super().__init__()
        self.df = create_dir_df(images_dir, annotation_dir)
        # if transform is None:
        #     transform = v2.Compose([
        #         v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
        #     ])
        self.transform = transform

    def __len__(self):
        if self.df is None:
            return 0
        else:
            return len(self.df)
    
    def __getitem__(self, index: int):
        img_path = self.df.loc[index, 'image_path']
        img = decode_image(img_path)

        annot_path = self.df.loc[index, 'annotation_path']
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            cls_id = self.cls_to_id[cls_name]

            xml_box = obj.find('bndbox')
            xmin = int(xml_box.find('xmin').text)
            ymin = int(xml_box.find('ymin').text)
            xmax = int(xml_box.find('xmax').text)
            ymax = int(xml_box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])  # pixel coords
            
            labels.append(cls_id)


        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:])
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)

        return img, {'boxes': boxes, 'labels':torch.tensor(labels)}
    