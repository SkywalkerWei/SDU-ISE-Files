import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import numpy as np
import cv2
import random

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # ratio, padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class COCODataset(Dataset):
    def __init__(self, annotation_file, img_dir, img_size=640, use_mosaic=True, is_train=True, debug=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.use_mosaic = use_mosaic
        self.is_train = is_train
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
            
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.images}
        self.img_ids = list(self.img_id_to_filename.keys())

        if debug:
            self.img_ids = self.img_ids[:1000] # Use a small subset for debugging

        self.cat_id_to_label = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        if self.is_train and self.use_mosaic and random.random() < 0.5:
            img, labels = self.load_mosaic(index)
        else:
            img, labels = self.load_image(index)
        
        # Letterbox
        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.is_train)
        
        if labels.size > 0:
            labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], ratio[0] * img.shape[1], ratio[1] * img.shape[0], padw=pad[0], padh=pad[1])
            
        # To tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        targets = torch.zeros((len(labels), 6))
        if len(labels):
            targets[:, 1:] = torch.from_numpy(labels)
        
        return torch.from_numpy(img).float() / 255.0, targets

    def load_image(self, index):
        img_id = self.img_ids[index]
        file_name = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.img_dir, file_name)
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        labels = []
        if img_id in self.img_id_to_anns:
            for ann in self.img_id_to_anns[img_id]:
                bbox = ann['bbox'] # [x, y, width, height]
                cat_id = ann['category_id']
                label = self.cat_id_to_label[cat_id]
                
                # Convert to YOLO format (class, x_center, y_center, width, height) normalized
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                width_norm = bbox[2] / w
                height_norm = bbox[3] / h
                labels.append([label, x_center, y_center, width_norm, height_norm])
        return img, np.array(labels)

    def load_mosaic(self, index):
        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center
        indices = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]  # 3 additional image indices

        for i, idx in enumerate(indices):
            img, labels = self.load_image(idx)
            h, w, _ = img.shape

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if labels.size > 0:
                labels[:, 1] = (labels[:, 1] * w + padw) / (s * 2)
                labels[:, 2] = (labels[:, 2] * h + padh) / (s * 2)
                labels[:, 3] *= w / (s * 2)
                labels[:, 4] *= h / (s * 2)
                labels4.append(labels)
        
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 1, out=labels4[:, 1:])

        return img4, labels4

    @staticmethod
    def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
        return y
    
    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0)