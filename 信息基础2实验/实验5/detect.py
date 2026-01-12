import torch
import cv2
import numpy as np
import yaml
import argparse

from models import Model
from utils import non_max_suppression, xywh2xyxy

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect(weights, source, data, device):
    # Load model
    with open(data, 'r') as f:
        data_dict = yaml.safe_load(f)
    nc = int(data_dict['nc'])
    names = data_dict['names']
    
    model = Model(nc=nc).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    
    # Load image
    img0 = cv2.imread(source) # BGR
    assert img0 is not None, f"Image not found at {source}"
    
    # Preprocess
    img = letterbox(img0, new_shape=640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)
    
    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # Draw boxes
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=[np.random.randint(0, 255) for _ in range(3)], line_thickness=2)
    
    # Save result
    output_path = 'result.jpg'
    cv2.imwrite(output_path, img0)
    print(f"Detection result saved to {output_path}")

# Helper functions from yolov5 utils
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/epoch_50.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images/bus.jpg', help='source')
    parser.add_argument('--data', type=str, default='voc.yaml', help='voc.yaml path')
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detect(opt.weights, opt.source, opt.data, device)