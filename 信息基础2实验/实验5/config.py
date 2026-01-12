# config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PASCAL VOC 数据集类别
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
NUM_CLASSES = len(PASCAL_CLASSES)

# 训练超参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
IMAGE_SIZE = 416 

# 模型相关
# 这些是针对COCO数据集预计算的，但对于VOC同样适用，也可以重新聚类计算
# (宽度, 高度)
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],      # 对应 stride 8 的小目标检测层
    [(30, 61), (62, 45), (59, 119)],     # 对应 stride 16 的中目标检测层
    [(116, 90), (156, 198), (373, 326)], # 对应 stride 32 的大目标检测层
]
# 每个检测层输出的网格尺寸
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8] # [13, 26, 52]

# 数据集路径
DATASET_PATH = "VOCdevkit/VOC2007" 

# 权重保存与加载
SAVE_MODEL_DIR = "weights/"
LOAD_MODEL_FILE = "yolov3_voc.pth.tar"