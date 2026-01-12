# create_csv_files.py

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- 配置区 ---
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
VOC_ROOT_PATH = "VOCdevkit/VOC2007/"
# --- 脚本主逻辑 ---

def convert_voc_annotation_to_yolo(image_id, voc_root_path):
    xml_file = os.path.join(voc_root_path, "Annotations", f"{image_id}.xml")
    if not os.path.exists(xml_file):
        print(f"警告：找不到标注文件 {xml_file}")
        return None, None
        
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    dw = 1.0 / width
    dh = 1.0 / height

    all_objects_str = []
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        class_name = obj.find("name").text
        
        if class_name not in PASCAL_CLASSES or int(difficult) == 1:
            continue
            
        class_id = PASCAL_CLASSES.index(class_name)
        
        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymin = float(xmlbox.find("ymin").text)
        ymax = float(xmlbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        
        x_center_norm = x_center * dw
        y_center_norm = y_center * dh
        w_norm = w * dw
        h_norm = h * dh
        
        object_str = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        all_objects_str.append(object_str)

    image_path = os.path.join("JPEGImages", f"{image_id}.jpg")
    
    # 返回图片路径和拼接好的所有物体标签字符串
    return image_path, " ".join(all_objects_str)


def generate_csv_file(dataset_split, voc_root_path):
    print(f"开始处理 {dataset_split} 数据集...")
    
    output_csv_name = "train.csv" if dataset_split == "trainval" else "test.csv"
    output_path = os.path.join(voc_root_path, output_csv_name)
    
    image_sets_file = os.path.join(voc_root_path, "ImageSets", "Main", f"{dataset_split}.txt")
    
    with open(image_sets_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    with open(output_path, "w") as out_file:
        for image_id in tqdm(image_ids, desc=f"解析并写入 {dataset_split} 数据"):
            image_path, labels_str = convert_voc_annotation_to_yolo(image_id, voc_root_path)
            
            if image_path and labels_str:
                out_file.write(f"{image_path} {labels_str}\n")

    print(f"成功生成 {output_path}")


if __name__ == "__main__":
    generate_csv_file("trainval", VOC_ROOT_PATH)
    generate_csv_file("test", VOC_ROOT_PATH)