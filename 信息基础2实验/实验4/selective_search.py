import skimage.io
import skimage.segmentation
import skimage.color
import skimage.feature
import skimage.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import os
import time
from itertools import product

# --- 核心配置 ---
# 检查GPU是否可用，并设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# --- 核心辅助函数 ---

def _calculate_iou(box_a, boxes_b):
    """
    计算单个边界框(box_a)与一组边界框(boxes_b)之间的交并比(IoU)。
    这是NMS算法的核心计算。
    """
    # 确定相交矩形的坐标
    xA = torch.max(box_a[0], boxes_b[:, 0])
    yA = torch.max(box_a[1], boxes_b[:, 1])
    xB = torch.min(box_a[2], boxes_b[:, 2])
    yB = torch.min(box_a[3], boxes_b[:, 3])

    # 计算相交区域的面积
    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    # 分别计算每个边界框的面积
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # 计算并集面积
    union_area = box_a_area + boxes_b_area - inter_area

    # 计算IoU
    iou = inter_area / union_area
    return iou

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    执行非极大值抑制(NMS)来消除冗余的、重叠度高的边界框。
    """
    if not boxes.numel():
        return []

    # 根据分数对边界框进行降序排序
    _, indices = scores.sort(descending=True)
    
    keep_indices = [] # 用于存储最终保留的边界框的索引
    while indices.numel() > 0:
        # 选取当前分数最高的框
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if indices.numel() == 1:
            break
            
        # 计算当前框与剩余所有框的IoU
        iou = _calculate_iou(boxes[current_idx], boxes[indices[1:]])
        
        # 保留那些与当前框重叠度低于阈值的框，进入下一轮筛选
        non_overlapping_indices = indices[1:][iou < iou_threshold]
        indices = non_overlapping_indices
        
    return keep_indices

def _calculate_color_hist(img_tensor, mask, bins=25):
    """为图像的一个蒙版(mask)区域计算颜色直方图。"""
    pixels = img_tensor[:, mask.bool()]
    if pixels.shape[1] == 0:
        return torch.zeros(img_tensor.shape[0] * bins, device=device)
    histograms = [torch.histc(pixels[i], bins=bins, min=0, max=1.0) for i in range(img_tensor.shape[0])]
    hist_tensor = torch.cat(histograms)
    if torch.sum(hist_tensor) > 0: hist_tensor = hist_tensor / torch.sum(hist_tensor)
    return hist_tensor

def _calculate_texture_hist(gray_tensor, mask):
    """为灰度图的一个蒙版(mask)区域计算LBP纹理直方图。"""
    gray_np = gray_tensor.squeeze().cpu().numpy()
    mask_np = mask.cpu().numpy()
    rows, cols = np.where(mask_np)
    if len(rows) == 0: return torch.zeros(10, device=device)
    row_min, row_max, col_min, col_max = rows.min(), rows.max(), cols.min(), cols.max()
    region_gray_ubyte = skimage.util.img_as_ubyte(gray_np[row_min:row_max+1, col_min:col_max+1])
    region_mask = mask_np[row_min:row_max+1, col_min:col_max+1]
    lbp = skimage.feature.local_binary_pattern(region_gray_ubyte, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp[region_mask], bins=10, range=(0, 10))
    hist_tensor = torch.from_numpy(hist).float().to(device)
    if torch.sum(hist_tensor) > 0: hist_tensor = hist_tensor / torch.sum(hist_tensor)
    return hist_tensor

def _extract_regions(image_tensor, segments_tensor):
    """从初始分割结果中提取每个区域的属性（大小、位置、颜色和纹理特征）。"""
    regions = {}
    num_segments = segments_tensor.max().item() + 1
    img_for_gray_cpu = image_tensor.permute(1, 2, 0).cpu().numpy()
    if img_for_gray_cpu.shape[2] == 1: img_for_gray_cpu = np.stack([img_for_gray_cpu.squeeze()] * 3, axis=-1)
    gray_tensor = torch.from_numpy(skimage.color.rgb2gray(img_for_gray_cpu)).to(device)
    for i in range(num_segments):
        mask = (segments_tensor == i)
        if not mask.any(): continue
        region_size = mask.sum().item()
        rows, cols = torch.where(mask)
        bbox = (cols.min().item(), rows.min().item(), cols.max().item(), rows.max().item())
        regions[i] = {
            "mask": mask, "size": region_size, "bbox": bbox,
            "color_hist": _calculate_color_hist(image_tensor, mask),
            "texture_hist": _calculate_texture_hist(gray_tensor.unsqueeze(0), mask)
        }
    return regions

def _calculate_similarity(r1, r2, image_size, weights):
    """根据颜色、纹理、尺寸和填充度计算两个区域的相似度。"""
    color_sim = torch.sum(torch.min(r1['color_hist'], r2['color_hist']))
    texture_sim = torch.sum(torch.min(r1['texture_hist'], r2['texture_hist']))
    size_sim = 1.0 - (r1['size'] + r2['size']) / image_size
    bbox_union_size = (max(r1['bbox'][2], r2['bbox'][2]) - min(r1['bbox'][0], r2['bbox'][0])) * \
                      (max(r1['bbox'][3], r2['bbox'][3]) - min(r1['bbox'][1], r2['bbox'][1]))
    fill_sim = 1.0 - (bbox_union_size - r1['size'] - r2['size']) / image_size
    return (color_sim * weights['color'] + texture_sim * weights['texture'] +
            size_sim * weights['size'] + fill_sim * weights['fill'])

def _merge_regions(r1, r2):
    """合并两个区域，并更新它们的属性。"""
    new_mask = r1['mask'] | r2['mask']
    new_size = r1['size'] + r2['size']
    new_bbox = (min(r1['bbox'][0], r2['bbox'][0]), min(r1['bbox'][1], r2['bbox'][1]),
                max(r1['bbox'][2], r2['bbox'][2]), max(r1['bbox'][3], r2['bbox'][3]))
    return {
        "mask": new_mask, "size": new_size, "bbox": new_bbox,
        "color_hist": (r1['color_hist'] * r1['size'] + r2['color_hist'] * r2['size']) / new_size,
        "texture_hist": (r1['texture_hist'] * r1['size'] + r2['texture_hist'] * r2['size']) / new_size
    }

def _find_neighbours(segments_tensor):
    """遍历分割图，找到所有相互邻接的区域对。"""
    neighbours = set()
    height, width = segments_tensor.shape
    for y in range(height - 1):
        for x in range(width - 1):
            if segments_tensor[y, x] != segments_tensor[y, x+1]:
                neighbours.add(tuple(sorted((segments_tensor[y, x].item(), segments_tensor[y, x+1].item()))))
            if segments_tensor[y, x] != segments_tensor[y+1, x]:
                neighbours.add(tuple(sorted((segments_tensor[y, x].item(), segments_tensor[y+1, x].item()))))
    return list(neighbours)

# --- 算法主流程 ---

def _selective_search_single_pass(image_tensor, image_np_for_segmentation, scale, sigma, min_size, sim_weights):
    """
    使用给定的参数（颜色空间、分割尺度等）执行单次的选择性搜索流程。
    """
    image_size = image_tensor.shape[1] * image_tensor.shape[2]
    # 1. 初始分割
    segments_np = skimage.segmentation.felzenszwalb(image_np_for_segmentation, scale=scale, sigma=sigma, min_size=min_size)
    segments_tensor = torch.from_numpy(segments_np).to(device)
    # 2. 提取区域特征
    regions = _extract_regions(image_tensor, segments_tensor)
    if not regions: return []
    
    # 存储所有生成的提案，每个提案包含(bbox, score)
    proposals_with_scores = []
    
    # 3. 计算初始邻居和相似度
    neighbours = _find_neighbours(segments_tensor)
    similarities = {(i, j): _calculate_similarity(regions[i], regions[j], image_size, sim_weights)
                    for (i, j) in neighbours if i in regions and j in regions}
    
    # 4. 迭代合并
    max_label = segments_tensor.max().item() + 1
    while similarities:
        # 找到最相似的区域对
        sorted_sim = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        valid_pair_found = False
        for (i_candidate, j_candidate), _ in sorted_sim:
            if i_candidate in regions and j_candidate in regions:
                i, j = i_candidate, j_candidate
                valid_pair_found = True
                break
        if not valid_pair_found: break

        # 合并区域
        new_region = _merge_regions(regions[i], regions[j])
        new_label = max_label; max_label += 1
        regions[new_label] = new_region

        # **核心改进**: 计算新区域的"物体性"分数 (填充度) 并保存
        bbox = new_region['bbox']
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area > 0:
            fill_ratio_score = new_region['size'] / bbox_area
            proposals_with_scores.append((bbox, fill_ratio_score))

        # 更新相似度列表
        keys_to_remove = [k for k in similarities.keys() if k[0] in (i, j) or k[1] in (i, j)]
        old_neighbours = {k[0] if k[1] in (i,j) else k[1] for k in keys_to_remove}
        for k in keys_to_remove: del similarities[k]
        old_neighbours.discard(i); old_neighbours.discard(j)
        for neighbour_label in old_neighbours:
            if neighbour_label in regions:
                sim = _calculate_similarity(regions[new_label], regions[neighbour_label], image_size, sim_weights)
                similarities[tuple(sorted((new_label, neighbour_label)))] = sim
        del regions[i]; del regions[j]
        
    return proposals_with_scores

def selective_search_with_diversity(image_path):
    """
    实现多样性策略：在多种颜色空间和分割尺度下运行算法，并汇总所有结果。
    """
    original_img_np = skimage.io.imread(image_path)
    if original_img_np.shape[2] == 4: original_img_np = skimage.color.rgba2rgb(original_img_np)
    img_float_np = skimage.util.img_as_float(original_img_np)
    
    # 使用字典来存储唯一的bbox及其最高分数，避免重复
    unique_proposals = {}
    
    # 定义多样性策略
    color_spaces = ['RGB', 'HSV', 'Lab']
    scales = [100, 200, 300]
    sim_weights = {'color': 1.5, 'texture': 0.5, 'size': 1.0, 'fill': 1.0}

    print("--- 开始执行多样性选择性搜索 ---")
    for color_space, scale in product(color_spaces, scales):
        print(f"正在执行 -> 颜色空间: {color_space}, 分割尺度: {scale}")
        # 转换颜色空间
        if color_space == 'HSV': img_pass_np = skimage.color.rgb2hsv(img_float_np)
        elif color_space == 'Lab':
            img_pass_np = skimage.color.rgb2lab(img_float_np) # 归一化到[0,1]
            img_pass_np[:, :, 0] /= 100.0
            img_pass_np[:, :, 1] = (img_pass_np[:, :, 1] + 128) / 255.0
            img_pass_np[:, :, 2] = (img_pass_np[:, :, 2] + 128) / 255.0
        else: img_pass_np = img_float_np
        
        image_tensor = torch.from_numpy(np.ascontiguousarray(img_pass_np)).permute(2, 0, 1).float().to(device)
        
        # 执行单次搜索
        proposals = _selective_search_single_pass(
            image_tensor, img_float_np, scale=scale, sigma=0.8, min_size=scale, sim_weights=sim_weights)
        
        # 更新唯一提案字典，保留每个bbox的最高分
        for bbox, score in proposals:
            unique_proposals[bbox] = max(unique_proposals.get(bbox, 0), score)

        print(f"  ... 本轮找到 {len(proposals)} 个提案。当前总计唯一提案: {len(unique_proposals)}")

    print(f"--- 所有轮次执行完毕。共生成 {len(unique_proposals)} 个唯一提案。 ---")
    return unique_proposals, original_img_np

def post_process_and_visualize(image, proposals_dict, iou_thresh=0.7, max_proposals=40):
    """
    对所有候选框进行最终的后处理（过滤和NMS），并可视化结果。
    """
    print(f"\n--- 开始使用NMS进行后处理 ---")
    print(f"原始唯一提案数量: {len(proposals_dict)}")

    if not proposals_dict:
        print("没有可处理的提案。")
        return

    # 将提案字典转换为Tensor，用于NMS
    bboxes = list(proposals_dict.keys())
    scores = list(proposals_dict.values())
    boxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=device)
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    
    # 执行NMS，使用我们的"物体性"分数
    keep_indices = non_maximum_suppression(boxes_tensor, scores_tensor, iou_thresh)
    
    nms_boxes_tensor = boxes_tensor[keep_indices]
    nms_scores_tensor = scores_tensor[keep_indices]
    
    # 根据分数再次排序，确保我们只看最好的
    _, sorted_indices = nms_scores_tensor.sort(descending=True)
    
    final_boxes = nms_boxes_tensor[sorted_indices].cpu().numpy().tolist()
    
    print(f"经过NMS处理后，剩余 {len(final_boxes)} 个高质量提案。")

    # 可视化最终结果
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # 截取分数最高的N个框进行展示
    bboxes_to_show = final_boxes[:max_proposals]

    print(f"最终可视化展示 {len(bboxes_to_show)} 个最具代表性的提案。")
    for x_min, y_min, x_max, y_max in bboxes_to_show:
        rect = mpatches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  fill=False, edgecolor='red', linewidth=2) # 加粗边框
        ax.add_patch(rect)
    
    ax.set_title(f'Selective Search Proposals (showing {len(bboxes_to_show)})')
    ax.set_xlabel('Width'); ax.set_ylabel('Height')
    ax.axis('on')
    
    output_filename = 'selective_search_result_final_optimized.png'
    plt.savefig(output_filename)
    print(f"最终结果已保存至: {output_filename}")
    plt.close()

if __name__ == '__main__':
    start_time = time.time()
    image_file = 'sample.jpeg'
    
    if not os.path.exists(image_file):
        print(f"错误: 图像文件 '{image_file}' 不存在。")
    else:
        # 1. 生成所有候选框及其分数
        proposals, original_image = selective_search_with_diversity(image_path=image_file)
        
        # 2. 对结果进行后处理和可视化
        # iou_thresh: NMS阈值，调高会保留更多框
        # max_proposals: 最终希望看到的最大框数
        post_process_and_visualize(original_image, proposals, iou_thresh=0.7, max_proposals=40)
    
    end_time = time.time()
    print(f"\n程序总执行时间: {end_time - start_time:.2f} 秒。")