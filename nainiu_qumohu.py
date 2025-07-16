import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet101
from sklearn.metrics.pairwise import pairwise_distances
import shutil

output_root = "/home/wangpu/wangpu/hip_joints/qumohu/outputs1"  # 定义输出根目录
os.makedirs(output_root, exist_ok=True)  # 自动创建目录（如果不存在）

DISTANCE_THRESHOLD = 0.28  # 示例值，需根据实际数据分布调整
# 1. 加载预训练模型（ResNet-101）
model = resnet101(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 移除最后一层
model.eval()

# 2. 定义图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.225])
])

# 3. 提取支持集特征
support_features = []
for img_path in os.listdir(r"/home/wangpu/wangpu/hip_joints/qumohu/ground_truth"):
    img = cv2.imread(f"/home/wangpu/wangpu/hip_joints/qumohu/ground_truth/{img_path}")
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(img_tensor).squeeze().numpy()
    support_features.append(feature)
support_features = np.array(support_features)

# 4. 处理每个子文件夹
for subdir in os.listdir("/home/wangpu/wangpu/hip_joints/qumohu/daiqulandian"):
    query_dir = os.path.join("/home/wangpu/wangpu/hip_joints/qumohu/daiqulandian", subdir)
    query_features = []
    img_paths = []

    # 提取查询集特征
    for img_file in os.listdir(query_dir):
        img = cv2.imread(os.path.join(query_dir, img_file))
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feature = model(img_tensor).squeeze().numpy()
        query_features.append(feature)
        img_paths.append(img_file)

    # 5. 计算融合距离
    raw_dist = pairwise_distances(query_features, support_features, metric='manhattan')
    deep_dist = pairwise_distances(query_features, support_features, metric='cosine')

    # 归一化
    raw_dist_norm = raw_dist / np.max(raw_dist)
    deep_dist_norm = deep_dist / np.max(deep_dist)

    # 融合距离（α=0.4）
    alpha = 0.4
    fused_dist = alpha * raw_dist_norm + (1 - alpha) * deep_dist_norm

    # # 6. 筛选策略（取每行最小距离）
    # min_dist_indices = np.argmin(fused_dist, axis=1)
    # selected_images = [img_paths[i] for i in min_dist_indices]


    # # 7. 保存结果（示例：每个子文件夹保留Top-5）
    # top_k = 5
    # sorted_indices = np.argsort(fused_dist.min(axis=1))[:top_k]
    min_distances = fused_dist.min(axis=1)  # shape: (n_queries,)
    valid_indices = np.where(min_distances <= DISTANCE_THRESHOLD)[0]
    sub_output_dir = os.path.join(output_root, subdir)
    os.makedirs(sub_output_dir, exist_ok=True)

    # for idx in sorted_indices:
    #     print(f"Selected: {img_paths[idx]}, Distance: {fused_dist.min(axis=1)[idx]:.4f}")
    #     src_path = os.path.join(query_dir, img_paths[idx])  # 原始文件路径
    #     dest_dir = os.path.join(sub_output_dir, img_paths[idx])  # 目标路径
    #
    #     # 复制文件到输出目录
    #     os.makedirs(dest_dir, exist_ok=True)
    #     dest_path = os.path.join(dest_dir, img_paths[idx])
    #     shutil.copy(src_path, dest_path)
    #     print(f"Copied: {img_paths[idx]} -> {dest_path}")
    for idx in valid_indices:
        src_path = os.path.join(query_dir, img_paths[idx])
        dest_path = os.path.join(sub_output_dir, img_paths[idx])
        shutil.copy(src_path, dest_path)
        print(f"[Threshold Filter] Saved: {dest_path} (Distance: {min_distances[idx]:.4f})")

        # 输出统计信息
    print(f"Subdir '{subdir}': Total {len(img_paths)} images, {len(valid_indices)} passed threshold.")