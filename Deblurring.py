import os
import cv2
import torch
import shutil
import numpy as np
from torchvision import transforms
from torchvision.models import resnet101
from skimage.metrics import structural_similarity


support_dir = "/home/wangpu/wangpu/hip_joints/qumohu/ground_truth"
query_root = "/home/wangpu/wangpu/hip_joints/qumohu/daiqulandian"
output_root = "/home/wangpu/wangpu/hip_joints/qumohu/outputs1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 224
BATCH_SIZE = 8
DISTANCE_THRESHOLD = 0.30

ALPHA = 0.38   # Manhattan
BETA = 0.29    # Cosine
GAMMA = 0.33   # SSIM

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(output_root, exist_ok=True)


backbone = resnet101(pretrained=True).to(DEVICE)
backbone.eval()

stem = torch.nn.Sequential(
    backbone.conv1,
    backbone.bn1,
    backbone.relu,
    backbone.maxpool
).to(DEVICE)

layer1 = backbone.layer1.to(DEVICE)
layer2 = backbone.layer2.to(DEVICE)
layer3 = backbone.layer3.to(DEVICE)   
layer4 = backbone.layer4.to(DEVICE)   

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def list_images(folder):
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        ext = os.path.splitext(f)[1].lower()
        if os.path.isfile(path) and ext in IMAGE_EXTS:
            files.append(path)
    return sorted(files)

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def channel_recalibration(x):
    z = torch.mean(x, dim=(2, 3), keepdim=True)
    z = (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)
    w = torch.sigmoid(z)
    return x * w

def extract_batch_features(batch_tensor):
    with torch.no_grad():
        x = stem(batch_tensor)
        x = layer1(x)
        x = layer2(x)

        conv4 = layer3(x)
        conv5 = layer4(conv4)

        conv4 = channel_recalibration(conv4)

        f4 = torch.nn.functional.adaptive_avg_pool2d(conv4, 1).flatten(1)
        f5 = torch.nn.functional.adaptive_avg_pool2d(conv5, 1).flatten(1)

        feat = torch.cat([f4, f5], dim=1)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    return feat.cpu().numpy()

def extract_features_and_gray(img_paths):
    features = []
    grays = []
    batch = []

    for i, path in enumerate(img_paths, 1):
        img_rgb = read_rgb(path)

        batch.append(preprocess(img_rgb))

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        grays.append(gray)

        if len(batch) == BATCH_SIZE or i == len(img_paths):
            batch_tensor = torch.stack(batch).to(DEVICE)
            feat = extract_batch_features(batch_tensor)
            features.append(feat)
            batch = []

    return np.vstack(features).astype(np.float32), np.stack(grays).astype(np.float32)

def normalize_dist(dist):
    m = dist.max()
    if m <= 1e-12:
        return np.zeros_like(dist, dtype=np.float32)
    return (dist / m).astype(np.float32)

def manhattan_distance(query_gray, support_gray):
    q = query_gray.reshape(len(query_gray), -1)
    s = support_gray.reshape(len(support_gray), -1)
    return np.abs(q[:, None, :] - s[None, :, :]).sum(axis=2)

def cosine_distance(query_feat, support_feat):
    q = query_feat / (np.linalg.norm(query_feat, axis=1, keepdims=True) + 1e-12)
    s = support_feat / (np.linalg.norm(support_feat, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(q @ s.T, -1.0, 1.0)
    return 1.0 - sim

def ssim_distance(query_gray, support_gray):
    dist = np.zeros((len(query_gray), len(support_gray)), dtype=np.float32)
    for i in range(len(query_gray)):
        for j in range(len(support_gray)):
            score = structural_similarity(
                query_gray[i],
                support_gray[j],
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                win_size=11,
                data_range=1.0,
            )
            dist[i, j] = 1.0 - float(score)
    return dist


support_paths = list_images(support_dir)
support_features, support_gray = extract_features_and_gray(support_paths)


for subdir in os.listdir(query_root):
    query_dir = os.path.join(query_root, subdir)
    if not os.path.isdir(query_dir):
        continue

    query_paths = list_images(query_dir)
    if len(query_paths) == 0:
        print(f"[Skip] No images found in {query_dir}")
        continue

    query_features, query_gray = extract_features_and_gray(query_paths)

    raw_dist = manhattan_distance(query_gray, support_gray)
    deep_dist = cosine_distance(query_features, support_features)
    ssim_dist = ssim_distance(query_gray, support_gray)

    raw_dist = normalize_dist(raw_dist)
    deep_dist = normalize_dist(deep_dist)
    ssim_dist = normalize_dist(ssim_dist)

    fused_dist = ALPHA * raw_dist + BETA * deep_dist + GAMMA * ssim_dist

    min_distances = fused_dist.min(axis=1)
    valid_indices = np.where(min_distances <= DISTANCE_THRESHOLD)[0]

    sub_output_dir = os.path.join(output_root, subdir)
    os.makedirs(sub_output_dir, exist_ok=True)

    for idx in valid_indices:
        src_path = query_paths[idx]
        dst_path = os.path.join(sub_output_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"[Accepted] Saved: {dst_path} (Distance: {min_distances[idx]:.4f})")

    print(f"Subdir '{subdir}': Total {len(query_paths)} images, {len(valid_indices)} passed threshold.")
