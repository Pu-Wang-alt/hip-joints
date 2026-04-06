import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.fftpack import dct
import shutil

class PaperImageDeduplicator:
    def __init__(self, root_folder, output_log="duplicates.log"):
        self.root = root_folder
        self.log_path = output_log
        self.file_records = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)
        self.encoder.eval()
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

    def _compute_global_hash(self, img_gray):
        """Eq 14: Global aware hashing (DCT)"""
        # Downsample to 32x32
        resized = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA)
        # Compute DCT
        dct_data = dct(dct(resized.T, norm='ortho').T, norm='ortho')
        # Extract low-frequency component (top-left 8x8) and binarize to 64-bit
        dct_low = dct_data[:8, :8]
        med = np.median(dct_low)
        hash_64 = (dct_low > med).astype(int).flatten()
        return hash_64

    def _compute_local_hash(self, img_gray):
        """Eq 15: Local difference hashing (Gradient structure)"""
        resized = cv2.resize(img_gray, (384, 384)) # 3x3 grid of 128x128 blocks
        h, w = resized.shape
        block_h, block_w = h // 3, w // 3
        
        local_hashes = []
        for i in range(3):
            for j in range(3):
                block = resized[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                # Compute gradients
                grad_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                # Hadamard product
                product = np.multiply(grad_x, grad_y)
                # Sorting encoding (simplified rank to 8-bit hash via binning)
                hist, _ = np.histogram(product, bins=8)
                local_hashes.append(np.argsort(hist))
        return local_hashes

    def _compute_roi_hash(self, img_rgb):
        """Eq 16: Anatomical ROI hashes (U-Net enc / Deep features)"""
        img_resized = cv2.resize(img_rgb, (256, 256))
        # Preprocess for PyTorch
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = (img_tensor / 255.0).to(self.device)
        
        with torch.no_grad():
            features = self.encoder(img_tensor)
            pooled = self.avgpool(features)
            # Flatten and reduce to R^256 vector
            flat_features = pooled.view(pooled.size(0), -1).cpu().numpy().flatten()
            roi_hash = flat_features[:256] # Truncate to 256 dimensions
            return roi_hash / (np.linalg.norm(roi_hash) + 1e-8) # Normalize

    def process_image(self, file_path):
        
        try:
            img = cv2.imread(file_path)
            if img is None: return None
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h_g = self._compute_global_hash(img_gray)
            h_l = self._compute_local_hash(img_gray)
            h_r = self._compute_roi_hash(img_rgb)
            
            
            timestamp = os.path.getctime(file_path)
            
            return {
                'path': file_path,
                'h_g': h_g,
                'h_l': h_l,
                'h_r': h_r,
                'time': timestamp
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def scan_and_deduplicate(self):
        
        all_images = []
        for root, dirs, files in os.walk(self.root):
            all_images.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg'))])
        
        print(f"Extract the features of {len(all_images)} images...")
        for path in all_images:
            record = self.process_image(path)
            if record:
                self.file_records.append(record)

        
        self.file_records.sort(key=lambda x: x['time'])
        
        kept_records = []
        removed_files = []
        
        with open(self.log_path, 'w') as log_file:
            for current in self.file_records:
                is_duplicate = False
                for kept in kept_records:
                    
                    delta_t_hours = abs(current['time'] - kept['time']) / 3600.0
                    l2_dist = np.linalg.norm(current['h_r'] - kept['h_r']) ** 2
                    
                    # Eq 17: R_keep calculation (Z is normalized out in decision boundary)
                    time_penalty = np.exp(-delta_t_hours / 24.0)
                    morph_penalty = np.exp(-0.7 * l2_dist)
                    
                    r_keep = time_penalty * morph_penalty
                    
                    # Algorithm 1: if P_keep >= 0.5 then keep
                    if r_keep >= 0.5:
                        is_duplicate = True
                        log_file.write(f"Duplicate detected (R_keep={r_keep:.3f}): {current['path']} -> Original: {kept['path']}\n")
                        break
                
                if not is_duplicate:
                    kept_records.append(current)
                else:
                    removed_files.append(current['path'])
                    
                    backup_path = current['path'] + ".bak"
                    shutil.move(current['path'], backup_path)
                    
        return removed_files

if __name__ == "__main__":
    deduplicator = PaperImageDeduplicator(r"C:\Users\20253\Desktop\hip_joints")
    removed = deduplicator.scan_and_deduplicate()
    print(f"Deduplication done! {len(removed)} files were blocked, see log {deduplicator.log_path}")
