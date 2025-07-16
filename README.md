# hip-joints


A Python toolkit for preprocessing and filtering medical images to prepare datasets for deep learning research. Features include similarity-based filtering, duplicate removal, and color contamination detection.

## Scripts Overview

### 1. Image Similarity Filtering (`nainiu_qumohu.py`)
**Purpose**: Filters blurry/irrelevant images by comparing them to high-quality ground truth samples  
**Core Logic**:  
- Uses ResNet-101 to extract deep features  
- Combines Manhattan + Cosine distances to calculate fused similarity metric  
- Retains images below preset distance threshold  

**Key Libraries**:  
`PyTorch`, `OpenCV`, `scikit-learn`, `NumPy`

### 2. Duplicate Image Removal (`qu_chong3.py`)
**Purpose**: Identifies and removes exact/visually similar duplicates  
**Detection Modes**:  
- `hash`: MD5-based exact matching  
- `phash`: Perceptual hashing (recommended)  
- `bytes`: Byte sample comparison  

**Processing Flow**:  
1. Recursively scan directory and calculate signatures  
2. Group images by signature similarity  
3. Retain earliest created file in each group  
4. Backup/delete duplicates  

**Key Libraries**:  
`Pillow`, `imagehash`, `concurrent.futures`

### 3. Contaminated Image Detection (`xiaolandian.py`)
**Purpose**: Identifies images with color spots/stains  
**Detection Method**:  
1. Analyze RGB channel differentials  
2. Flag pixels with max-min value > threshold  
3. Apply edge masking to prevent false positives  

**Output Features**:  
- Generates HTML report with contamination statistics  
- Automatic folder segregation  
- Contamination distribution histograms  

**Key Libraries**:  
`OpenCV`, `NumPy`, `Pandas`, `Matplotlib`

---

## How to Use

### 1. Prerequisites
```bash
pip install torch torchvision opencv-python scikit-learn numpy Pillow imagehash pandas matplotlib
