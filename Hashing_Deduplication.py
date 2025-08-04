import os
import hashlib
import shutil
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor


class ImageDeduplicator:
    def __init__(self, root_folder, output_log="duplicates.log", phash_threshold=5):
        self.root = root_folder
        self.log_path = output_log
        self.file_records = {}
        self.duplicates = set()
        self.phash_threshold = phash_threshold  # 汉明距离阈值

    def _get_file_signature(self, file_path, mode='hash'):
        try:
            if mode == 'hash':
                return self._calc_hash(file_path)
            elif mode == 'phash':
                return self._calc_phash(file_path)
            elif mode == 'bytes':
                return self._calc_bytes(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def _calc_hash(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _calc_phash(self, file_path):
        """生成可比较的哈希对象"""
        with Image.open(file_path) as img:
            return imagehash.phash(img)

    def _calc_bytes(self, file_path):
        file_size = os.path.getsize(file_path)
        sample_points = [0, file_size // 2, max(0, file_size - 1024)]

        samples = []
        with open(file_path, 'rb') as f:
            for pos in sample_points:
                f.seek(pos)
                samples.append(f.read(1024))
        return b''.join(samples)

    def scan_duplicates(self, mode='hash', workers=4):
        all_images = []
        for root, dirs, files in os.walk(self.root):
            all_images.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._get_file_signature, path, mode): path for path in all_images}

            hash_groups = {}
            for future in futures:
                file_path = futures[future]
                sig = future.result()
                if not sig: continue

                # 特殊处理phash模式
                if mode == 'phash':
                    matched = False
                    for existing_hash in hash_groups:
                        if sig - existing_hash < self.phash_threshold:
                            hash_groups[existing_hash].append(file_path)
                            matched = True
                            break
                    if not matched:
                        hash_groups[sig] = [file_path]
                else:
                    if sig in hash_groups:
                        hash_groups[sig].append(file_path)
                    else:
                        hash_groups[sig] = [file_path]

            self.file_records = hash_groups

    def remove_duplicates(self, backup=True):
        removed = []
        with open(self.log_path, 'w') as log_file:
            for sig, files in self.file_records.items():
                if len(files) > 1:
                    sorted_files = sorted(files, key=lambda x: os.path.getctime(x))
                    keeper = sorted_files[0]

                    for file in sorted_files[1:]:
                        log_file.write(f"Duplicate: {file} -> Original: {keeper}\n")
                        if backup:
                            backup_path = file + ".bak"
                            shutil.move(file, backup_path)
                            removed.append(backup_path)
                        else:
                            os.remove(file)
                            removed.append(file)
        return removed


# 使用示例
if __name__ == "__main__":
    deduplicator = ImageDeduplicator(
        r"C:\Users\20253\Desktop\hip_joints",
        phash_threshold=1
    )
    deduplicator.scan_duplicates(mode='phash', workers=6)
    removed_files = deduplicator.remove_duplicates(backup=True)
    print(f"完成去重！已移除{len(removed_files)}个重复文件，日志见{deduplicator.log_path}")