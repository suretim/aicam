import h5py
import numpy as np 
from pathlib import Path
class ESP32DataLoader:
    def __init__(self, data_dir, device_id=None):
        self.data_dir = Path(data_dir)
        self.device_id = device_id

    def load(self, max_samples=None):
        """加载数据并返回 (features, labels) 元组"""
        # 实现数据加载逻辑
        files = self._find_data_files()
        features, labels = [], []

        for file in files:
            with h5py.File(file, 'r') as f:
                feats = f["features"][:]
                lbls = f["labels"][:]

                if max_samples and len(feats) > max_samples:
                    idx = np.random.choice(len(feats), max_samples, False)
                    feats, lbls = feats[idx], lbls[idx]

                features.append(feats)
                labels.append(lbls)

        return np.concatenate(features), np.concatenate(labels)

    def _find_data_files(self):
        """查找匹配的数据文件"""
        pattern = f"{self.device_id}_*.h5" if self.device_id else "*.h5"
        files = list(self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No data files found for {self.device_id}")
        return files