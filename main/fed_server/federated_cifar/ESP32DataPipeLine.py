import h5py
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ESP32DataPipeline:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.cache = {}  # 可選的緩存機制


    def _normalize_features(self,features: np.ndarray) -> np.ndarray:
        """標準化特徵數據 (每個維度0均值1方差)"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-7)  # 避免除零



    def load_and_preprocess_esp32_data(self,data_dir: str,
                                       client_id: str = None,
                                       max_samples: int = None):
        """
        從目錄加載ESP32數據並預處理
        參數:
            data_dir: 數據目錄
            client_id: 指定設備ID (None則加載所有)
            max_samples: 每設備最大樣本數 (None則全部)
        返回:
            (features, labels) 元組
        """
        data_files = []

        # 查找匹配文件
        if client_id is None:
            data_files = list(Path(data_dir).glob("*.h5"))
        else:
            data_files = list(Path(data_dir).glob(f"{client_id}_*.h5"))

        if not data_files:
            raise FileNotFoundError(f"找不到 {client_id} 的數據文件")

        all_features = []
        all_labels = []

        for file in data_files:
            with h5py.File(file, 'r') as f:
                # 讀取數據
                features = f["features"][:]
                labels = f["labels"][:]

                # 限制樣本數
                if max_samples is not None and len(features) > max_samples:
                    indices = np.random.choice(len(features), max_samples, replace=False)
                    features = features[indices]
                    labels = labels[indices]

                all_features.append(features)
                all_labels.append(labels)

        # 合併數據
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)

        # 數據預處理
        features = self._normalize_features(features)
        labels = labels.astype(np.int32)

        # 打亂數據
        shuffle_idx = np.random.permutation(len(features))
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        return features, labels


    def load_device_data(self, device_id):
        """加載單個設備數據並緩存"""
        if device_id in self.cache:
            return self.cache[device_id]

        features, labels = self.load_and_preprocess_esp32_data(
            self.data_dir,
            client_id=device_id
        )

        self.cache[device_id] = (features, labels)
        return features, labels

    def get_federated_dataset(self, devices, samples_per_device=None):
        """創建聯邦學習數據集"""
        federated_data = []

        for device_id in devices:
            features, labels = self.load_device_data(device_id)

            if samples_per_device and len(features) > samples_per_device:
                indices = np.random.choice(
                    len(features),
                    samples_per_device,
                    replace=False
                )
                features = features[indices]
                labels = labels[indices]

            federated_data.append((features, labels))

        return federated_data

    def get_centralized_dataset(self, devices=None, test_size=0.2):
        """創建集中式訓練數據集"""
        if devices is None:
            devices = self.get_available_devices()

        all_features = []
        all_labels = []

        for device_id in devices:
            features, labels = self.load_device_data(device_id)
            all_features.append(features)
            all_labels.append(labels)

        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)

        # 分割訓練/測試集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return (X_train, y_train), (X_test, y_test)

    def get_available_devices(self):
        """獲取所有可用設備ID"""
        files = self.data_dir.glob("*.h5")
        return list({f.stem.split("_")[0] for f in files})


# 使用示例
pipeline = ESP32DataPipeline("data")
devices = pipeline.get_available_devices()
federated_data = pipeline.get_federated_dataset(devices, samples_per_device=500)

from concurrent.futures import ThreadPoolExecutor

from my_load import MY_LOAD as loader

# 创建数据加载器
data_loader = loader(data_dir="data", device_id="esp32_001")

from pathlib import Path

# 獲取數據目錄下所有 .h5 文件
data_dir = "data"
files = list(Path(data_dir).glob("*.h5"))

# 檢查文件列表
print(f"找到 {len(files)} 個數據文件")
for file in files[:3]:  # 顯示前3個文件
    print(file.name)
# 只加載特定ESP32設備的數據 (例如設備ID為esp32_001)
target_device = "esp32_001"
device_files = [f for f in files if f.stem.startswith(target_device)]

if not device_files:
    print(f"找不到設備 {target_device} 的數據文件")
else:
    # 並行加載該設備的所有數據文件
    device_features, device_labels = data_loader._parallel_load(device_files)

    print(f"設備 {target_device} 的數據:")
    print(f"- 樣本數: {len(device_features)}")
    print(f"- 最新文件: {device_files[-1].name}")

# 當數據量很大時，可以分批加載處理
batch_size = 5  # 每次處理5個文件
total_files = len(files)

for i in range(0, total_files, batch_size):
    batch_files = files[i:i + batch_size]
    print(f"\n處理文件 {i + 1}-{min(i + batch_size, total_files)}/{total_files}")

    # 並行加載當前批次
    batch_features, batch_labels = data_loader._parallel_load(batch_files)

    # 在這裡添加您的處理代碼
    # 例如: 訓練模型、計算統計量等
    mean_features = np.mean(batch_features, axis=0)
    print(f"本批次特徵均值: {mean_features[:5]}...")  # 顯示前5維


