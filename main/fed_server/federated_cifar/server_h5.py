import h5py
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ESP32TOH5:
    def __init__(self, data_dir, device_id=None):
        self.data_dir = Path(data_dir)
        self.device_id = device_id


    def save_esp32_features(self,
                            features: np.ndarray,
                            labels: np.ndarray,
                            metadata: dict = None):
        """
        保存ESP32特徵數據到HDF5文件
        參數:
            device_id: 設備唯一標識符
            features: 編碼器輸出 (n_samples, 64)
            labels: 對應標籤 (n_samples,)
            output_dir: 存儲目錄
            metadata: 附加元數據
        """
        # 創建目錄
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/{self.device_id}_{timestamp}.h5"

        # 保存到HDF5
        with h5py.File(filename, 'w') as f:
            if features.ndim == 0:  # Scalar check
                f.create_dataset("features", data=features)
            else:
                f.create_dataset("features", data=features, compression="gzip")

            # Same for labels
            if labels.ndim == 0:
                f.create_dataset("labels", data=labels)
            else:
                f.create_dataset("labels", data=labels, compression="gzip")

            # Save metadata (no compression)
            #if metadata:
            #    for key, value in metadata.items():
            #        f.create_dataset(f"metadata/{key}", data=value)

            # 保存主要數據
            #f.create_dataset("features", data=features, compression="gzip" )
            #f.create_dataset("labels", data=labels, compression="gzip" )

            # 保存元數據
            #if metadata is None:
            metadata = {}
            metadata.update({
                "device_id": self.device_id,
                "timestamp": timestamp,
                "num_samples": len(features),
                "feature_dim": features.shape[1]
            })
            f.attrs.update(metadata)

        print(f"數據已保存到 {filename}")
        return filename

    def _normalize_features(self,features: np.ndarray) -> np.ndarray:
        """標準化特徵數據 (每個維度0均值1方差)"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-7)  # 避免除零


    def save_with_versioning(self,device_id, features, labels, output_dir="data", max_versions=5):
        """帶版本控制的數據保存"""
        # 查找現有版本
        existing_files = sorted(Path(output_dir).glob(f"{device_id}_v*.h5"))

        # 確定新版本號
        version = 1
        if existing_files:
            last_version = int(existing_files[-1].stem.split("_v")[-1])
            version = last_version + 1

        # 刪除舊版本
        if len(existing_files) >= max_versions:
            for file in existing_files[:-(max_versions - 1)]:
                file.unlink()

        # 保存新文件
        filename = f"{output_dir}/{device_id}_v{version}.h5"
        self.save_esp32_features(device_id, features, labels, filename)
if __name__ == '__main__':

    data_dir="data"
    device_id="client_002"
    data_gen=ESP32TOH5(data_dir,device_id)
    # 使用示例
    features = np.random.rand(100, 64).astype(np.float32)  # 模擬ESP32輸出
    labels = np.random.randint(0, 3, size=100)
    data_gen.save_esp32_features(
        device_id="esp32_001",
        features=features,
        labels=labels,
        metadata={
            "location": "lab_A",
            "sensor_type": "accelerometer_v2"
        }
    )
