

# 1. 创建模型
model = create_classification_head(input_dim=64, hidden_dim=32, output_dim=3)

# 2. 创建客户端实例
client = ESP32Client(
    device_id="esp32_001",
    data_dir="path/to/data",
    model=model,
    augment=True
)

# 3. 模拟联邦学习训练轮次
initial_params = model.get_weights()
config = {
    "batch_size": 64,
    "epochs": 3,
    "lr": 0.01
}

# 执行训练
updated_params, num_examples, metrics = client.fit(initial_params, config)

print(f"\n训练结果:")
print(f"- 训练样本数: {num_examples}")
print(f"- 最终损失: {metrics['train_loss']:.4f}")
print(f"- 最终准确率: {metrics['train_accuracy']:.4f}")

# 评估模型
loss, num, val_metrics = client.evaluate(updated_params, {})
print(f"\n评估结果 - 准确率: {val_metrics['val_accuracy']:.4f}")