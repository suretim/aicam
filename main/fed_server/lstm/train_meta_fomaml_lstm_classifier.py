#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# 超参数
# =============================
DATA_GLOB = "./data/*.csv"      # CSV: 每行 [feat1,...,featK,label]；无标签用 -1
SEQ_LEN = 64
FEATURE_DIM = 64                # LSTM 编码器输出
BATCH_SIZE = 32
EPOCHS_CONTRASTIVE = 10         # 对比学习轮数
EPOCHS_META = 30                # FOMAML 元学习轮数
INNER_STEPS = 1                 # 内层微调步数
INNER_LR = 1e-2
META_LR = 1e-3
NUM_CLASSES = 3

np.random.seed(42)
tf.random.set_seed(42)

# =============================
# 1) 加载 CSV -> 滑窗样本
# =============================
X_labeled_list, y_labeled_list = [], []
X_unlabeled_list = []

csv_files = sorted(glob.glob(DATA_GLOB))
if len(csv_files) == 0:
    raise FileNotFoundError(f"未找到数据：{DATA_GLOB}")

for file in csv_files:
    df = pd.read_csv(file).fillna(-1)
    data = df.values.astype(np.float32)
    feats = data[:, :-1]
    labels = data[:, -1]

    for i in range(len(data) - SEQ_LEN + 1):
        window_x = feats[i:i+SEQ_LEN]
        window_y = labels[i+SEQ_LEN-1]
        if window_y == -1:
            X_unlabeled_list.append(window_x)
        else:
            X_labeled_list.append(window_x)
            y_labeled_list.append(int(window_y))

X_unlabeled = np.array(X_unlabeled_list, dtype=np.float32)
if len(X_labeled_list) > 0:
    X_labeled = np.array(X_labeled_list, dtype=np.float32)
    y_labeled = np.array(y_labeled_list, dtype=np.int32)
    print("有标签样本:", X_labeled.shape)
else:
    X_labeled = np.empty((0, SEQ_LEN, X_unlabeled.shape[2] if X_unlabeled.size>0 else 3), dtype=np.float32)
    y_labeled = np.empty((0,), dtype=np.int32)
    print("警告：没有有标签样本，仅能做无监督预训练")

NUM_FEATS = X_labeled.shape[2] if X_labeled.size>0 else (X_unlabeled.shape[2] if X_unlabeled.size>0 else 3)

# =============================
# 2) 对比学习 (SimCLR)
# =============================
def augment_window(x):
    return x + np.random.normal(0, 0.01, x.shape).astype(np.float32)

def make_contrastive_pairs(X):
    anchors, positives = [], []
    for w in X:
        anchors.append(w)
        positives.append(augment_window(w))
    return np.stack(anchors).astype(np.float32), np.stack(positives).astype(np.float32)

class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def call(self, z_i, z_j):
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        logits = tf.matmul(z_i, z_j, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(z_i)[0])
        loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_j = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        return tf.reduce_mean(loss_i + loss_j)

def build_lstm_encoder(seq_len, num_feats, feature_dim=FEATURE_DIM):
    inp = layers.Input(shape=(seq_len, num_feats))
    x = layers.LSTM(feature_dim, return_sequences=False)(inp)
    out = layers.Dense(feature_dim, activation="relu")(x)
    return models.Model(inp, out, name="lstm_encoder")

lstm_encoder = build_lstm_encoder(SEQ_LEN, NUM_FEATS, FEATURE_DIM)
contrastive_opt = tf.keras.optimizers.Adam()

if X_unlabeled.size == 0:
    X_unlabeled = np.random.randn(200, SEQ_LEN, NUM_FEATS).astype(np.float32)

anchors, positives = make_contrastive_pairs(X_unlabeled)
contrast_ds = tf.data.Dataset.from_tensor_slices((anchors, positives)).shuffle(2048).batch(BATCH_SIZE)

ntxent = NTXentLoss(temperature=0.2)
for ep in range(EPOCHS_CONTRASTIVE):
    for a, p in contrast_ds:
        with tf.GradientTape() as tape:
            za = lstm_encoder(a, training=True)
            zp = lstm_encoder(p, training=True)
            loss = ntxent(za, zp)
        grads = tape.gradient(loss, lstm_encoder.trainable_variables)
        contrastive_opt.apply_gradients(zip(grads, lstm_encoder.trainable_variables))
    print(f"[Contrastive] Epoch {ep+1}/{EPOCHS_CONTRASTIVE}, loss={float(loss.numpy()):.4f}")

# =============================
# 3) FOMAML 元学习
# =============================
def build_meta_model(encoder, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(SEQ_LEN, NUM_FEATS))
    x = encoder(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="meta_lstm_classifier")

meta_model = build_meta_model(lstm_encoder, NUM_CLASSES)
meta_optimizer = tf.keras.optimizers.Adam(META_LR)

# 生成支持/查询任务
def sample_tasks(X, y, num_tasks=5, support_size=10, query_size=20):
    tasks = []
    for _ in range(num_tasks):
        idx = np.random.choice(len(X), support_size+query_size, replace=False)
        X_support, y_support = X[idx[:support_size]], y[idx[:support_size]]
        X_query, y_query = X[idx[support_size:]], y[idx[support_size:]]
        tasks.append((X_support, y_support, X_query, y_query))
    return tasks

# FOMAML 外层更新
def outer_update_fomaml(meta_model, meta_optimizer, tasks, lr_inner=INNER_LR):
    meta_grads = [tf.zeros_like(var) for var in meta_model.trainable_variables]
    for X_support, y_support, X_query, y_query in tasks:
        original_weights = [w.numpy() for w in meta_model.trainable_variables]
        # inner update
        with tf.GradientTape() as tape_inner:
            preds_support = meta_model(X_support, training=True)
            loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_support, preds_support))
        grads_inner = tape_inner.gradient(loss_support, meta_model.trainable_variables)
        temp_vars = [w - lr_inner*g for w,g in zip(meta_model.trainable_variables, grads_inner)]
        # query 上求梯度
        with tf.GradientTape() as tape:
            for var, temp in zip(meta_model.trainable_variables, temp_vars):
                var.assign(temp)
            preds_query = meta_model(X_query, training=True)
            loss_query = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_query, preds_query))
        grads = tape.gradient(loss_query, meta_model.trainable_variables)
        meta_grads = [mg + g for mg, g in zip(meta_grads, grads)]
        # 恢复 meta 参数
        for var, orig in zip(meta_model.trainable_variables, original_weights):
            var.assign(orig)
    meta_grads = [mg / len(tasks) for mg in meta_grads]
    meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))

# =============================
# 元训练循环
# =============================
if X_labeled.size > 0:
    for ep in range(EPOCHS_META):
        tasks = sample_tasks(X_labeled, y_labeled, num_tasks=5)
        outer_update_fomaml(meta_model, meta_optimizer, tasks)
        print(f"[Meta] Epoch {ep+1}/{EPOCHS_META} finished.")
else:
    print("跳过元学习：没有有标签数据。")

# =============================
# 4) 导出 TFLite
# =============================
def save_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", out_path)

save_tflite(lstm_encoder, "lstm_encoder_contrastive.tflite")
if X_labeled.size > 0:
    save_tflite(meta_model, "meta_lstm_classifier.tflite")
