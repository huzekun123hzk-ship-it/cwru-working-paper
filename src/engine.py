#!/usr/bin/env python3
"""训练与评估引擎。协议 R/S 共享此代码，保证唯一变量是划分协议。

重要：训练时不做任何频率轴增强（不 roll/flip 频率维）——频率轴带物理量纲，
翻转会破坏故障频带定位（ch45 铁律 + manual §11.3）。这与 ch41 原 train_model
的行为不同，是本研究的刻意设计。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, train_ds, val_ds, class_weights, seed,
                epochs=None, lr=None, batch_size=None, device=None):
    epochs = epochs or config.EPOCHS
    lr = lr or config.LR
    batch_size = batch_size or config.BATCH_SIZE
    device = device or config.DEVICE
    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        val_acc = evaluate(model, val_loader, device)["accuracy"]
        history["train_loss"].append(running / len(train_ds))
        history["val_acc"].append(val_acc)
    return model, history


@torch.no_grad()
def evaluate(model, loader, device=None):
    device = device or config.DEVICE
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return compute_metrics(y_true, y_pred)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.N_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.N_CLASSES)))
    per_class = {}
    for i, name in config.IDX_TO_CLASS.items():
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def make_test_loader(test_ds, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
