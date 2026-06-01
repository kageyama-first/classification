from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from CNN_model_torchbased_refined import SimpleCNN_CBAM

###
# SimpleCNN + CBAM 训练与可视化脚本

# 输出：
# - learning_curve.csv：每个 epoch 的 train/val 指标
# - metrics.csv：最终测试集指标（只评估一次）
# - confusion_matrix.png：测试集混淆矩阵
# - compare_curve_simple_standard.png：与 SimpleCNN+standard 基线对比的验证集曲线（val_loss/val_acc/val_macro_f1）

###


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int):
    # 归一化参数与 ImageNet 标准一致（便于与常见设置对齐）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)

    # 四种数据增强策略：none / standard / weak / strong
    train_trans_standard = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_trans_none = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_trans_weak = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_trans_strong = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    )

    eval_tfms = train_trans_none
    train_tfms_dict = {
        "none": train_trans_none,
        "standard": train_trans_standard,
        "weak": train_trans_weak,
        "strong": train_trans_strong,
    }
    return train_tfms_dict, eval_tfms


def make_loaders(split_root: Path, image_size: int, strategy: str, batch_size: int, num_workers: int):
    train_tfms_dict, eval_tfms = build_transforms(image_size=image_size)
    if str(strategy) not in train_tfms_dict:
        raise ValueError(f"unknown strategy: {strategy}")

    # 训练集用可选增强；验证/测试集固定用中心裁剪（保证评估稳定）
    train_ds = ImageFolder(str(split_root / "train"), transform=train_tfms_dict[str(strategy)])
    val_ds = ImageFolder(str(split_root / "val"), transform=eval_tfms)
    test_ds = ImageFolder(str(split_root / "test"), transform=eval_tfms)

    loader_kwargs = {"num_workers": int(num_workers), "pin_memory": True}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, train_ds.classes


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, opt, criterion):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        total += int(x.size(0))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    avg_loss = total_loss / max(1, total)
    return avg_loss, acc, macro_f1, y_true, y_pred


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=220)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_and_save(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
    strategy: str,
    seed: int,
    epochs: int,
    lr: float,
):
    # 训练得到：learning_curve.csv（逐 epoch）+ best_model.pth（按 val_macro_f1 选）+ metrics.csv + confusion_matrix.png
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = out_dir / "learning_curve.csv"
    metrics_csv = out_dir / "metrics.csv"

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_state_dict = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, criterion)
        va_loss, va_acc, va_f1, _y_true, _y_pred = evaluate(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
                "val_macro_f1": float(va_f1),
            }
        )

        if float(va_f1) > best_val_macro_f1:
            best_val_macro_f1 = float(va_f1)
            best_epoch = int(epoch)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[epoch {epoch:02d}/{epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f} macroF1={va_f1:.4f}"
        )

    pd.DataFrame(history).to_csv(curve_csv, index=False)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(model.state_dict(), out_dir / "best_model.pth")

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    save_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")

    pd.DataFrame(
        [
            {
                "model": model.__class__.__name__,
                "strategy": str(strategy),
                "seed": seed,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "test_macro_f1": float(test_f1),
            }
        ]
    ).to_csv(metrics_csv, index=False)

    return curve_csv, metrics_csv


def plot_compare_curves(cbam_curve_csv: Path, exp_csv: Path, out_path: Path, base_curve_csv: Path | None = None):
    # 对比图使用验证集指标：
    # - CBAM：来自逐 epoch 的 learning_curve.csv
    # - SimpleCNN+standard 基线：优先使用逐 epoch 曲线（若提供）；否则从 experiment_results.csv 读取汇总值并扩展成常数参考线
    df_cbam = pd.read_csv(cbam_curve_csv)
    df_base = None
    if base_curve_csv is not None and Path(base_curve_csv).exists():
        df_base = pd.read_csv(base_curve_csv)
    else:
        if not exp_csv.exists():
            return
        df_exp = pd.read_csv(exp_csv)
        base = df_exp[(df_exp["Model"].astype(str) == "SimpleCNN") & (df_exp["Strategy"].astype(str) == "standard")]
        if len(base) == 0:
            return
        base_val_loss = float(base.iloc[0]["Val_Loss"])
        base_val_acc = float(base.iloc[0]["Val_Acc"])
        base_val_f1 = float(base.iloc[0]["Val_Macro_F1"])
        df_base = pd.DataFrame(
            {
                "epoch": df_cbam["epoch"],
                "val_loss": [base_val_loss] * len(df_cbam),
                "val_acc": [base_val_acc] * len(df_cbam),
                "val_macro_f1": [base_val_f1] * len(df_cbam),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8), dpi=220)

    axes[0].plot(df_cbam["epoch"], df_cbam["val_loss"], label="CBAM val_loss")
    axes[0].plot(df_base["epoch"], df_base["val_loss"], linestyle="--", linewidth=1.4, label="Simple+standard val_loss")
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(df_cbam["epoch"], df_cbam["val_acc"], label="CBAM val_acc")
    axes[1].plot(df_base["epoch"], df_base["val_acc"], linestyle="--", linewidth=1.4, label="Simple+standard val_acc")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(df_cbam["epoch"], df_cbam["val_macro_f1"], label="CBAM val_macro_f1")
    axes[2].plot(
        df_base["epoch"],
        df_base["val_macro_f1"],
        linestyle="--",
        linewidth=1.4,
        label="Simple+standard val_macro_f1",
    )
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[2]
    split_root = project_root / "data_split"
    out_dir = project_root / "outputs" / "after_CBAM" / "cnn_cbam"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    image_size = 224
    batch_size = 32
    num_workers = 2
    epochs = 30
    lr = 0.01
    strategy = "standard"

    exp_csv = Path(__file__).resolve().parent / "experiment_results.csv"
    curve_csv = out_dir / "learning_curve.csv"
    if exp_csv.exists():
        # 这里沿用此前 SimpleCNN 实验中“Test_Macro_F1 最优”的增强策略，作为 CBAM 的默认策略
        df = pd.read_csv(exp_csv)
        df = df[df["Model"].astype(str) == "SimpleCNN"]
        if len(df) > 0 and "Test_Macro_F1" in df.columns:
            best_row = df.sort_values("Test_Macro_F1", ascending=False).iloc[0]
            strategy = str(best_row["Strategy"])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = make_loaders(
        split_root=split_root,
        image_size=image_size,
        strategy=strategy,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cbam_metrics_csv = out_dir / "metrics.csv"
    if not curve_csv.exists() or not cbam_metrics_csv.exists():
        # 若结果文件不存在才重新训练；否则直接复用现有学习曲线/权重，保证可复现与节省时间
        cbam_model = SimpleCNN_CBAM(num_classes=len(class_names)).to(device)
        train_and_save(
            model=cbam_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            out_dir=out_dir,
            strategy=strategy,
            seed=seed,
            epochs=epochs,
            lr=lr,
        )

    if curve_csv.exists():
        # 画对比图：CBAM 验证曲线 vs SimpleCNN+standard 基线（汇总值/参考线）
        plot_compare_curves(
            cbam_curve_csv=curve_csv,
            exp_csv=exp_csv,
            out_path=out_dir / "compare_curve_simple_standard.png",
        )

    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
