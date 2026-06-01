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


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)

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


def main():
    project_root = Path(__file__).resolve().parents[2]
    split_root = project_root / "data_split"
    out_dir = project_root / "outputs" / "after" / "cnn_cbam"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    image_size = 224
    batch_size = 32
    num_workers = 2
    epochs = 30
    lr = 0.01
    strategy = "standard"

    exp_csv = Path(__file__).resolve().parent / "experiment_results.csv"
    if exp_csv.exists():
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

    model = SimpleCNN_CBAM(num_classes=len(class_names)).to(device)
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

    pd.DataFrame(history).to_csv(out_dir / "learning_curve.csv", index=False)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(model.state_dict(), out_dir / "best_model.pth")

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    save_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")

    pd.DataFrame(
        [
            {
                "model": "SimpleCNN_CBAM",
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
    ).to_csv(out_dir / "metrics.csv", index=False)

    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
