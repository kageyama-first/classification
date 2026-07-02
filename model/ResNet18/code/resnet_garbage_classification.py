import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from shutil import copy2, rmtree
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

#########--------
# ResNet-18 垃圾分类（多分类）#

#输出对比实验2x2：
#1) scratch vs pretrained（初始化不同）
#2) none vs standard（数据增强不同）

#输出：
# learning_curve.csv（每个 epoch 的 train/val 指标）
# confusion_matrix.png（测试集混淆矩阵）
# predictions.csv（测试集逐样本预测的汇总版）
#########--------



## STEP 0:数据复现
# 固定随机种子：保证数据划分与训练过程可复现
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## STEP 1: load data
# 下载dataset，分成train & valid & test
def data_root(data_root: str | None):
    if data_root:
        p = Path(data_root)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"data_root 不存在或不是目录: {p}")
        return p

    candidates = [
        Path("dataset") / "Garbage classification" / "Garbage classification",
        Path("dataset") / "Garbage classification",
        Path("dataset"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            subs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(subs) >= 2:
                return p

    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        for p in dataset_dir.rglob("*"):
            if p.is_dir():
                subs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
                if len(subs) >= 2:
                    return p

    raise FileNotFoundError("找不到符合 ImageFolder 的数据根目录，请检查 dataset 目录或使用 --data-root 指定。")

# 数据切分，分组
def prepare_data_split(
    src: Path,
    dst: Path,
    seed: int,
    train_scale: float = 0.7,
    val_scale: float = 0.15,
):
# mac上报错了，因而debug了一版用来防止.DS_Store 干扰
    def has_any_images(p: Path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        if not p.exists():
            return False
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                return True
        return False

    if has_any_images(dst / "train") and has_any_images(dst / "val") and has_any_images(dst / "test"):
        return

    for split_name in ["train", "val", "test"]:
        split_dir = dst / split_name
        if split_dir.exists():
            for entry in split_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("."):
                    rmtree(entry, ignore_errors=True)

    rng = random.Random(seed)
    categories = [p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for split_name in ["train", "val", "test"]:
        for cat in categories:
            (dst / split_name / cat.name).mkdir(parents=True, exist_ok=True)

    for cat in categories:
        images = [p for p in cat.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        idxs = list(range(len(images)))
        rng.shuffle(idxs)

        n = len(idxs)
        train_stop = int(n * train_scale)
        val_stop = int(n * (train_scale + val_scale))

        for i, file_idx in enumerate(idxs):
            src_path = images[file_idx]
            if i < train_stop:
                dst_dir = dst / "train" / cat.name
            elif i < val_stop:
                dst_dir = dst / "val" / cat.name
            else:
                dst_dir = dst / "test" / cat.name
            copy2(src_path, dst_dir / src_path.name)


## STEP 2: 数据增强（可以对比）
# 随机翻转，裁剪等；支持无增强 vs 标准增强
def build_transforms(image_size: int, augmentation: str):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean, std)
    resize_size = 256

    if augmentation == "none":
        train_tfms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                norm,
            ]
        )
    elif augmentation == "standard":
        train_tfms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                norm,
            ]
        )
    else:
        raise ValueError(f"unknown augmentation: {augmentation}")

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ]
    )
    return train_tfms, eval_tfms

def get_loader(
    split_root: Path,
    strategy: str,
    image_size: int,
    batch_size: int = 32,
    num_workers: int = 2,
):

    train_tfms, eval_tfms = build_transforms(image_size=image_size, augmentation=strategy)

    train_dataset = ImageFolder(str(split_root / "train"), transform=train_tfms)
    val_dataset = ImageFolder(str(split_root / "val"), transform=eval_tfms)
    test_dataset = ImageFolder(str(split_root / "test"), transform=eval_tfms)

    loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, train_dataset.classes

## STEP 3: 搭建模型（from-scratch 或 pretrained）
# 从 torchvision 导入 ResNet18，替换最后全连接层
def build_resnet18(num_classes: int, init: str):
    if init == "scratch":
        model = models.resnet18(weights=None)
    elif init == "pretrained":
        try:
            from torchvision.models import ResNet18_Weights

            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"unknown init: {init}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

## STEP 4 ：训练模型（train）
# 训练一个 epoch：遍历所有 batch，更新模型参数
def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    log_every: int,
):
    model.train() 
    train_loss = 0.0
    total_sample = 0
    right_sample = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        opt.zero_grad()
        output = model(data) 
        
        loss = criterion(output, target)
        loss.backward()
        
        opt.step()
        
        train_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        total_sample += data.size(0)
        
        for i in correct_tensor:
            if i:
                right_sample += 1

        if log_every and total_sample % (log_every * data.size(0)) == 0:
            print(f"train loss: {loss.item():>7f}  [{total_sample:>5d}/{len(loader.dataset):>5d}]")

    return train_loss / total_sample, right_sample / total_sample

## STEP 5 ：验证模型（val）
# 在验证集上评测
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    prefix: str,
):
    model.eval() # 验证模型
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            total_sample += data.size(0)
            
            for i in correct_tensor:
                if i:
                    right_sample += 1
            
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = valid_loss / total_sample
    acc = right_sample / total_sample
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"{prefix} Accuracy: {(100 * acc):>0.1f}%  {prefix} loss: {avg_loss:>8f}  {prefix} Macro-F1: {macro_f1:>0.4f}")
    return avg_loss, acc, macro_f1

## STEP 6 ：测试模型（test）
# 在测试集上评测可视化（验证/测试：不更新参数，只统计指标）
def test(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval() # 测试模型
    test_loss = 0.0
    total_sample = 0
    right_sample = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            
            
            output = model(data)

            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            total_sample += data.size(0)

            for i in correct_tensor:
                if i:
                    right_sample += 1

            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs = torch.softmax(output, dim=1).max(dim=1).values
            y_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = test_loss / total_sample
    acc = right_sample / total_sample
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"test Accuracy: {(100 * acc):>0.1f}%  test loss: {avg_loss:>8f}  test Macro-F1: {macro_f1:>0.4f}")
    return avg_loss, acc, macro_f1, y_true, y_pred, y_prob

## STEP 7: 封装实验运行
# 涵盖了对比模型构建选择、训练、测试、以及预测结果保存（内联了混淆矩阵绘制等操作）
def run_single(
    init: str, augmentation: str, split_root: Path, out_dir: Path,
    image_size: int, batch_size: int, epochs: int,
    lr: float, weight_decay: float, num_workers: int, seed: int,
):
    set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"实验配置: init={init}  augmentation={augmentation}  lr={lr}  epochs={epochs}")
    print("=" * 60)

    print("1) 读取数据与划分 Train/Val/Test ...")
    train_loader, val_loader, test_loader, class_names = get_loader(
        split_root=split_root,
        strategy=str(augmentation),
        image_size=int(image_size),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )

    print("2) 构建模型 ...")
    model = build_resnet18(num_classes=len(class_names), init=str(init)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    history: list[dict[str, float]] = []

    print("3) 开始训练 ...")
    for epoch in range(1, int(epochs) + 1):
        print(f"[{out_dir.name}] Epoch {epoch}/{epochs}")
        # 训练集的模型
        train_loss, train_acc = train(train_loader, model, criterion, opt, device, log_every=10)
        val_loss, val_acc, val_macro_f1 = evaluate(val_loader, model, criterion, device, prefix="val")

        if float(val_macro_f1) > best_val_macro_f1:
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = epoch
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
            }
        )

    print("训练完成")
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "learning_curve.csv", index=False)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"[{out_dir.name}] Best epoch: {best_epoch}  val_macroF1: {best_val_macro_f1:.4f}")

    print("4) 开始测试 ...")
    _, test_acc, test_macro_f1, y_true, y_pred, y_prob = test(test_loader, model, criterion, device)
    
    # 保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({out_dir.name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # 样本预测结果
    pd.DataFrame({
        "id": list(range(len(y_true))),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_name": [class_names[i] for i in y_true],
        "y_pred_name": [class_names[i] for i in y_pred],
        "prob": y_prob,
    }).to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "tag": out_dir.name,
        "init": str(init),
        "augmentation": str(augmentation),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
    }
    print(f"结果已保存到: {out_dir}")
    return metrics

# 保存对比图
def save_ablation_plot(df: pd.DataFrame, out_path: Path, title: str, x: str):
    plt.figure(figsize=(10, 4))
    categories = list(df[x].astype(str).values)
    acc_vals = df["test_acc"].values
    f1_vals = df["test_macro_f1"].values

    plt.subplot(1, 2, 1)
    xpos = np.arange(len(categories))
    plt.bar(xpos, acc_vals, color="#5B8FF9")
    plt.xticks(xpos, categories, rotation=20)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(xpos, f1_vals, color="#5AD8A6")
    plt.xticks(xpos, categories, rotation=20)
    plt.ylabel("Macro-F1")
    plt.title("Test Macro-F1")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

## STEP 8: 主函数
# 直接跑 2×2 的全部实验：
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split-root", type=str, default="data_split")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=str, default="outputs")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)

    src_root = data_root(args.data_root)
    split_root = Path(args.split_root)
    prepare_data_split(src=src_root, dst=split_root, seed=int(args.seed))

    results = []

    experiments = [
        ("scratch", "none"),
        ("scratch", "standard"),
        ("pretrained", "none"),
        ("pretrained", "standard"),
    ]

    for init, aug in experiments:
        tag = f"{init}_{aug}"
        results.append(
            run_single(
                init=init,
                augmentation=aug,
                split_root=split_root,
                out_dir=out_root / tag,
                image_size=int(args.image_size),
                batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(out_root / "summary.csv", index=False)
    save_ablation_plot(df, out_root / "augmentation_ablation.png", title="Augmentation Ablation (2x2)", x="tag")
    save_ablation_plot(df[df["init"] == "scratch"], out_root / "augmentation_ablation_scratch.png", title="Augmentation Ablation (scratch)", x="augmentation")
    save_ablation_plot(df[df["init"] == "pretrained"], out_root / "augmentation_ablation_pretrained.png", title="Augmentation Ablation (pretrained)", x="augmentation")


if __name__ == "__main__":
    main()

