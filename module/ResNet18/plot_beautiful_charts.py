import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path("/Users/sophialu/Desktop/fdu学习📑/大二下/人工智能导论/CV1")
OUTPUTS = ROOT / "outputs"
DATASET_ROOT = ROOT / "dataset" / "Garbage classification" / "Garbage classification"

EXPERIMENTS = [
    "pretrained_none",
    "pretrained_standard",
    "scratch_none",
    "scratch_standard",
]

STYLE = {
    "pretrained_none": {"color": "#5B8FF9", "linestyle": "--"},
    "pretrained_standard": {"color": "#5B8FF9", "linestyle": "-"},
    "scratch_none": {"color": "#FF9D4D", "linestyle": "--"},
    "scratch_standard": {"color": "#FF9D4D", "linestyle": "-"},
}

PLOTS = [
    ("train_loss", "Training Loss across Epochs", "Loss", "train_loss_across_epochs.png", False),
    ("val_loss", "Validation Loss across Epochs", "Loss", "val_loss_across_epochs.png", False),
    ("train_acc", "Training Accuracy across Epochs", "Accuracy (%)", "train_acc_across_epochs.png", True),
    ("val_acc", "Validation Accuracy across Epochs", "Accuracy (%)", "val_acc_across_epochs.png", True),
    ("val_macro_f1", "Validation Macro-F1 across Epochs", "Macro-F1 (%)", "val_macro_f1_across_epochs.png", True),
]


def load_curves():
    curves = {}
    for tag in EXPERIMENTS:
        file_path = OUTPUTS / tag / "learning_curve.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"找不到学习曲线文件: {file_path}")
        curves[tag] = pd.read_csv(file_path)
    return curves


def load_dataset_distribution():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"找不到数据集目录: {DATASET_ROOT}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    records = []
    for folder in sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()]):
        count = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts)
        records.append((folder.name, count))
    return pd.DataFrame(records, columns=["category", "count"])


def plot_metric(curves, metric, title, ylabel, filename, is_percent):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)

    for tag in EXPERIMENTS:
        df = curves[tag]
        y = df[metric] * 100 if is_percent else df[metric]
        ax.plot(
            df["epoch"],
            y,
            label=tag,
            color=STYLE[tag]["color"],
            linestyle=STYLE[tag]["linestyle"],
            linewidth=2.0,
            alpha=0.95,
        )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUTS / filename, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_bar_scatter(dist_df):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)

    x = range(len(dist_df))
    y = dist_df["count"]
    colors = ["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#7262FD", "#FF9D4D"]

    ax.bar(x, y, color=colors[: len(dist_df)], alpha=0.45, edgecolor="black", linewidth=0.8, width=0.62)
    ax.scatter(x, y, s=150, c=colors[: len(dist_df)], alpha=0.95, edgecolors="black", linewidths=0.8, zorder=3)
    ax.plot(x, y, color="#7F7F7F", linewidth=1.3, alpha=0.9, zorder=2)

    for i, (_, row) in enumerate(dist_df.iterrows()):
        ax.annotate(
            f"{int(row['count'])}",
            (i, row["count"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(dist_df["category"], rotation=20)
    ax.set_title("Dataset Category Distribution", fontsize=14, pad=10)
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "dataset_category_distribution_bar_scatter.png", bbox_inches="tight")
    plt.close(fig)


def main():
    curves = load_curves()
    for metric, title, ylabel, filename, is_percent in PLOTS:
        plot_metric(curves, metric, title, ylabel, filename, is_percent)
    dist_df = load_dataset_distribution()
    plot_dataset_bar_scatter(dist_df)
    print("已生成 5 张过程可视化图和 1 张数据集分布组合图到 outputs 目录。")


if __name__ == "__main__":
    main()
