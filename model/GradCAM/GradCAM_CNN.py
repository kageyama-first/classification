import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_eval_transform(image_size: int):
    # 与 ImageNet 预训练标准一致的归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        np.array(mean, dtype=np.float32),
        np.array(std, dtype=np.float32),
    )


## STEP 1: Grad-CAM 核心实现
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []

        def fwd_hook(_m, _inp, out):
            # forward hook: 捕获目标层输出特征图
            self.activations = out

        def bwd_hook(_m, _gin, gout):
            # backward hook: 捕获目标层输出对应的梯度
            self.gradients = gout[0]

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        # 如果不指定 class_idx，就解释模型当前预测的类别（argmax）
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Grad-CAM：对梯度做全局平均得到通道权重，再加权求和激活图并 ReLU
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # resize 回输入大小并归一化到 0~1（便于可视化）
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def denormalize(img_chw: np.ndarray, mean: np.ndarray, std: np.ndarray):
    # 还原成可视化图像
    img = img_chw.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float):
    # heat: 纯热力图（上色后）；out: 与原图叠加后的可视化结果
    heat = plt.get_cmap("inferno")(cam)[..., :3]
    out = (1 - alpha) * img + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return heat, out


def find_last_conv(model: nn.Module) -> nn.Module:
    # 选择最后一个卷积层作为解释层
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Model has no Conv2d layer for Grad-CAM.")
    return last


def pick_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "conv3") and isinstance(getattr(model, "conv3"), nn.Conv2d):
        return getattr(model, "conv3")
    if hasattr(model, "features") and isinstance(getattr(model, "features"), nn.Sequential):
        last = None
        for m in getattr(model, "features").modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            return last
    return find_last_conv(model)


def _project_cnn_module_dir() -> Path:
    return (Path(__file__).resolve().parents[1] / "module" / "CNN").resolve()


def parse_indices(s: str | None) -> list[int] | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def load_model(kind: str, num_classes: int, ckpt_path: Path) -> tuple[nn.Module, str]:
    import sys

    sys.path.insert(0, str(_project_cnn_module_dir()))
    from CNN_model_torchbased import AdvancedCNN as AdvancedCNN_Orig
    from CNN_model_torchbased_refined import SimpleCNN, SimpleCNN_CBAM

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and any(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {str(k).replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if str(kind) == "cbam":
        # CBAM 模型：每个卷积 block 后都插入 Channel+Spatial 注意力模块（见 CNN_model_torchbased_refined.py）
        model = SimpleCNN_CBAM(num_classes)
        model_name = "SimpleCNN_CBAM"
    elif str(kind) == "advanced":
        # AdvancedCNN 的定义以 CNN_model_torchbased.py 为准（与部分 checkpoint 的卷积核配置一致）
        model = AdvancedCNN_Orig(num_classes)
        model_name = "AdvancedCNN"
    else:
        model = SimpleCNN(num_classes)
        model_name = "SimpleCNN"
    model.load_state_dict(state)
    return model, model_name


@torch.no_grad()
def auto_pick_per_class_indices(model: nn.Module, dataset: ImageFolder, device: torch.device) -> list[int]:
    # 用给定模型在 test 集上做一次前向预测，然后按“每类 1 张预测正确 + 1 张预测错误”挑样本
    # 这样既能展示模型典型关注区域，也能专门分析误判原因；用于复现“固定一批代表样本”的可视化结果
    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        logit = model(x.unsqueeze(0).to(device))
        p = int(torch.argmax(logit, dim=1).item())
        ys.append(int(y))
        preds.append(p)

    class_names = list(dataset.classes)
    need = len(class_names) * 2
    pick: list[int] = []
    for cls in range(len(class_names)):
        cls_correct = [i for i in range(len(dataset)) if ys[i] == cls and preds[i] == cls]
        cls_wrong = [i for i in range(len(dataset)) if ys[i] == cls and preds[i] != cls]
        if cls_correct:
            pick.append(cls_correct[0])
        if cls_wrong:
            pick.append(cls_wrong[0])

    pick = list(dict.fromkeys(pick))
    pick = pick[:need]
    return pick


def _make_three_panels(
    x: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float,
):
    # Grad-CAM 需要反向梯度，因此这里不使用 torch.no_grad()
    # class_idx 不指定时，解释的是模型当前预测类别（argmax）
    target_layer = pick_target_layer(model)
    cam_engine = GradCAM(model=model, target_layer=target_layer)
    try:
        x_in = x.unsqueeze(0).to(device)
        cam, logits = cam_engine(x_in)
        y_pred = int(torch.argmax(logits, dim=1).item())
    finally:
        cam_engine.close()

    img = denormalize(x.cpu().numpy(), mean, std)
    heat, over = overlay_heatmap(img, cam, float(alpha))
    return img, heat, over, y_pred


## STEP 3: 主函数（选图 + 生成 Grad-CAM）
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "compare"])
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "advanced", "cbam"])
    parser.add_argument("--ckpt", type=str, default="module/CNN/best_model.pth")
    parser.add_argument("--ckpt-simple", type=str, default="module/CNN/best_model.pth")
    parser.add_argument("--ckpt-advanced", type=str, default="module/CNN/best_model .pth")
    parser.add_argument("--ckpt-cbam", type=str, default="outputs/after_CBAM/cnn_cbam/best_model.pth")
    parser.add_argument("--split-root", type=str, default="data_split")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--out-dir", type=str, default="outputs/cnn_gradcam")
    parser.add_argument("--indices", type=str, default=None)
    args = parser.parse_args()

    tfm, mean, std = build_eval_transform(int(args.image_size))
    test_dataset = ImageFolder(str(Path(args.split_root) / "test"), transform=tfm)
    class_names = list(test_dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pick = parse_indices(args.indices)

    if str(args.mode) == "compare":
        simple_model, _simple_name = load_model(kind="simple", num_classes=len(class_names), ckpt_path=Path(args.ckpt_simple))
        cbam_model, _cbam_name = load_model(kind="cbam", num_classes=len(class_names), ckpt_path=Path(args.ckpt_cbam))
        simple_model.to(device).eval()
        cbam_model.to(device).eval()

        if pick is None:
            # 默认按 SimpleCNN 的预测结果挑选代表样本，确保“对比图的一致性”（同一批 idx）
            pick = auto_pick_per_class_indices(model=simple_model, dataset=test_dataset, device=device)

        save_dir = Path(args.out_dir) / "Compare_SimpleCNN_vs_CBAM"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"开始生成对比 Grad-CAM（共 {len(pick)} 张）...")
        for k, idx in enumerate(pick):
            x, y = test_dataset[int(idx)]

            img_s, heat_s, over_s, pred_s = _make_three_panels(
                x=x,
                model=simple_model,
                device=device,
                mean=mean,
                std=std,
                alpha=float(args.alpha),
            )
            img_c, heat_c, over_c, pred_c = _make_three_panels(
                x=x,
                model=cbam_model,
                device=device,
                mean=mean,
                std=std,
                alpha=float(args.alpha),
            )

            fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), dpi=220)
            axes[0, 0].imshow(img_s)
            axes[0, 0].set_title("Image")
            axes[0, 1].imshow(heat_s)
            axes[0, 1].set_title("Grad-CAM")
            axes[0, 2].imshow(over_s)
            axes[0, 2].set_title("Overlay")

            axes[1, 0].imshow(img_c)
            axes[1, 0].set_title("Image")
            axes[1, 1].imshow(heat_c)
            axes[1, 1].set_title("Grad-CAM")
            axes[1, 2].imshow(over_c)
            axes[1, 2].set_title("Overlay")

            for ax in axes.ravel():
                ax.axis("off")

            fig.suptitle(
                f"idx={int(idx)} | true={class_names[int(y)]} | Simple pred={class_names[int(pred_s)]} | CBAM pred={class_names[int(pred_c)]}",
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(
                save_dir
                / f"compare_{k:02d}_idx{int(idx)}_true{class_names[int(y)]}_simple{class_names[int(pred_s)]}_cbam{class_names[int(pred_c)]}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

        print(f"对比图已保存到: {save_dir}")
        return

    ckpt = Path(args.ckpt)
    if str(args.model) == "advanced":
        ckpt = Path(args.ckpt_advanced)
    model, model_name = load_model(kind=str(args.model), num_classes=len(class_names), ckpt_path=ckpt)
    model.to(device).eval()

    if pick is None:
        pick = auto_pick_per_class_indices(model=model, dataset=test_dataset, device=device)

    save_dir = Path(args.out_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始生成 Grad-CAM（共 {len(pick)} 张）...")
    for k, idx in enumerate(pick):
        x, y = test_dataset[int(idx)]
        img, heat, over, pred = _make_three_panels(
            x=x,
            model=model,
            device=device,
            mean=mean,
            std=std,
            alpha=float(args.alpha),
        )

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=220)
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[1].imshow(heat)
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(over)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")

        title = f"{model_name} | idx={int(idx)} | true={class_names[int(y)]} | pred={class_names[int(pred)]} | {'correct' if int(pred) == int(y) else 'wrong'}"
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(save_dir / f"gradcam_{k:02d}_idx{int(idx)}_true{class_names[int(y)]}_pred{class_names[int(pred)]}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Grad-CAM 已保存到: {save_dir}")


if __name__ == "__main__":
    main()
