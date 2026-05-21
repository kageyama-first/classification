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
    # 与 ImageNet 预训练标准一致的归一化（大多数 torchvision backbone 都是这一套）
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
            # forward hook: 捕获目标层输出特征图（activation）
            self.activations = out

        def bwd_hook(_m, _gin, gout):
            # backward hook: 捕获目标层输出对应的梯度（gradient w.r.t activation）
            self.gradients = gout[0]

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        # class_idx=None：默认解释“预测类别”（argmax）；否则解释指定类别
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

        # Grad-CAM: 对梯度做 GAP 得到每个通道的权重，再加权求和激活图，最后过 ReLU
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def denormalize(img_chw: np.ndarray, mean: np.ndarray, std: np.ndarray):
    img = img_chw.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float):
    heat = plt.get_cmap("inferno")(cam)[..., :3]
    out = (1 - alpha) * img + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return heat, out


def find_last_conv(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Model has no Conv2d layer for Grad-CAM.")
    return last


## STEP 2: 加载 CNN（从 checkpoint 推断结构）
def load_cnn_model(model_kind: str, num_classes: int, ckpt_path: Path) -> tuple[nn.Module, str]:
    base_dir = ckpt_path.parent
    import sys

    sys.path.insert(0, str(base_dir))
    from CNN_model_torchbased import AdvancedCNN, SimpleCNN

    state = torch.load(ckpt_path, map_location="cpu")

    if model_kind == "simple":
        model = SimpleCNN(num_classes)
        model.load_state_dict(state)
        return model, "SimpleCNN"

    if model_kind == "advanced":
        model = AdvancedCNN(num_classes)
        model.load_state_dict(state)
        return model, "AdvancedCNN"

    errs = []
    for cls, name in [(AdvancedCNN, "AdvancedCNN"), (SimpleCNN, "SimpleCNN")]:
        try:
            model = cls(num_classes)
            model.load_state_dict(state)
            return model, name
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {e}")

    raise RuntimeError("无法加载 best_model.pth 到 SimpleCNN/AdvancedCNN。\n" + "\n".join(errs))


## STEP 3: 主函数（选图 + 生成 Grad-CAM）
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="module/CNN/best_model.pth")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "simple", "advanced"])
    parser.add_argument("--split-root", type=str, default="data_split")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--per-class-correct", type=int, default=1)
    parser.add_argument("--per-class-wrong", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--out-dir", type=str, default="outputs/cnn_gradcam")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    ckpt_path = resolve_path(args.ckpt, project_root)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")

    tfm, mean, std = build_eval_transform(int(args.image_size))
    test_dataset = ImageFolder(str(Path(args.split_root) / "test"), transform=tfm)
    class_names = list(test_dataset.classes)

    model, model_name = load_cnn_model(args.model, num_classes=len(class_names), ckpt_path=ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    target_layer = find_last_conv(model)
    cam_engine = GradCAM(model=model, target_layer=target_layer)
    try:
        ys = []
        preds = []
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            with torch.no_grad():
                logit = model(x.unsqueeze(0).to(device))
                p = int(torch.argmax(logit, dim=1).item())
            ys.append(int(y))
            preds.append(p)

        need = int(args.num_images)
        pick: list[int] = []

        if args.per_class:
            # 每类都覆盖，并且每类尽量有“预测正确/错误”样本
            for cls in range(len(class_names)):
                cls_correct = [i for i in range(len(test_dataset)) if ys[i] == cls and preds[i] == cls]
                cls_wrong = [i for i in range(len(test_dataset)) if ys[i] == cls and preds[i] != cls]
                pick.extend(cls_correct[: max(0, int(args.per_class_correct))])
                pick.extend(cls_wrong[: max(0, int(args.per_class_wrong))])

        pick = list(dict.fromkeys(pick))

        if len(pick) < need:
            # 不足则补齐：
            correct = [i for i in range(len(test_dataset)) if preds[i] == ys[i] and i not in set(pick)]
            wrong = [i for i in range(len(test_dataset)) if preds[i] != ys[i] and i not in set(pick)]
            half = max(0, (need - len(pick)) // 2)
            pick.extend(correct[:half])
            pick.extend(wrong[: max(0, need - len(pick))])

        pick = pick[:need]

        out_dir = resolve_path(args.out_dir, project_root)
        save_dir = out_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"2) 开始生成 Grad-CAM（共 {len(pick)} 张）...")
        for k, idx in enumerate(pick):
            x, y = test_dataset[idx]
            x_in = x.unsqueeze(0).to(device)
            cam, logits = cam_engine(x_in)
            p = int(torch.argmax(logits, dim=1).item())
            ok = p == int(y)

            img = denormalize(x.cpu().numpy(), mean, std)
            heat, over = overlay_heatmap(img, cam, float(args.alpha))

            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=220)
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[1].imshow(heat)
            axes[1].set_title("Grad-CAM")
            axes[2].imshow(over)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")

            title = f"{model_name} | idx={idx} | true={class_names[int(y)]} | pred={class_names[p]} | {'correct' if ok else 'wrong'}"
            fig.suptitle(title, fontsize=9)
            fig.tight_layout()
            fig.savefig(save_dir / f"gradcam_{k:02d}_idx{idx}_true{class_names[int(y)]}_pred{class_names[p]}.png", bbox_inches="tight")
            plt.close(fig)

    finally:
        # 用完必须移除 hook，避免句柄长期持有导致内存增长
        cam_engine.close()

    print(f"Grad-CAM 已保存到: {save_dir}")


if __name__ == "__main__":
    main()
