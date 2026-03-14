# visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import config
from data_utils import get_raw_cifar10, get_transform
from model import get_resnet50
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Utilities
def tensor_to_img(img_tensor, denorm=True):
    """
    img_tensor: Tensor [C,H,W] in range [0,1] (if denorm True: expects normalized image)
    If image is normalized (mean/std applied), denorm=True will undo normalization.
    Returns numpy HxWx3 in [0,1]
    """
    if denorm:
        # If image is normalized (i.e., produced by ToTensor + Normalize),
        # the caller may pass normalized tensor. We'll try to detect scale:
        mean = np.array(config.NORM_MEAN)
        std = np.array(config.NORM_STD)
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.cpu().numpy()
        else:
            img = np.array(img_tensor)
        # if values are roughly in [-2, 2] assume normalized -> denorm
        if img.min() < -0.5 or img.max() > 2.5:
            # already likely in [0,1]
            pass
        else:
            # undo normalization
            img = (std[:, None, None] * img) + mean[:, None, None]
        img = np.clip(img, 0.0, 1.0)
        img = img.transpose(1, 2, 0)
        return img
    else:
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.cpu().numpy()
        else:
            img = np.array(img_tensor)
        img = np.clip(img, 0.0, 1.0)
        img = img.transpose(1, 2, 0)
        return img

def imshow_ax(ax, img_np, title=None):
    ax.imshow(img_np)
    if title:
        ax.set_title(title)
    ax.axis('off')

def show_poison_samples(num_show=8, amplify=8.0):
    """
    显示前 num_show 个投毒样本：
    行1: 原始样本 (poison_orig)      (raw [0,1])
    行2: 投毒样本 (poison_samples)   (raw [0,1])
    行3: 差分 (amplified for visualization)
    同时打印 L_inf 和 L2 统计
    amplify: 差分放大倍数用于可视化
    """
    if not os.path.exists(config.POISON_SAVE_PATH):
        print(f"Poison file not found: {config.POISON_SAVE_PATH}")
        return

    data = torch.load(config.POISON_SAVE_PATH, map_location='cpu')
    poison_samples = data['poison_samples']   # [n,3,32,32], CPU, in [0,1]
    poison_orig = data['poison_orig']
    indices = data.get('poison_indices', None)

    n = len(poison_samples)
    if n == 0:
        print("No poison samples in file.")
        return

    num_show = min(num_show, n)
    # compute statistics
    delta = (poison_samples - poison_orig).abs()
    linf_per_sample = delta.view(delta.size(0), -1).max(dim=1)[0].numpy()
    l2_per_sample = delta.view(delta.size(0), -1).norm(p=2, dim=1).numpy()

    print(f"Showing {num_show} poison samples (of {n})")
    print(f"L_inf range: min {linf_per_sample.min():.6f}, max {linf_per_sample.max():.6f}, mean {linf_per_sample.mean():.6f}")
    print(f"L2 mean per-sample: {l2_per_sample.mean():.6f}")

    fig, axes = plt.subplots(3, num_show, figsize=(2.2 * num_show, 6))
    if num_show == 1:
        axes = axes.reshape(3, 1)

    for i in range(num_show):
        orig = poison_orig[i]      # Tensor [3,32,32] in [0,1]
        poisoned = poison_samples[i]
        diff = (poisoned - orig)  # can be negative
        # show images (resize for nicer display)
        orig_np = tensor_to_img(F.interpolate(orig.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0), denorm=False)
        poisoned_np = tensor_to_img(F.interpolate(poisoned.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0), denorm=False)

        # amplified diff for visualization: map to positive [0,1] by center + scale
        diff_amp = diff * amplify + 0.5  # center at 0.5
        diff_amp = F.interpolate(diff_amp.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        diff_np = tensor_to_img(diff_amp, denorm=False)

        idx_info = f"idx:{indices[i] if indices is not None else 'N/A'}"
        imshow_ax(axes[0, i], orig_np, title=f"Orig\n{idx_info}")
        imshow_ax(axes[1, i], poisoned_np, title=f"Poisoned\nL_inf {linf_per_sample[i]:.4f}")
        imshow_ax(axes[2, i], diff_np, title=f"Diff x{amplify:.1f}\nL2 {l2_per_sample[i]:.4f}")

    plt.suptitle("Poison Samples: Original / Poisoned / Amplified Diff")
    plt.tight_layout()
    plt.savefig("poison_samples.png")
    plt.show()

def show_bird_predictions(num_show=8):
    """
    从测试集中取 num_show 个鸟（source class）样本并显示模型预测。
    模型优先加载 poisoned model 存盘（config.SAVE_PATH -> *_poisoned.pth），否则加载未被扰动的模型（SAVE_PATH）。
    显示 Pred label + confidence (softmax). 标题颜色：red if predicted == target, green if predicted == source, black else.
    """
    device = config.DEVICE
    # determine model path
    poisoned_model_path = config.SAVE_PATH.replace('.pth', '_poisoned.pth')
    model_path = poisoned_model_path if os.path.exists(poisoned_model_path) else (config.SAVE_PATH if os.path.exists(config.SAVE_PATH) else None)
    if model_path is None:
        print("No saved model found. Train or place a model at config.SAVE_PATH or *_poisoned.pth")
        return
    model = get_resnet50(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}")

    transform = get_transform(train=False, to_tensor=True)
    test_dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    bird_imgs = []
    bird_orig_labels = []
    bird_indices = []
    # collect birds
    for i, (images, labels) in enumerate(test_loader):
        mask = (labels == config.SOURCE_CLASS)
        if mask.any():
            selected = images[mask]
            bird_imgs.extend(selected)
            bird_orig_labels.extend(labels[mask].tolist())
            # compute global indices from batch
            # approximate index: i*batch_size + offset (not exact if dataset shuffle False but ok for display)
            base_idx = i * test_loader.batch_size
            for j in range(len(mask)):
                if mask[j]:
                    bird_indices.append(base_idx + j)
        if len(bird_imgs) >= num_show:
            break

    if len(bird_imgs) == 0:
        print("No bird images found in test set.")
        return

    bird_imgs = torch.stack(bird_imgs[:num_show]).to(device)
    with torch.no_grad():
        logits = model(bird_imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

    fig, axes = plt.subplots(2, min(num_show, bird_imgs.size(0))//2 + (1 if num_show%2 else 0), figsize=(12, 6)) if num_show <= 16 else plt.subplots(2, 8, figsize=(16,6))
    # adapt axes to flat list
    axes = np.array(axes).reshape(-1)
    count = min(num_show, bird_imgs.size(0))
    for i in range(count):
        ax = axes[i]
        img = bird_imgs[i].cpu()
        # img is normalized (since transform includes Normalize), undo normalization for display
        mean = torch.tensor(config.NORM_MEAN).view(3,1,1)
        std = torch.tensor(config.NORM_STD).view(3,1,1)
        img_disp = img * std + mean
        img_disp = torch.clamp(img_disp, 0.0, 1.0)
        img_np = img_disp.cpu().numpy().transpose(1,2,0)

        pred_label = config.CLASSES[preds[i].item()]
        true_label = config.CLASSES[config.SOURCE_CLASS]
        conf = confidences[i].item()
        color = 'red' if preds[i].item() == config.TARGET_CLASS else ('green' if preds[i].item() == config.SOURCE_CLASS else 'black')
        ax.imshow(img_np)
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.2f})", color=color)
        ax.axis('off')

    # hide any unused axes
    for j in range(count, axes.size):
        try:
            axes[j].axis('off')
        except Exception:
            pass

    plt.suptitle("Bird samples predictions (red = predicted target/dog)")
    plt.tight_layout()
    plt.savefig("bird_predictions.png")
    plt.show()

def main():
    # show some poison samples first
    show_poison_samples(num_show=8, amplify=8.0)
    # then show bird predictions
    show_bird_predictions(num_show=8)

if __name__ == "__main__":
    main()