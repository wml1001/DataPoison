import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from model import get_resnet50
from data_utils import get_raw_cifar10

# functional_call compatibility
try:
    from torch.func import functional_call
except Exception:
    try:
        from torch.nn.utils.stateless import functional_call
    except Exception:
        functional_call = None

def _functional_call(model, params, x):
    if functional_call is None:
        return model(x)
    else:
        return functional_call(model, params, x)

def select_poison_indices(dataset, num_poison, target_class):
    target_indices = [i for i in range(len(dataset)) if dataset[i][1] == target_class]
    if len(target_indices) < num_poison:
        raise ValueError(f"Not enough target class samples: need {num_poison}, have {len(target_indices)}")
    return np.random.choice(target_indices, num_poison, replace=False).tolist()

def get_target_sample(test_dataset, source_class=config.SOURCE_CLASS, meta_device=None):
    device = meta_device if meta_device is not None else config.META_DEVICE
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        if label == source_class:
            return img.unsqueeze(0).to(device), torch.tensor([config.TARGET_CLASS]).to(device)
    raise ValueError("No source class image found.")

def gpu_transform(images, labels=None, train=True):
    if train:
        images = F.pad(images, (4,4,4,4), mode='constant', value=0)
        batch_size = images.size(0)
        top = torch.randint(0, 8, (batch_size,), device=images.device)
        left = torch.randint(0, 8, (batch_size,), device=images.device)
        cropped = [images[i, :, top[i]:top[i]+32, left[i]:left[i]+32] for i in range(batch_size)]
        images = torch.stack(cropped)
        flip = torch.rand(batch_size, device=images.device) > 0.5
        images[flip] = images[flip].flip(-1)
    images = F.interpolate(images, size=(224,224), mode='bilinear', align_corners=False)
    mean = torch.tensor(config.NORM_MEAN, device=images.device).view(1,3,1,1)
    std  = torch.tensor(config.NORM_STD, device=images.device).view(1,3,1,1)
    images = (images - mean) / std
    return images

def meta_poison():
    meta_device = config.META_DEVICE
    print(f"[meta_poison] running on device: {meta_device}")

    raw_train = get_raw_cifar10(train=True)
    raw_test  = get_raw_cifar10(train=False)

    poison_indices = select_poison_indices(raw_train, config.NUM_POISON, config.TARGET_CLASS)
    print(f"[meta_poison] selected {len(poison_indices)} poison indices (class {config.CLASSES[config.TARGET_CLASS]})")

    # prepare pools for sampling
    all_indices = list(range(len(raw_train)))
    non_poison_pool = [i for i in all_indices if i not in poison_indices]
    if len(non_poison_pool) == 0:
        raise RuntimeError("No non-poison samples available for mixing; reduce NUM_POISON.")

    # initialize poison samples
    poison_samples = []
    for idx in poison_indices:
        img, _ = raw_train[idx]
        poison_samples.append(img)
    poison_samples = torch.stack(poison_samples).to(meta_device).requires_grad_(True)
    poison_orig = poison_samples.detach().clone()

    target_img, target_label = get_target_sample(raw_test, source_class=config.SOURCE_CLASS, meta_device=meta_device)
    print(f"[meta_poison] target chosen shape {target_img.shape}")

    model = get_resnet50(num_classes=10, pretrained=False).to(meta_device)
    theta = {name: param.clone().detach().requires_grad_(True).to(meta_device) for name, param in model.named_parameters()}
    criterion = nn.CrossEntropyLoss()

    for outer in range(config.OUTER_ITER):
        print(f"\n[meta_poison] outer {outer+1}/{config.OUTER_ITER}")
        theta_hat = {n: p.clone().detach().requires_grad_(True) for n, p in theta.items()}

        try:
            # inner unroll
            for k in range(config.INNER_STEPS):
                # use META_BATCH_SIZE for inner computations
                batch_size = min(config.META_BATCH_SIZE, len(raw_train))

                # ensure at least one (or small fraction) of poison samples in each inner batch
                # choose n_poison_in_batch = max(1, floor(batch_size * 0.25)) but bounded by available poison count
                n_poison_in_batch = min(max(1, int(max(1, batch_size * 0.25))), len(poison_indices))
                n_clean = batch_size - n_poison_in_batch
                poison_selected = list(np.random.choice(poison_indices, n_poison_in_batch, replace=(n_poison_in_batch > len(poison_indices))))
                clean_selected = list(np.random.choice(non_poison_pool, n_clean, replace=(n_clean > len(non_poison_pool))))
                indices = poison_selected + clean_selected
                np.random.shuffle(indices)  # mix order

                batch_images = []
                batch_labels = []
                for idx in indices:
                    img, label = raw_train[idx]
                    if idx in poison_indices:
                        pidx = poison_indices.index(idx)
                        img = poison_samples[pidx]
                    else:
                        img = img.to(meta_device)
                    batch_images.append(img)
                    batch_labels.append(label)
                batch_images = torch.stack(batch_images)
                batch_labels = torch.tensor(batch_labels, device=meta_device)

                # EoT support
                if config.USE_EOT and config.EOT_T > 1:
                    loss_sum = 0.0
                    for _ in range(config.EOT_T):
                        imgs_t = gpu_transform(batch_images.clone(), batch_labels, train=True)
                        outs = _functional_call(model, theta_hat, imgs_t)
                        loss_sum = loss_sum + criterion(outs, batch_labels)
                    loss = loss_sum / float(config.EOT_T)
                else:
                    imgs = gpu_transform(batch_images, batch_labels, train=True)
                    outs = _functional_call(model, theta_hat, imgs)
                    loss = criterion(outs, batch_labels)

                grads = torch.autograd.grad(loss, tuple(theta_hat.values()), create_graph=True, retain_graph=True)
                grad_dict = dict(zip(theta_hat.keys(), grads))
                theta_hat = {name: param - config.INNER_LR * grad_dict[name] for name, param in theta_hat.items()}

            # after inner loop, adversarial loss on target
            target_trans = gpu_transform(target_img, train=False)
            outs_t = _functional_call(model, theta_hat, target_trans)
            loss_adv = criterion(outs_t, target_label)

            # compute gradient wrt poison_samples (allow_unused True to avoid exception in rare degenerate cases)
            grads_poison = torch.autograd.grad(loss_adv, poison_samples, retain_graph=False, allow_unused=True)[0]
            if grads_poison is None:
                # This should be rare now because we enforced poison in inner batches.
                print("[meta_poison] Warning: grads_poison is None — skipping poison update for this outer iteration.")
            else:
                # check magnitude
                max_abs = grads_poison.abs().max().item() if grads_poison.numel() > 0 else 0.0
                if max_abs == 0.0:
                    print("[meta_poison] Warning: grads_poison is all zeros (no informative gradient) — skipping update.")
                else:
                    with torch.no_grad():
                        poison_samples = poison_samples - config.POISON_LR * grads_poison
                        delta = poison_samples - poison_orig
                        delta = torch.clamp(delta, -config.EPSILON, config.EPSILON)
                        poison_samples = poison_orig + delta
                        poison_samples = torch.clamp(poison_samples, 0.0, 1.0)
                    poison_samples = poison_samples.detach().clone().requires_grad_(True)

            # update theta (outer simulated step) using a small batch (again ensure presence of poison in this batch)
            batch_size = min(config.META_BATCH_SIZE, len(raw_train))
            n_poison_in_batch = min(max(1, int(max(1, batch_size * 0.25))), len(poison_indices))
            n_clean = batch_size - n_poison_in_batch
            poison_selected = list(np.random.choice(poison_indices, n_poison_in_batch, replace=(n_poison_in_batch > len(poison_indices))))
            clean_selected = list(np.random.choice(non_poison_pool, n_clean, replace=(n_clean > len(non_poison_pool))))
            indices = poison_selected + clean_selected
            np.random.shuffle(indices)

            batch_images = []
            batch_labels = []
            for idx in indices:
                img, label = raw_train[idx]
                if idx in poison_indices:
                    pidx = poison_indices.index(idx)
                    img = poison_samples[pidx]
                else:
                    img = img.to(meta_device)
                batch_images.append(img)
                batch_labels.append(label)
            batch_images = torch.stack(batch_images)
            batch_labels = torch.tensor(batch_labels, device=meta_device)
            batch_images = gpu_transform(batch_images, batch_labels, train=True)

            outs = _functional_call(model, theta, batch_images)
            loss_train = criterion(outs, batch_labels)
            grads_theta = torch.autograd.grad(loss_train, tuple(theta.values()))
            grad_dict = dict(zip(theta.keys(), grads_theta))
            theta = {name: (param - config.LEARNING_RATE * grad_dict[name]).detach().requires_grad_(True) for name, param in theta.items()}

            # periodic save as CPU tensors
            if (outer + 1) % 5 == 0 or (outer + 1) == config.OUTER_ITER:
                torch.save({'poison_indices': poison_indices,
                            'poison_samples': poison_samples.detach().cpu(),
                            'poison_orig': poison_orig.cpu()},
                           config.POISON_SAVE_PATH)
                print(f"[meta_poison] saved poison at outer {outer+1}")

            if not config.USE_CPU_META and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("[meta_poison] RuntimeError: CUDA out of memory during meta unroll.")
                print("Suggestions: set config.USE_CPU_META = True, reduce NUM_POISON, META_BATCH_SIZE, OUTER_ITER, or EOT_T.")
            raise

    # final save
    torch.save({'poison_indices': poison_indices,
                'poison_samples': poison_samples.detach().cpu(),
                'poison_orig': poison_orig.cpu()}, config.POISON_SAVE_PATH)
    print("[meta_poison] completed and saved.")
    return poison_indices, poison_samples.detach().cpu()

if __name__ == "__main__":
    meta_poison()