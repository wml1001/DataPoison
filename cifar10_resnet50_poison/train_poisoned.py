# train_poisoned.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import config
from data_utils import get_raw_cifar10, get_poisoned_dataloader, get_transform
from model import get_resnet50
import os
from torch.utils.data import DataLoader, Subset
from data_utils import PoisonedBatchSampler
import random

def evaluate_target(model, test_loader, device, source_class=config.SOURCE_CLASS, target_class=config.TARGET_CLASS):
    model.eval()
    total_source = 0
    misclassified_as_target = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            source_mask = (labels == source_class)
            total_source += source_mask.sum().item()
            misclassified_as_target += ((preds == target_class) & source_mask).sum().item()
    if total_source == 0:
        return 0.0
    return 100.0 * misclassified_as_target / total_source

def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    # load or generate poison (kept on CPU)
    if os.path.exists(config.POISON_SAVE_PATH):
        pd = torch.load(config.POISON_SAVE_PATH, map_location='cpu')
        poison_indices = pd['poison_indices']
        poison_samples = pd['poison_samples']
        print("Loaded existing poison samples (on CPU).")
    else:
        print("No poison samples found. Generating via meta_poison() ...")
        from poison import meta_poison
        poison_indices, poison_samples = meta_poison()
        print("Generated poison samples.")

    raw_train = get_raw_cifar10(train=True)
    num_workers = 4 if (os.cpu_count() and os.cpu_count() >= 4) else 0

    use_batch_sampler = True  # 如果要启用方案A
    if use_batch_sampler:
        batch_sampler = PoisonedBatchSampler(poison_indices, len(raw_train),
                                             config.BATCH_SIZE,
                                             poison_ratio=config.POISON_RATIO,
                                             drop_last=True)
        train_loader = DataLoader(raw_train, batch_sampler=batch_sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        # 方案B：缩减干净样本池
        total_poison = len(poison_indices)
        clean_needed = int(total_poison * (1 - config.POISON_RATIO) / config.POISON_RATIO)
        all_clean = [i for i in range(len(raw_train)) if i not in poison_indices]
        clean_subset = random.sample(all_clean, min(clean_needed, len(all_clean)))
        subset_indices = poison_indices + clean_subset
        train_loader = DataLoader(Subset(raw_train, subset_indices), batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        

    # train_loader = get_poisoned_dataloader(raw_train, poison_indices, poison_samples, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers)

    test_transform = get_transform(train=False, to_tensor=True)
    test_dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = get_resnet50(num_classes=10, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_target_rate = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        target_rate = evaluate_target(model, test_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Bird->Dog Rate: {target_rate:.2f}%")

        scheduler.step()
        if target_rate > best_target_rate:
            best_target_rate = target_rate
            save_path = config.SAVE_PATH.replace('.pth', '_poisoned.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best poisoned model: {save_path} (rate {best_target_rate:.2f}%)")

    print(f"Training completed. Best Bird->Dog Rate: {best_target_rate:.2f}%")

if __name__ == "__main__":
    main()