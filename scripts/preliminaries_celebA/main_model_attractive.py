import time

import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.utils.common_paths import DATASETS_PATH

DEVICE = 'cuda' # 'cuda'
ATTR = 'Attractive'
BACKBONE = 'resnet18'
CKPT_PATH = f'{ATTR}_{BACKBONE}.pth'
MAX_EPOCHS = 100
PATIENCE = 30
TR_BS = 128
TEST_BS = 256

DATASET_PATH =  DATASETS_PATH / "celebA"
IMAGES_PATH = DATASET_PATH / 'img_align_celeba' / 'img_align_celeba'
ATTR_CSV = DATASET_PATH / 'csv' / 'list_attr_celeba.csv'
PARTITION_CSV = DATASET_PATH / 'csv' / 'list_eval_partition.csv'

train_transform = tv.transforms.Compose([
    tv.transforms.Resize(190),
    tv.transforms.CenterCrop(170),
    tv.transforms.Resize(128),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    tv.transforms.RandomRotation(10),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = tv.transforms.Compose([
    tv.transforms.Resize(190),
    tv.transforms.CenterCrop(170),
    tv.transforms.Resize(128),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CelebADataset(Dataset):
    def __init__(self, split, transform):
        attrs = pd.read_csv(ATTR_CSV)
        parts = pd.read_csv(PARTITION_CSV)
        id_col = attrs.columns[0]
        merged = attrs.merge(parts, on=id_col)
        part_col = parts.columns[1]
        self.data = merged[merged[part_col] == split].reset_index(drop=True)
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = Image.open(IMAGES_PATH / row[self.id_col]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor([(row[ATTR] + 1) // 2], dtype=torch.float)
        return image, label


print(f'[INFO] device = {DEVICE}')
train_set = CelebADataset(0, train_transform)
valid_set = CelebADataset(1, eval_transform)
test_set  = CelebADataset(2, eval_transform)
print(f'[INFO] train size = {len(train_set)}, valid size = {len(valid_set)}, test size = {len(test_set)}')
print(f'[INFO] batch size = {TR_BS} -> {len(train_set) // TR_BS} batches per epoch')

train_loader = DataLoader(train_set, batch_size=TR_BS, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=TEST_BS, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=TEST_BS, num_workers=4, pin_memory=True)

model = tv.models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(512, 1)
model.to(DEVICE)
print('[INFO] model ready')

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
criterion = nn.BCEWithLogitsLoss()
best_acc = 0.0
wait = 0

for epoch in range(MAX_EPOCHS):
    print(f'[INFO] === epoch {epoch} ===')
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 50 == 0:
            elapsed = time.time() - epoch_start
            rate = (batch_idx + 1) * TR_BS / elapsed
            print(f'[INFO] epoch {epoch} batch {batch_idx}/{len(train_loader)} '
                  f'| loss {running_loss / (batch_idx + 1):.4f} '
                  f'| {rate:.1f} img/s | {elapsed:.0f}s')
    print(f'[INFO] epoch {epoch} training done in {time.time() - epoch_start:.0f}s')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            preds = (model(images.to(DEVICE)).squeeze(1) > 0).cpu()
            correct += (preds.unsqueeze(1) == labels.bool()).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'[INFO] epoch {epoch} val acc = {accuracy:.4f}')

    if accuracy > best_acc:
        best_acc = accuracy
        wait = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f'[INFO] new best {best_acc:.4f} -> saved')
    else:
        wait += 1
        print(f'[INFO] no improvement ({wait}/{PATIENCE})')
        if wait >= PATIENCE:
            print(f'[INFO] early stop at epoch {epoch}, best {best_acc:.4f}')
            break

print('[INFO] loading best checkpoint for test evaluation')
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

true_positive = 0
true_negative = 0
positives = 0
negatives = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = (model(images.to(DEVICE)).squeeze(1) > 0).cpu().unsqueeze(1)
        labels = labels.bool()
        true_positive += (preds & labels).sum().item()
        true_negative += (~preds & ~labels).sum().item()
        positives += labels.sum().item()
        negatives += (~labels).sum().item()

sensitivity = true_positive / positives
specificity = true_negative / negatives
balanced_accuracy = (sensitivity + specificity) / 2
plain_accuracy = (true_positive + true_negative) / (positives + negatives)
print(f'[INFO] test plain acc    = {plain_accuracy:.4f}')
print(f'[INFO] test balanced acc = {balanced_accuracy:.4f}  (sens {sensitivity:.4f}, spec {specificity:.4f})')