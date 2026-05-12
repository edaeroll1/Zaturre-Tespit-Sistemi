from pathlib import Path
from PIL import Image, UnidentifiedImageError
from collections import Counter

import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ── Sabitler ────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\WinOn\Desktop\data\raw\Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia")
SAVE_DIR     = Path(__file__).parent

IMG_SIZE   = 224
BATCH_SIZE = 16
VAL_RATIO  = 0.2
SEED       = 42
NUM_EPOCHS = 5
PATIENCE   = 3
MEAN       = 0.5099
STD        = 0.2546

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", DEVICE)

# ── Yardımcılar ──────────────────────────────────────────────────────────────
def find_dir(parent, name):
    for p in Path(parent).iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    raise FileNotFoundError(f"{parent} içinde '{name}' bulunamadı")

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in exts])

# ── Klasörler ────────────────────────────────────────────────────────────────
train_dir = find_dir(DATASET_ROOT, "Training")
test_dir  = find_dir(DATASET_ROOT, "Testing")

# ── Dataset ──────────────────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.transform = transform
        normal_dir    = find_dir(split_dir, "normal")
        pneumonia_dir = find_dir(split_dir, "pneumonia")
        self.samples  = (
            [(p, 0) for p in list_images(normal_dir)] +
            [(p, 1) for p in list_images(pneumonia_dir)]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

class SubsetWithTransform(Dataset):
    def __init__(self, base, indices):
        self.base, self.indices = base, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
])

full_train = ChestXrayDataset(train_dir, train_tf)
full_eval  = ChestXrayDataset(train_dir, eval_tf)
test_ds    = ChestXrayDataset(test_dir,  eval_tf)

val_size   = int(len(full_train) * VAL_RATIO)
train_size = len(full_train) - val_size
gen        = torch.Generator().manual_seed(SEED)
t_split, v_split = random_split(full_train, [train_size, val_size], generator=gen)

train_ds = SubsetWithTransform(full_train, t_split.indices)
val_ds   = SubsetWithTransform(full_eval,  v_split.indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# ── Class weights ─────────────────────────────────────────────────────────────
counts  = Counter(train_ds[i][1] for i in range(len(train_ds)))
n_total = len(train_ds)
class_weights = torch.tensor(
    [n_total / (2 * counts[c]) for c in [0, 1]], dtype=torch.float
).to(DEVICE)

# ── Model tanımları ──────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc    = nn.Linear(m.fc.in_features, 2)
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    c = m.features[0][0]
    m.features[0][0] = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    m.classifier[1]  = nn.Linear(m.classifier[1].in_features, 2)
    return m

MODELS = {
    "Custom CNN":      CustomCNN(),
    "ResNet18":        build_resnet18(),
    "EfficientNet-B0": build_efficientnet_b0(),
}

# ── Eğitim ───────────────────────────────────────────────────────────────────
def train(model, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc, best_w, patience_cnt = 0, copy.deepcopy(model.state_dict()), 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            correct  += (out.argmax(1) == lbls).sum().item()
            total    += lbls.size(0)
            loss_sum += loss.item() * lbls.size(0)

        model.eval()
        vc, vt, vl = 0, 0, 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, lbls)
                vc  += (out.argmax(1) == lbls).sum().item()
                vt  += lbls.size(0)
                vl  += loss.item() * lbls.size(0)

        val_acc = vc / vt
        print(f"{name} | Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_w, patience_cnt = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping")
            break

    model.load_state_dict(best_w)
    return model

# ── Ana döngü ────────────────────────────────────────────────────────────────
best_f1, best_name, best_model_obj = 0, None, None

for name, model in MODELS.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    trained = train(model, name)

    save_path = SAVE_DIR / f"{name}.pth"
    torch.save(trained.state_dict(), save_path)
    print(f"Kaydedildi: {save_path}")

    trained.eval()
    correct, total = 0, 0
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out   = trained(imgs.to(DEVICE))
            preds = out.argmax(1).cpu()
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            all_p.extend(preds.numpy())
            all_l.extend(lbls.numpy())

    p  = np.array(all_p); l = np.array(all_l)
    tp = ((p==1)&(l==1)).sum(); fp = ((p==1)&(l==0)).sum(); fn = ((p==0)&(l==1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"Test Accuracy: {correct/total:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1, best_name, best_model_obj = f1, name, trained

    gc.collect()

best_path = SAVE_DIR / "best_model.pth"
torch.save(best_model_obj.state_dict(), best_path)
print(f"\nEn iyi model: {best_name} (F1={best_f1:.4f})")
print(f"Kaydedildi: {best_path}")
