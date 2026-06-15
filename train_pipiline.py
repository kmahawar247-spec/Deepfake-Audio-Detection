import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

X = np.load("X_train.npy")     
y = np.load("y_train.npy")

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

def to_tensor(arr, labels):
    t = torch.tensor(arr[:, None, :, :], dtype=torch.float32)
    l = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(t, l)

train_ds = to_tensor(X_tr, y_tr)
val_ds   = to_tensor(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=w_tensor)

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = AudioCNN().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

best_val_acc = 0
for epoch in range(20):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
    val_acc = correct / total
    print(f"Epoch {epoch+1:02d} | val acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("  ✓ saved best model")
    sched.step()

print(f"\nBest validation accuracy: {best_val_acc:.4f}")