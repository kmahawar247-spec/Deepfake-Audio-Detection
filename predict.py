"""
predict.py  — FIXED (architecture matches original train_pipeline.py)
----------------------------------------------------------------------
Run:  python predict.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  AudioCNN — MUST EXACTLY MATCH the one used during training
#  (no BatchNorm — same as original train_pipeline.py)
# ─────────────────────────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────
#  Load saved model
# ─────────────────────────────────────────────────────────────────
def load_model(model_path="best_model.pth"):
    if not os.path.exists(model_path):
        print(f"ERROR: '{model_path}' not found in {os.getcwd()}")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model loaded from '{model_path}'  →  device: {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────
#  EER
# ─────────────────────────────────────────────────────────────────
def compute_eer(y_true, scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ─────────────────────────────────────────────────────────────────
#  Batched inference (safe for large test sets)
# ─────────────────────────────────────────────────────────────────
def run_inference(model, device, X_test, batch_size=128):
    all_probs, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i : i + batch_size]
            x      = torch.tensor(batch[:, np.newaxis, :, :], dtype=torch.float32).to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
    return np.concatenate(all_probs), np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────
#  Full evaluation on test set
# ─────────────────────────────────────────────────────────────────
def evaluate():
    from sklearn.metrics import (
        accuracy_score, f1_score,
        confusion_matrix, classification_report
    )

    print("\n" + "=" * 52)
    print("   Deepfake Audio Detector — Evaluation Report")
    print("=" * 52)

    for fname in ["X_test.npy", "y_test.npy"]:
        if not os.path.exists(fname):
            print(f"\nERROR: '{fname}' not found. Run feature_extraction.py first.")
            sys.exit(1)

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    print(f"\n  Test samples : {len(y_test)}")
    print(f"  Real  (0)    : {(y_test == 0).sum()}")
    print(f"  Fake  (1)    : {(y_test == 1).sum()}")
    print(f"  Shape        : {X_test.shape}\n")

    model, device = load_model()

    print("  Running inference...")
    probs, preds = run_inference(model, device, X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='binary')
    eer = compute_eer(y_test, probs)
    cm  = confusion_matrix(y_test, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n" + classification_report(y_test, preds, target_names=['Real', 'Fake']))
    print("─" * 52)
    print(f"  Overall Accuracy : {acc*100:.2f}%   (need ≥ 80%)")
    print(f"  F1 Score         : {f1:.4f}      (need ≥ 0.80)")
    print(f"  EER              : {eer*100:.2f}%    (need ≤ 12%)")
    print(f"  Real  accuracy   : {per_class_acc[0]*100:.2f}%   (need ≥ 75%)")
    print(f"  Fake  accuracy   : {per_class_acc[1]*100:.2f}%   (need ≥ 75%)")
    print("─" * 52)

    checks = [
        ("Overall Accuracy ≥ 80%", acc >= 0.80),
        ("F1 Score ≥ 0.80",        f1  >= 0.80),
        ("EER ≤ 12%",              eer <= 0.12),
        ("Real accuracy ≥ 75%",    per_class_acc[0] >= 0.75),
        ("Fake accuracy ≥ 75%",    per_class_acc[1] >= 0.75),
    ]
    print("\n  Verification checklist:")
    all_pass = True
    for name, passed in checks:
        if not passed: all_pass = False
        print(f"    [{'PASS' if passed else 'FAIL'}]  {name}")

    print(f"\n  {'All checks PASSED! Submission ready.' if all_pass else 'Some checks FAILED.'}")

    # Confusion matrix
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real','Fake'],
                    yticklabels=['Real','Fake'], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("  Confusion matrix saved → confusion_matrix.png")
        plt.close(fig)
    except Exception as e:
        print(f"  (Plot skipped: {e})")


# ─────────────────────────────────────────────────────────────────
#  Single-file inference  (for Streamlit app)
# ─────────────────────────────────────────────────────────────────
def predict_file(wav_path: str):
    import librosa
    SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, MAX_LEN = 16000, 3, 128, 512, 128

    y_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    target = SAMPLE_RATE * DURATION
    y_audio = np.pad(y_audio, (0, max(0, target-len(y_audio))))[:target]

    mel    = librosa.feature.melspectrogram(y=y_audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.pad(mel_db, ((0,0),(0, max(0, MAX_LEN-mel_db.shape[1]))))[:, :MAX_LEN]
    feat   = mel_db.astype(np.float32)

    model, device = load_model()
    x = torch.tensor(feat[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    is_fake    = bool(probs[1] > 0.5)
    return ("Fake (AI-Generated)" if is_fake else "Real (Human)"), float(max(probs))


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()