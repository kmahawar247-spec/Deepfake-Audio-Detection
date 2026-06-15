import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION    = 3       
N_MELS      = 128
HOP_LENGTH  = 512
MAX_LEN     = 128     

def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target = SAMPLE_RATE * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y

def extract_melspec(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0, MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]
    return mel_db.astype(np.float32)

def build_dataset(data_dir):
    X, y = [], []
    for label, folder in enumerate(['genuine', 'fake']):
        folder_path = os.path.join(data_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Loading {len(files)} files from {folder}...")
        for fname in files:
            try:
                audio = load_audio(os.path.join(folder_path, fname))
                feat  = extract_melspec(audio)
                X.append(feat)
                y.append(label)   # 0=genuine, 1=fake
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X_train, y_train = build_dataset("LA_norm/train")
    X_test,  y_test  = build_dataset("LA_norm/test")
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy",  X_test)
    np.save("y_test.npy",  y_test)
    print("Saved! Shapes:", X_train.shape, X_test.shape)
