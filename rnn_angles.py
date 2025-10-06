
import pandas as pd
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.signal import savgol_filter


#files = glob.glob("corrections_windows_0.csv")
#files = glob.glob("corrections_windows_20251005_1852230.csv")

#files = glob.glob("corrections_windows_20251005_2339290.csv")
files = glob.glob("corrections_windows_angles_20251006_1603280.csv")


#files = glob.glob("corrections_windows_20251006_1029030.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.interpolate(method='linear', axis=0, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)


blocks = ["normal","acc", "gyro", "mag"]
angles = ["t0","t2", "t3", "t4","et0","et2", "et3", "et4",]
axes = ["x", "y", "z"]
timesteps = 20  

seq_features = []
extra_features=[]
for block in blocks:
    for axis in axes:
        seq_features += [f"{block}_{axis}_{i}" for i in range(timesteps)]
for angle in angles:
        seq_features += [f"{angle}_{i}" for i in range(timesteps)]


for block in blocks:
    #print([c for ax in axes for c in df.columns if c.startswith(f"normal_{ax}_") ])
    
    for axis in axes:
        cols = [c for c in df.columns if c.startswith(f"{block}_{axis}_")]
        print(df[cols])
        print(cols)
        smoothed_cols = savgol_filter(np.array(df[cols]), 5, 2)
        df[f"s{block}_{axis}_mean"] = np.mean(savgol_filter(np.array(df[cols]), 20, 2),axis=1)

        # Moyenne, std, min, max
        df[f"{block}_{axis}_mean"] = df[cols].mean(axis=1)
        df[f"{block}_{axis}_std"] = df[cols].std(axis=1)
        df[f"{block}_{axis}_min"] = df[cols].min(axis=1)
        df[f"{block}_{axis}_max"] = df[cols].max(axis=1)
        """feature_cols += [f"{block}_{axis}_mean", f"{block}_{axis}_std",
                         f"{block}_{axis}_min", f"{block}_{axis}_max"]
        """
        extra_features += [f"{block}_{axis}_std"]
    # Norme of vector
    if block == "normal":
        continue
    normal = df[[f"normal_{ax}_mean" for ax in axes]]
    df[f"{block}_norm"] = np.sqrt(df[[f"{block}_{ax}_mean" for ax in axes]].pow(2).sum(axis=1))
    df[f"{block}_norm_crossnormal"] = np.sqrt((np.cross(normal,df[[f"{block}_{ax}_mean" for ax in axes]])**2).sum(axis=1))
    
    extra_features+= [f"{block}_norm_crossnormal"]
    

for block in angles:
    cols = [c for c in df.columns if c.startswith(f"{block}_")]
    df[f"{block}_mean"] = df[cols].mean(axis=1)
    df[f"{block}_std"] = df[cols].std(axis=1)
    df[f"{block}_min"] = df[cols].min(axis=1)
    df[f"{block}_max"] = df[cols].max(axis=1)
    extra_features += [f"{block}_mean", f"{block}_std",
                     f"{block}_min", f"{block}_max"]

#seq_features += [f"normal_{a}" for a in {"z"} for i in range(timesteps)]
seq_features = seq_features+extra_features
input_dim = 12+8+len(extra_features)
#X = df[seq_features].values.reshape(len(df), timesteps, input_dim)

X_seq = df[[f for f in seq_features if f not in extra_features]].values
X_extra = df[extra_features].values  # shape = (n_samples, n_extra)

X_extra_seq = np.repeat(X_extra[:, np.newaxis, :], timesteps, axis=1)

X = np.concatenate([X_seq.reshape(len(df), timesteps, 12+8), X_extra_seq], axis=2)

y = df["correction_applied"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Split train/test
"""train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])"""
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader

# y_tensor = labels (torch tensor) que tu as déjà
y_numpy = y_tensor.numpy()

# Stratified split 80/20
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(y_numpy)), y_numpy))

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh', dropout=0.1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # out: (batch, timesteps, hidden)
        out, hn = self.rnn(x)       # hn = dernier état caché (num_layers, batch, hidden)
        out = self.dropout(hn[-1])  # on prend le dernier état caché de la dernière couche
        out = self.fc(out)
        return out

#input_dim = 9       # acc/gyro/mag (x,y,z)
hidden_dim = 128
num_layers = 3
output_dim = 2      # binary: correction_applied yes/no

model = RNNClassifier(input_dim, hidden_dim, num_layers, output_dim)


#criterion = nn.CrossEntropyLoss()
class_counts = np.bincount(y)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)  
weights = weights / weights.sum()   # normalisation
criterion = nn.CrossEntropyLoss(weight=weights)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-5)  # plus petit
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss={total_loss/len(train_loader):.4f}")


model.eval()
y_true, y_pred, y_proba = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:,1]
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.numpy())
        y_proba.extend(probs.numpy())

threshold = 0.5
y_pred = (np.array(y_proba) >= threshold).astype(int)

print("Confusion matrix :")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report :")
print(classification_report(y_true, y_pred))
print(f"ROC-AUC : {roc_auc_score(y_true, y_proba):.3f}")


#torch.save(model.state_dict(), "lstm_model.pth")
torch.save(model, "rnn_model.pth")
model = torch.load("rnn_model.pth",weights_only=False)
model.eval()

with torch.no_grad():
    outputs = model(X_tensor)  
    probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  
    print(torch.softmax(outputs, dim=1)[:, :])
    preds_threshold = (probs >= threshold).astype(int)
    print(preds_threshold)

# Ajouter les résultats au DataFrame pour inspection
df["predicted_class_threshold"] = preds_threshold
df["predicted_proba_class1"] = probs
print(df[["sample", "time", "predicted_class_threshold", "predicted_proba_class1"]][20:40])
df.to_csv("predictions_results.csv", index=False)
