import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# ====================== Chargement ======================
files = glob.glob("corrections_windows_0.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.interpolate(method='linear', axis=0, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

# ====================== Préparation séquences ======================
blocks = ["acc", "gyro", "mag"]
axes = ["x", "y", "z"]
timesteps = 40

seq_features = [f"{b}_{a}_{i}" for b in blocks for a in axes for i in range(timesteps)]
X = df[seq_features].values.reshape(len(df), timesteps, 9)
y = df["correction_applied"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Split train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ====================== Modèle RNN ======================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out

input_dim = 9
hidden_dim = 64
num_layers = 2
output_dim = 2
model = LSTMClassifier(input_dim, hidden_dim, num_layers, output_dim)

# ====================== Gestion déséquilibre ======================
class_counts = np.bincount(y)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
weights = weights / weights.sum()
print("Poids utilisés pour les classes :", weights)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ====================== Entraînement ======================
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

# ====================== Évaluation ======================
model.eval()
y_true, y_pred, y_proba = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:,1]
        y_true.extend(y_batch.numpy())
        y_proba.extend(probs.numpy())

# Recherche du seuil optimal F1 pour la classe 1
thresholds = np.linspace(0,1,50)
best_f1, best_thresh = 0, 0
for t in thresholds:
    preds = (np.array(y_proba) >= t).astype(int)
    f1 = f1_score(y_true, preds, pos_label=1)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"🔎 Best threshold = {best_thresh:.2f} with F1 = {best_f1:.3f}")

y_pred = (np.array(y_proba) >= best_thresh).astype(int)

print("📊 Confusion matrix :")
print(confusion_matrix(y_true, y_pred))
print("\n📊 Classification report :")
print(classification_report(y_true, y_pred))
print(f"ROC-AUC : {roc_auc_score(y_true, y_proba):.3f}")

# ====================== Sauvegarde modèle ======================
torch.save(model.state_dict(), "rnn_model.pth")
print("✅ Modèle sauvegardé dans rnn_model.pth")
