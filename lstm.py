
import pandas as pd
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


files = glob.glob("corrections_windows_0.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.interpolate(method='linear', axis=0, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)


blocks = ["acc", "gyro", "mag"]
axes = ["x", "y", "z"]
timesteps = 40  

seq_features = []
for block in blocks:
    for axis in axes:
        seq_features += [f"{block}_{axis}_{i}" for i in range(timesteps)]

X = df[seq_features].values
y = df["correction_applied"].values

# features_par_timestep = 9 (acc+gyro+mag in xyz)
X = X.reshape(len(df), timesteps, 9)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Split train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # out: (batch, timesteps, hidden)
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(hn[-1])  # dernier état caché
        out = self.fc(out)
        return out

input_dim = 9       # acc/gyro/mag (x,y,z)
hidden_dim = 64
num_layers = 2
output_dim = 2      # binary: correction_applied yes/no

model = LSTMClassifier(input_dim, hidden_dim, num_layers, output_dim)


#criterion = nn.CrossEntropyLoss()
class_counts = np.bincount(y)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)  
weights = weights / weights.sum()   # normalisation
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

print("📊 Confusion matrix :")
print(confusion_matrix(y_true, y_pred))
print("\n📊 Classification report :")
print(classification_report(y_true, y_pred))
print(f"ROC-AUC : {roc_auc_score(y_true, y_proba):.3f}")


torch.save(model.state_dict(), "rnn_model.pth")
print("✅ Modèle sauvegardé dans rnn_model.pth")

