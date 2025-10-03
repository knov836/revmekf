import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Charger le fichier principal
df = pd.read_csv("corrections_windows_0.csv")

# Assurer qu'on a la cible
if "correction_applied" not in df.columns:
    raise ValueError("La colonne 'correction_applied' doit être présente dans ton CSV")

blocks = ["acc", "gyro", "mag"]
blocks = ["gyro"]
axes = ["x", "y", "z"]

for block in blocks:
    for axis in axes:
        # Colonnes commençant par le bloc et l'axe
        prefix = f"{block}_{axis}_"
        block_cols = [c for c in df.columns if c.startswith(prefix)]
        if not block_cols:
            print(f"Aucun bloc trouvé pour {block}, on passe...")
            continue
    
        df_block = df[["sample", "time", "correction_applied"] + block_cols]
    
        filename = f"corrections_windows_{block}_{axis}.csv"
        df_block.interpolate(method='linear', axis=0, inplace=True)

        # Si des NaN restent au début ou à la fin, on peut compléter par la valeur la plus proche
        df_block.fillna(method='bfill', inplace=True)
        df_block.fillna(method='ffill', inplace=True)
        
        
        df_block.to_csv(filename, index=False)
        print(f"Sauvegardé : {filename}")
    
        X = df_block[block_cols]
        y = df_block["correction_applied"]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    
        clf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        clf.fit(X_train_res, y_train_res)    
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] 
        
        print(f"\n Résultats pour {block}:")
        print(classification_report(y_test, y_pred))