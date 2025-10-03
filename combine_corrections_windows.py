import pandas as pd
import glob

# Récupérer tous les fichiers CSV
files = glob.glob("corrections_windows*.csv")

print("👉 Fichiers trouvés :", files)

dfs = []
for f in files:
    df = pd.read_csv(f)
    # On garde les colonnes de features + sample/time/correction_applied
    dfs.append(df)

# Vérifier que sample/time/correction_applied sont bien présents partout
base_cols = ["sample", "time"]

# Concaténer par colonnes, en alignant sur l’index
df_final = pd.concat([df.set_index(base_cols) for df in dfs], axis=1)

# Supprimer les éventuels doublons de colonnes
df_final = df_final.loc[:, ~df_final.columns.duplicated()]

# Remettre sample/time/correction_applied en colonnes normales
df_final.reset_index(inplace=True)

# Sauvegarde
df_final.to_csv("corrections_windows_combined.csv", index=False)

print("✅ Fichier combiné : corrections_windows_combined.csv")