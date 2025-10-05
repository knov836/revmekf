import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from scipy.signal import savgol_filter
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


#files = glob.glob("corrections_windows_0.csv")
files = glob.glob("corrections_windows_20251005_1852230.csv")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

df.interpolate(method='linear', axis=0, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

blocks = ["normal","acc", "gyro", "mag"]
axes = ["x", "y", "z"]
feature_cols = []
#feature_cols +=[f"normal_{ax}" for ax in axes]
for block in blocks:
    #print([c for ax in axes for c in df.columns if c.startswith(f"normal_{ax}_") ])
    
    
    for axis in axes:
        cols = [c for c in df.columns if c.startswith(f"{block}_{axis}_")]
        print(cols)
        smoothed_cols = savgol_filter(np.array(df[cols]), 5, 2)
        df[f"s{block}_{axis}_mean"] = np.mean(savgol_filter(np.array(df[cols]), 20, 2),axis=1)
        #feature_cols += cols
        #df[f"{block}_{axis}_mean"]=0
        for ind in [20]:
            df[f"{block}_{axis}_deriv_{ind}"] = np.diff(smoothed_cols,axis=1)[:,ind]
            df[f"{block}_{axis}_dderiv_{ind}"] = np.diff(np.diff(smoothed_cols,axis=1),axis=1)[:,ind]
            #feature_cols += [f"{block}_{axis}_deriv_{ind}",f"{block}_{axis}_dderiv_{ind}"]
            
        df[f"{block}_{axis}_deriv_mean"] = np.mean(np.diff(smoothed_cols,axis=1)[:,10:30],axis=1)
        df[f"{block}_{axis}_dderiv_mean"] = np.mean(np.diff(np.diff(smoothed_cols,axis=1),axis=1)[:,10:30],axis=1)
        feature_cols += [f"{block}_{axis}_deriv_mean",f"{block}_{axis}_dderiv_mean"]
        # Moyenne, std, min, max
        df[f"{block}_{axis}_mean"] = df[cols].mean(axis=1)
        df[f"{block}_{axis}_std"] = df[cols].std(axis=1)
        df[f"{block}_{axis}_min"] = df[cols].min(axis=1)
        df[f"{block}_{axis}_max"] = df[cols].max(axis=1)
        """feature_cols += [f"{block}_{axis}_mean", f"{block}_{axis}_std",
                         f"{block}_{axis}_min", f"{block}_{axis}_max"]
        """
        feature_cols += [f"{block}_{axis}_std"]
    # Norme of vector
    if block != "normal":
        df[f"{block}_norm"] = np.sqrt(df[[f"{block}_{ax}_mean" for ax in axes]].pow(2).sum(axis=1))
        normal = df[[f"normal_{ax}_mean" for ax in axes]]
        df[f"{block}_norm_crossnormal"] = np.sqrt((np.cross(normal,df[[f"{block}_{ax}_mean" for ax in axes]])**2).sum(axis=1))
        
        feature_cols+= [f"{block}_norm",f"{block}_norm_crossnormal"]
        
        # Differences between axes
        df[f"{block}_xy_diff"] = df[f"{block}_x_mean"] - df[f"{block}_y_mean"]
        df[f"{block}_xz_diff"] = df[f"{block}_x_mean"] - df[f"{block}_z_mean"]
        df[f"{block}_yz_diff"] = df[f"{block}_y_mean"] - df[f"{block}_z_mean"]
        feature_cols += [f"{block}_xy_diff", f"{block}_xz_diff", f"{block}_yz_diff"]
    
df[f"sacc_norm_xy"] = np.sqrt(df[[f"sacc_{ax}_mean" for ax in ["x","y"]]].pow(2).sum(axis=1))
feature_cols.append(f"sacc_norm_xy")

"""df[f"smag_norm_xy"] = np.sqrt(df[[f"smag_{ax}_mean" for ax in ["x","y"]]].pow(2).sum(axis=1))
feature_cols.append(f"smag_norm_xy")"""
#train/test
X = df[feature_cols]
y = df["correction_applied"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#Oversampling 
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

#global RandomForest training
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
clf.fit(X_train_res, y_train_res)

# Predictions and probabilities
"""y_proba = clf.predict_proba(X_test)[:,1]
threshold = 0.5  
y_pred = (y_proba >= threshold).astype(int)"""
y_proba = clf.predict_proba(X_test)[:,1]
threshold = 0.5  # plus bas pour détecter plus de 1
y_pred = (y_proba >= threshold).astype(int)
#Evaluation
print("📊 Confusion matrix :")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Classification report :")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\n📊 ROC-AUC : {roc_auc:.3f}")

#Importance of features


importances = clf.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values(by='importance', ascending=False)
print("\nTop 20 features :")
print(feat_imp.head(20))

plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'].head(20), feat_imp['importance'].head(20))
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 20 features to predict correction_applied")
plt.show()


sample = df[df["correction_applied"] == 1].iloc[10]

top_features = feat_imp.head(10)['feature'].tolist()  # top 10 features
print("Top features for this sample :")
for f in top_features:
    print(f"{f}: {sample[f]}")
    
