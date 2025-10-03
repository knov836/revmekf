#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Exemple: série aléatoire
#np.random.seed(42)
y = np.sin(1/3*np.linspace(0, 20, 200)) + 0.2 * np.random.randn(200)

# Fenêtrage (lag features)
p = 5
data = pd.DataFrame({
    f"lag_{i}": y[p-i:len(y)-i] for i in range(1, p+1)
})
data["target"] = y[p:]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data.drop(columns="target")
y_target = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print("Score R²:", rf.score(X_test, y_test))

import matplotlib.pyplot as plt

importances = rf.feature_importances_
plt.bar(X.columns, importances)
plt.title("Importance des décalages")
plt.show()


from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(rf, X, features=[0,1,2])
plt.show()
