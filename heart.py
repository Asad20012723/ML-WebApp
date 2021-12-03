from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_csv(
    r"C:\Users\ASAD\Desktop\Health project\ML-MT-WebApp-master\heart.csv")
data["trestbps"] = np.log(data["trestbps"])

data = data.drop(["fbs"], axis=1)
data = data.drop(["ca"], axis=1)
data["chol"] = np.log(data["chol"])
target = data["target"]
print(data.shape[1])

np.random.shuffle(data.values)
data = data.drop(["target"], axis=1)
print(data.columns)
sc = StandardScaler()
data = sc.fit_transform(data)

lr = LogisticRegression()
lr.fit(data, target)
cv_results = cross_validate(lr, data, target, cv=10)
print(cv_results)
joblib.dump(lr, "model2")
