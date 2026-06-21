import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/landmarks")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "y_test.npy", y_test)

joblib.dump(scaler, DATA_DIR / "scaler.pkl")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)