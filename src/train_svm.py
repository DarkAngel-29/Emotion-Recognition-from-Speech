import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
X = np.load("data/features/X.npy")
y = np.load("data/features/y.npy")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# Flatten
X_flat = X.reshape(X.shape[0], -1)
print("Flattened shape:", X_flat.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

#tarining part run at first time alone and then save the model and load the model for testing part 
"""
svm = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf')
)
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm_model.pkl")
print("Model saved successfully.")
"""

svm = joblib.load("models/svm_model.pkl")
print("Model Loaded successfully.")

#accuracy part

train_acc = svm.score(X_train, y_train)
test_acc = svm.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

#confusion matrix part

y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SVM")
plt.show()