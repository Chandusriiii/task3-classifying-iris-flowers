import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------- Scikit-learn RandomForest Classifier ---------
# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy_sklearn = accuracy_score(y_test, y_pred)
report_sklearn = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Scikit-learn RandomForest Classifier")
print(f"Accuracy: {accuracy_sklearn:.2f}")
print("Classification Report:\n", report_sklearn)

# --------- TensorFlow Neural Network Classifier ---------
# Convert the labels to categorical format
y_categorical = to_categorical(y)

# Split the dataset into training and testing sets
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# Standardize the features
X_train_tf = scaler.fit_transform(X_train_tf)
X_test_tf = scaler.transform(X_test_tf)

# Define the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train_tf.shape[1],)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes for iris dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_tf, y_train_tf, epochs=50, batch_size=5, validation_split=0.2, verbose=0)

# Evaluate the model
loss, accuracy_tf = model.evaluate(X_test_tf, y_test_tf, verbose=0)
print("\nTensorFlow Neural Network Classifier")
print(f"Test Accuracy: {accuracy_tf:.2f}")

# Make predictions
y_pred_tf = model.predict(X_test_tf)
y_pred_classes_tf = np.argmax(y_pred_tf, axis=1)
y_true_classes_tf = np.argmax(y_test_tf, axis=1)

# Classification report
report_tf = classification_report(y_true_classes_tf, y_pred_classes_tf, target_names=iris.target_names)
print("Classification Report:\n", report_tf)