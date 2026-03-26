import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("D:/ME/university/Fourth Year/Semester2/capston project/SmartMedicalDiagnosis/dataset/archive/dataset.csv")

print("Dataset Shape:", data.shape)

# Fill empty symptom cells
data = data.fillna("")

# Collect all symptoms
symptoms = []

for col in data.columns[1:]:
    symptoms.extend(data[col].unique())

# Remove empty values
symptoms = list(set(symptoms))
symptoms.remove("")

print("Total symptoms:", len(symptoms))

# Create symptom index
symptom_index = {symptom: i for i, symptom in enumerate(symptoms)}

# Create feature matrix
X = np.zeros((len(data), len(symptoms)))

for i in range(len(data)):
    for col in data.columns[1:]:
        symptom = data.iloc[i][col]
        if symptom != "":
            X[i][symptom_index[symptom]] = 1

# Target variable
y = data["Disease"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

pickle.dump(model, open("model/disease_model.pkl", "wb"))
print("Model saved successfully!")

pickle.dump(symptoms, open("model/symptoms_list.pkl", "wb"))
print("Symptoms list saved!")