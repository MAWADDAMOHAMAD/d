import pandas as pd

data = pd.read_csv(r"D:\ME\university\Fourth Year\Semester2\capston project\SmartMedicalDiagnosis\dataset\archive\dataset.csv")
symptoms = []

for column in data.columns:
    if column != "prognosis":
        symptoms.append(column)

print(symptoms)



data = pd.read_csv(r"D:\ME\university\Fourth Year\Semester2\capston project\SmartMedicalDiagnosis\dataset\archive\dataset.csv")

# ✅ FIX disease names (VERY IMPORTANT)
data["Disease"] = data["Disease"].str.strip()

# Get all unique diseases
print(data["Disease"].unique())