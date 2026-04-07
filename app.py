from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Load model and symptoms
# ==============================
model = pickle.load(open("model/disease_model.pkl", "rb"))
symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))

# ==============================
# ⚠️ Severe Diseases List
# ==============================
SEVERE_DISEASES = [
    "Heart attack",
    "Paralysis (brain hemorrhage)",
    "Tuberculosis",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "AIDS",
    "Pneumonia",
    "Dengue",
    "Typhoid"
]

# ==============================
# Prediction Function
# ==============================
def predict_disease(user_symptoms):

    input_vector = np.zeros(len(symptoms_list))

    for symptom in user_symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])

    return prediction[0]

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms_list)


@app.route("/predict", methods=["POST"])
def predict():

    symptoms = []

    for i in range(1, 8):
        symptom = request.form.get(f"symptom{i}")
        if symptom:
            symptoms.append(symptom)

    # Predict disease
    disease = predict_disease(symptoms)

    # ==============================
    # ⚠️ Warning System
    # ==============================
    warning_message = None

    if disease in SEVERE_DISEASES:
        warning_message = "⚠️ This condition may be serious. Please consult a doctor immediately."

    return render_template(
        "result.html",
        disease=disease,
        symptoms=symptoms,
        warning=warning_message
    )

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)