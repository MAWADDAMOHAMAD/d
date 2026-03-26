from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load model and symptoms
model = pickle.load(open("model/disease_model.pkl", "rb"))
symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))

def predict_disease(user_symptoms):

    input_vector = np.zeros(len(symptoms_list))

    for symptom in user_symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])

    return prediction[0]




@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms_list)

@app.route("/predict", methods=["POST"])
def predict():

    symptoms = []

    for i in range(1,8):
        symptom = request.form.get(f"symptom{i}")
        if symptom:
            symptoms.append(symptom)

    disease = predict_disease(symptoms)

    return render_template("result.html", disease=disease, symptoms=symptoms)


if __name__ == "__main__":
    app.run(debug=True)

data = pd.read_csv("dataset/dataset.csv")

symptom_list = [col for col in data.columns if col != "prognosis"]