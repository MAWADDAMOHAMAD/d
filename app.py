from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# ==============================
# Database
# ==============================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ==============================
# Models
# ==============================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.String(300), nullable=False)

# ==============================
# Load ML Model
# ==============================
model = pickle.load(open("model/disease_model.pkl", "rb"))
symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))

# ==============================
# Severe Diseases
# ==============================
SEVERE_DISEASES = [
    "Heart attack","Paralysis (brain hemorrhage)","Tuberculosis",
    "Hepatitis B","Hepatitis C","Hepatitis D","Hepatitis E",
    "Alcoholic hepatitis","AIDS","Pneumonia","Dengue","Typhoid"
]

# ==============================
# Medicine Data
# ==============================
MEDICINE_DATA = {
    "Pneumonia": {
        "medicines": ["Azithromycin", "Paracetamol"],
        "usage": "Take antibiotics as prescribed. Rest well."
    },
    "Diabetes": {
        "medicines": ["Metformin"],
        "usage": "Monitor sugar levels daily."
    }
}

# ==============================
# Helper: Predict Disease
# ==============================
def predict_disease(user_symptoms):
    input_vector = np.zeros(len(symptoms_list))

    for s in user_symptoms:
        if s in symptoms_list:
            input_vector[symptoms_list.index(s)] = 1

    return model.predict([input_vector])[0]

# ==============================
# HOME (Dashboard)
# ==============================
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    history = History.query.filter_by(username=session["user"]).all()

    return render_template(
        "index.html",
        symptoms=symptoms_list,
        history=history,
        user=session["user"]
    )

# ==============================
# LOGIN
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return "❌ Invalid username or password"

    return render_template("login.html")

# ==============================
# REGISTER
# ==============================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            return "❌ Username already exists"

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

# ==============================
# LOGOUT
# ==============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==============================
# HISTORY PAGE
# ==============================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    history = History.query.filter_by(username=session["user"]).all()

    return render_template(
        "history.html",
        history=history,
        user=session["user"]
    )
    # ==============================
# DELETE ONE ITEM
# ==============================
@app.route("/delete/<int:id>")
def delete_item(id):
    if "user" not in session:
        return redirect("/login")

    item = History.query.get(id)

    if item and item.username == session["user"]:
        db.session.delete(item)
        db.session.commit()

    return redirect("/history")


# ==============================
# DELETE ALL HISTORY
# ==============================
@app.route("/delete_all")
def delete_all():
    if "user" not in session:
        return redirect("/login")

    History.query.filter_by(username=session["user"]).delete()
    db.session.commit()

    return redirect("/history")

# ==============================
# PREDICT
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Not logged in"})

    symptoms = []

    for i in range(1, 8):
        s = request.form.get(f"symptom{i}")
        if s:
            symptoms.append(s)

    disease = predict_disease(symptoms)

    # Save history
    new_record = History(
        username=session["user"],
        disease=disease,
        symptoms=", ".join(symptoms)
    )

    db.session.add(new_record)
    db.session.commit()

    warning = None
    if disease in SEVERE_DISEASES:
        warning = "⚠️ Serious condition detected!"

    medicine = MEDICINE_DATA.get(disease, {
        "medicines": ["Consult a doctor"],
        "usage": "No data available"
    })

    return jsonify({
        "disease": disease,
        "warning": warning,
        "medicines": medicine["medicines"],
        "usage": medicine["usage"]
    })

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)