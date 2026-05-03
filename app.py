from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "secret123"

# ==============================
# Session
# ==============================
app.permanent_session_lifetime = timedelta(days=7)

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
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(255))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    disease = db.Column(db.String(100))
    symptoms = db.Column(db.String(300))

# ==============================
# ML Model
# ==============================
model = pickle.load(open("model/disease_model.pkl", "rb"))
symptoms_list = pickle.load(open("model/symptoms_list.pkl", "rb"))

# ==============================
# Data
# ==============================
description_df = pd.read_csv('dataset/archive/symptom_Description.csv')
description_df.columns = description_df.columns.str.strip()

precaution_df = pd.read_csv('dataset/archive/symptom_precaution.csv')
precaution_df.columns = precaution_df.columns.str.strip()

# ==============================
# Severe Diseases
# ==============================
SEVERE_DISEASES = [
    "heart attack","paralysis (brain hemorrhage)","tuberculosis",
    "hepatitis b","hepatitis c","hepatitis d","hepatitis e",
    "alcoholic hepatitis","aids","pneumonia","dengue","typhoid"
]

# ==============================
# Helpers
# ==============================
def get_description(disease):
    result = description_df[description_df['Disease'] == disease]
    if not result.empty:
        return result['Description'].values[0]
    return "No description available."

def get_precautions(disease):
    result = precaution_df[precaution_df['Disease'] == disease]
    if not result.empty:
        return result.iloc[0, 1:].dropna().tolist()
    return ["Consult a doctor"]

def predict_disease(symptoms):
    vector = np.zeros(len(symptoms_list))
    for s in symptoms:
        if s in symptoms_list:
            vector[symptoms_list.index(s)] = 1
    return model.predict([vector])[0]

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("about.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ✅ FIXED (was missing → caused error)
@app.route("/try-system")
def try_system():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
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
# AUTH
# ==============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name'].strip()
        last_name = request.form['last_name'].strip()
        email = request.form['email'].strip()
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(email=email).first():
            return "⚠️ Email already exists"

        base_username = (first_name + last_name).lower()
        username = base_username
        counter = 1

        while User.query.filter_by(username=username).first():
            username = f"{base_username}{counter}"
            counter += 1

        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password=password
        )

        db.session.add(new_user)
        db.session.commit()

        session["register_message"] = f"✅ Your username is: {username}"
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    message = session.pop("register_message", None)

    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if user and check_password_hash(user.password, request.form["password"]):
            session["user"] = user.username

            # ✅ ADMIN SUPPORT
            if user.username == "admin":
                return redirect(url_for("admin"))

            return redirect(url_for("dashboard"))

        return "❌ Invalid credentials"

    return render_template("login.html", message=message)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ==============================
# HISTORY (FIXED - was missing)
# ==============================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    history = History.query.filter_by(username=session["user"]).all()

    return render_template("history.html", history=history, user=session["user"])

# ==============================
# DELETE
# ==============================
@app.route("/delete/<int:id>")
def delete_item(id):
    item = History.query.get(id)

    if item and item.username == session.get("user"):
        db.session.delete(item)
        db.session.commit()

    return redirect("/history")

@app.route("/delete_all")
def delete_all():
    History.query.filter_by(username=session.get("user")).delete()
    db.session.commit()
    return redirect("/history")

# ==============================
# ADMIN (FULLY WORKING)
# ==============================
@app.route('/admin')
def admin():
    if session.get('user') != 'admin':
        return "Access denied"

    total_users = User.query.count()
    total_history = History.query.count()

    top = db.session.query(
        History.disease,
        db.func.count(History.disease)
    ).group_by(History.disease).order_by(db.func.count(History.disease).desc()).first()

    top_disease = top[0] if top else "N/A"

    data = db.session.query(
        User.first_name,
        User.last_name,
        User.email,
        History.symptoms,
        History.disease
    ).join(History, User.username == History.username).all()

    return render_template(
        "admin.html",
        data=data,
        total_users=total_users,
        total_history=total_history,
        top_disease=top_disease
    )

# ==============================
# PREDICTION
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return jsonify({"error": "Not logged in"}), 401

    symptoms = []
    for i in range(1, 8):
        s = request.form.get(f"symptom{i}")
        if s:
            symptoms.append(s)

    disease = str(predict_disease(symptoms))
    disease_clean = disease.lower().strip()

    description = get_description(disease)
    precautions = get_precautions(disease)

    warning = None
    show_hospital_button = False

    if disease_clean in SEVERE_DISEASES:
        warning = "⚠️ Serious condition! Please seek immediate medical attention."
        show_hospital_button = True

    # Save history
    new_history = History(
        username=session["user"],
        disease=disease,
        symptoms=", ".join(symptoms)
    )
    db.session.add(new_history)
    db.session.commit()

    return jsonify({
        "disease": disease,
        "description": description,
        "warning": warning,
        "show_hospital_button": show_hospital_button,
        "precautions": precautions
    })

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)